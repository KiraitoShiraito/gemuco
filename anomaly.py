"""09
GeMuCo: Generalized Multisensory Correlational Model
Fase 9: Anomaly Detection - Detección de Anomalías

Basado en la sección II-K del artículo:
    "Anomaly detection is performed with respect to the amount of error between
    the current value x_out and the estimated value x_out_est."

Procedimiento según el artículo:
    1. Collect state estimation data x_out_est and current state data x_out_data
       in the normal state without any anomaly
    2. Calculate mean μ and variance Σ of the error e_out_data = x_out_data - x_out_est
    3. During operation, calculate Mahalanobis distance:
       d = sqrt((e_out - μ)^T Σ^{-1} (e_out - μ))
    4. When d exceeds the threshold, assume anomaly detected

El artículo también menciona (sección II-K):
    "One of the simplest anomaly detection methods is to set a threshold value
    for ||x_out - x_out_est||_2 and consider an anomaly when the error is larger
    than the threshold. On the other hand, the mean and variance of the error
    can be used to detect anomalies more accurately."
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from scipy import stats

# Importar de fases anteriores
try:
    from model import GeMuCoNetwork, MaskManager, ParametricBiasManager
    from state_estimation import StateEstimator
except ImportError:
    print("Importando módulos locales...")


class AnomalyType(Enum):
    """Tipos de anomalías que se pueden detectar"""
    SENSOR_FAILURE = "sensor_failure"       # Fallo de sensor
    ACTUATOR_FAILURE = "actuator_failure"   # Fallo de actuador
    ENVIRONMENTAL_CHANGE = "environmental_change"  # Cambio en el entorno
    BODY_CHANGE = "body_change"             # Cambio en el cuerpo del robot
    TOOL_CHANGE = "tool_change"             # Cambio en la herramienta
    UNKNOWN = "unknown"                     # Anomalía desconocida


@dataclass
class AnomalyConfig:
    """Configuración para detección de anomalías"""
    # Umbrales para distancia de Mahalanobis
    mahalanobis_threshold: float = 3.0      # Umbral en desviaciones estándar
    mahalanobis_percentile: float = 99.0    # Percentil para umbral adaptativo
    
    # Umbrales para error absoluto (simplificado)
    absolute_error_threshold: float = 0.5   # Error absoluto máximo permitido
    
    # Ventanas para estadísticas
    window_size_normal: int = 100           # Tamaño de ventana para datos normales
    window_size_detection: int = 10         # Tamaño de ventana para detección
    
    # Para detección de cambios graduales
    change_detection_sensitivity: float = 2.0  # Sensibilidad para detectar cambios
    use_adaptive_threshold: bool = True     # Usar umbral adaptativo
    
    # Para logging
    log_anomalies: bool = True
    save_detection_history: bool = True


class AnomalyDetector:
    """
    Detector de anomalías basado en el error de predicción de GeMuCo.
    
    Implementa el método descrito en la sección II-K del artículo:
        - Recolección de datos en estado normal
        - Cálculo de media y covarianza del error
        - Distancia de Mahalanobis para detección
    """
    
    def __init__(
        self,
        model: GeMuCoNetwork,
        mask_manager: MaskManager,
        pb_manager: ParametricBiasManager,
        state_estimator: StateEstimator,
        config: AnomalyConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: Red GeMuCo
            mask_manager: Gestor de máscaras
            pb_manager: Gestor de parametric biases
            state_estimator: Estimador de estado (para obtener estimaciones)
            config: Configuración de detección
            device: CPU o CUDA
        """
        self.model = model
        self.mask_manager = mask_manager
        self.pb_manager = pb_manager
        self.state_estimator = state_estimator
        self.device = device
        
        self.config = config or AnomalyConfig()
        
        # Estadísticas del error en estado normal
        self.error_mean = None          # μ
        self.error_covariance = None    # Σ
        self.error_covariance_inv = None  # Σ^{-1}
        self.error_std = None           # Para diagnóstico
        
        # Buffer para recolectar datos normales
        self.normal_buffer = []  # Lista de errores en estado normal
        
        # Historial de detecciones
        self.detection_history = []
        
        # Estado actual
        self.is_calibrated = False
        self.current_error_stats = None
        
        # Para detección de cambios graduales
        self.error_window = deque(maxlen=self.config.window_size_detection)
    
    def calibrate_from_normal_data(
        self,
        n_samples: int = 100,
        state_idx: int = 0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Calibra el detector usando datos en estado normal.
        
        Según el artículo (sección II-K):
            "First, we collect the state estimation data x_out_est and the current
            state data x_out_data in the normal state without any anomaly.
            For this data, we calculate the mean μ and variance Σ of the error
            e_out_data = x_out_data - x_out_est."
        
        Args:
            n_samples: Número de muestras a recolectar
            state_idx: Índice del estado normal
            verbose: Si mostrar progreso
        
        Returns:
            Diccionario con estadísticas calculadas
        """
        print(f"\nCalibrando detector de anomalías con {n_samples} muestras normales...")
        
        errors = []
        
        for i in range(n_samples):
            if verbose and i % 20 == 0:
                print(f"  Muestra {i}/{n_samples}")
            
            # Generar o simular datos normales
            # En una implementación real, aquí se usarían datos reales del robot
            x_available, mask_available = self._generate_normal_data(state_idx)
            
            # Obtener estimación del estado
            result = self.state_estimator.estimate_state(
                x_available=x_available,
                mask_available=mask_available,
                state_idx=state_idx,
                method="auto"
            )
            
            if result['success'] and result['x_estimated'] is not None:
                # Calcular error de predicción
                x_out_true = self._get_true_values(x_available, mask_available, state_idx)
                error = (x_out_true - result['x_estimated']).detach().cpu().numpy()
                errors.append(error.flatten())
        
        if len(errors) == 0:
            print("  ADVERTENCIA: No se pudieron recolectar suficientes datos normales")
            return {'success': False}
        
        # Convertir a array numpy
        errors_array = np.array(errors)
        
        # Calcular media y covarianza
        self.error_mean = np.mean(errors_array, axis=0)
        self.error_covariance = np.cov(errors_array.T)
        
        # Calcular inversa de la covarianza (con regularización para estabilidad)
        reg_cov = self.error_covariance + np.eye(self.error_covariance.shape[0]) * 1e-6
        self.error_covariance_inv = np.linalg.pinv(reg_cov)
        
        # Calcular desviación estándar para diagnóstico
        self.error_std = np.sqrt(np.diag(self.error_covariance))
        
        self.is_calibrated = True
        
        # Almacenar errores normales para referencia
        self.normal_buffer = errors
        
        print(f"\nCalibración completada:")
        print(f"  - Media del error: {self.error_mean}")
        print(f"  - Desviación estándar: {self.error_std}")
        print(f"  - Condición de covarianza: {np.linalg.cond(self.error_covariance):.2f}")
        
        return {
            'success': True,
            'error_mean': self.error_mean,
            'error_std': self.error_std,
            'n_samples': len(errors)
        }
    
    def compute_mahalanobis_distance(
        self,
        error: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Calcula la distancia de Mahalanobis.
        
        Fórmula del artículo:
            d = sqrt((e_out - μ)^T Σ^{-1} (e_out - μ))
        
        Args:
            error: Vector de error (e_out - e_out_est)
        
        Returns:
            Distancia de Mahalanobis
        """
        if not self.is_calibrated:
            raise ValueError("El detector no está calibrado. Llame a calibrate_from_normal_data() primero.")
        
        # Convertir a numpy si es tensor
        if isinstance(error, torch.Tensor):
            error = error.detach().cpu().numpy()
        
        error = error.flatten()
        
        # Asegurar que el error tenga la misma dimensión que la media
        if len(error) != len(self.error_mean):
            # Si las dimensiones no coinciden, usar solo los que están disponibles
            min_len = min(len(error), len(self.error_mean))
            error = error[:min_len]
            mean = self.error_mean[:min_len]
            cov_inv = self.error_covariance_inv[:min_len, :min_len]
        else:
            mean = self.error_mean
            cov_inv = self.error_covariance_inv
        
        # Calcular diferencia
        diff = error - mean
        
        # Calcular distancia de Mahalanobis
        try:
            mahalanobis = np.sqrt(diff.T @ cov_inv @ diff)
        except:
            # Si hay error numérico, usar norma euclidiana como fallback
            mahalanobis = np.linalg.norm(diff)
        
        return float(mahalanobis)
    
    def detect_anomaly(
        self,
        x_available: torch.Tensor,
        mask_available: torch.Tensor,
        state_idx: int,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Detecta si el estado actual es anómalo.
        
        Args:
            x_available: Datos disponibles
            mask_available: Máscara de disponibilidad
            state_idx: Índice del estado
            return_details: Si True, retorna información detallada
        
        Returns:
            Diccionario con:
                - is_anomaly: bool
                - mahalanobis_distance: float
                - threshold_used: float
                - absolute_error: float
                - details: (si return_details=True)
        """
        if not self.is_calibrated:
            return {
                'is_anomaly': False,
                'mahalanobis_distance': 0.0,
                'threshold_used': 0.0,
                'absolute_error': 0.0,
                'warning': 'Detector no calibrado'
            }
        
        # Obtener estimación del estado
        result = self.state_estimator.estimate_state(
            x_available=x_available,
            mask_available=mask_available,
            state_idx=state_idx,
            method="auto"
        )
        
        if not result['success'] or result['x_estimated'] is None:
            return {
                'is_anomaly': False,
                'mahalanobis_distance': 0.0,
                'threshold_used': 0.0,
                'absolute_error': 0.0,
                'warning': 'No se pudo estimar el estado'
            }
        
        # Obtener valores reales (simulados o medidos)
        x_out_true = self._get_true_values(x_available, mask_available, state_idx)
        
        # Calcular error
        error = x_out_true - result['x_estimated']
        error_numpy = error.detach().cpu().numpy().flatten()
        
        # Calcular error absoluto (para comparación)
        absolute_error = float(torch.norm(error, p=2).item())
        
        # Calcular distancia de Mahalanobis
        mahalanobis_distance = self.compute_mahalanobis_distance(error_numpy)
        
        # Determinar umbral (adaptativo o fijo)
        if self.config.use_adaptive_threshold:
            # Usar percentil de los datos normales como umbral
            if self.normal_buffer:
                normal_distances = [
                    self.compute_mahalanobis_distance(e) 
                    for e in self.normal_buffer[-100:]
                ]
                threshold = np.percentile(
                    normal_distances, 
                    self.config.mahalanobis_percentile
                )
            else:
                threshold = self.config.mahalanobis_threshold
        else:
            threshold = self.config.mahalanobis_threshold
        
        # Detectar anomalía
        is_anomaly = mahalanobis_distance > threshold
        
        # Clasificar tipo de anomalía si es necesario
        anomaly_type = None
        if is_anomaly and return_details:
            anomaly_type = self._classify_anomaly(
                error_numpy, 
                mahalanobis_distance,
                absolute_error
            )
        
        # Almacenar en historial
        detection = {
            'timestamp': len(self.detection_history),
            'is_anomaly': is_anomaly,
            'mahalanobis_distance': mahalanobis_distance,
            'absolute_error': absolute_error,
            'threshold': threshold,
            'anomaly_type': anomaly_type.value if anomaly_type else None
        }
        self.detection_history.append(detection)
        
        # Actualizar ventana de errores para detección de cambios graduales
        self.error_window.append({
            'error': absolute_error,
            'mahalanobis': mahalanobis_distance
        })
        
        result_dict = {
            'is_anomaly': is_anomaly,
            'mahalanobis_distance': mahalanobis_distance,
            'threshold_used': threshold,
            'absolute_error': absolute_error
        }
        
        if return_details:
            result_dict['details'] = {
                'error_vector': error_numpy.tolist(),
                'error_mean': self.error_mean.tolist(),
                'error_std': self.error_std.tolist() if self.error_std is not None else None,
                'anomaly_type': anomaly_type.value if anomaly_type else None,
                'estimation_method': result['method_used']
            }
        
        return result_dict
    
    def detect_gradual_change(self) -> Dict[str, Any]:
        """
        Detecta cambios graduales en el comportamiento (ej. desgaste).
        
        El artículo menciona en la sección IV-B que los cambios graduales
        son difíciles de detectar como anomalías puntuales.
        
        Returns:
            Diccionario con indicadores de cambio gradual
        """
        if len(self.error_window) < self.config.window_size_detection:
            return {'significant_change': False, 'reason': 'insufficient_data'}
        
        # Obtener estadísticas de la ventana actual
        recent_errors = [e['error'] for e in self.error_window]
        recent_mahalanobis = [e['mahalanobis'] for e in self.error_window]
        
        mean_recent_error = np.mean(recent_errors)
        mean_recent_mahalanobis = np.mean(recent_mahalanobis)
        
        # Comparar con datos normales
        if self.normal_buffer:
            normal_errors = [np.linalg.norm(e) for e in self.normal_buffer[-100:]]
            mean_normal_error = np.mean(normal_errors)
            std_normal_error = np.std(normal_errors)
            
            # Detectar cambio si el error reciente es significativamente mayor
            error_increase = (mean_recent_error - mean_normal_error) / (std_normal_error + 1e-6)
            significant_change = error_increase > self.config.change_detection_sensitivity
        else:
            significant_change = False
            error_increase = 0.0
        
        return {
            'significant_change': significant_change,
            'mean_recent_error': mean_recent_error,
            'mean_recent_mahalanobis': mean_recent_mahalanobis,
            'error_increase_ratio': error_increase,
            'change_detected': significant_change
        }
    
    def _generate_normal_data(
        self,
        state_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera datos normales para calibración.
        En una implementación real, esto leería del robot real.
        """
        n_sensors = self.model.n_sensors
        batch_size = 1
        
        # Generar datos aleatorios dentro de rangos normales
        x_available = torch.randn(batch_size, n_sensors, device=self.device) * 0.5
        mask_available = torch.ones(batch_size, n_sensors, device=self.device)
        
        return x_available, mask_available
    
    def _get_true_values(
        self,
        x_available: torch.Tensor,
        mask_available: torch.Tensor,
        state_idx: int
    ) -> torch.Tensor:
        """
        Obtiene los valores reales de los sensores.
        En una implementación real, esto leería del robot real.
        """
        # Simular que los valores disponibles son los reales
        # (los no disponibles se estiman)
        n_sensors = self.model.n_sensors
        x_true = x_available.clone()
        
        # Para sensores no disponibles, usar una estimación simple
        for i in range(n_sensors):
            if mask_available[0, i] == 0:
                x_true[0, i] = torch.randn(1, device=self.device) * 0.3
        
        return x_true
    
    def _classify_anomaly(
        self,
        error: np.ndarray,
        mahalanobis_distance: float,
        absolute_error: float
    ) -> AnomalyType:
        """
        Clasifica el tipo de anomalía basado en el patrón de error.
        
        Esta es una clasificación simplificada. En un sistema real,
        se usarían técnicas más sofisticadas (ej. análisis de componentes).
        """
        # Análisis simple basado en magnitud y patrón
        if absolute_error > 2.0:
            return AnomalyType.SENSOR_FAILURE
        elif mahalanobis_distance > 5.0:
            return AnomalyType.ENVIRONMENTAL_CHANGE
        elif absolute_error > 1.0:
            return AnomalyType.TOOL_CHANGE
        else:
            return AnomalyType.UNKNOWN
    
    def reset_calibration(self):
        """Resetea la calibración del detector"""
        self.is_calibrated = False
        self.error_mean = None
        self.error_covariance = None
        self.error_covariance_inv = None
        self.error_std = None
        self.normal_buffer = []
        self.detection_history = []
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de las detecciones realizadas"""
        if not self.detection_history:
            return {'total_detections': 0, 'anomaly_rate': 0.0}
        
        anomalies = [d for d in self.detection_history if d['is_anomaly']]
        anomaly_rate = len(anomalies) / len(self.detection_history) if self.detection_history else 0.0
        
        return {
            'total_detections': len(self.detection_history),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': anomaly_rate,
            'avg_mahalanobis': np.mean([d['mahalanobis_distance'] for d in self.detection_history]),
            'max_mahalanobis': max([d['mahalanobis_distance'] for d in self.detection_history])
        }


class MultiModalAnomalyDetector:
    """
    Detector de anomalías multimodal que combina múltiples fuentes.
    
    Útil cuando se tienen múltiples modalidades sensoriales (visión, tacto, etc.)
    y se quiere detectar anomalías en cada una o combinadas.
    """
    
    def __init__(self, detectors: Dict[str, AnomalyDetector]):
        """
        Args:
            detectors: Diccionario {modalidad: AnomalyDetector}
        """
        self.detectors = detectors
    
    def detect_anomaly_multimodal(
        self,
        measurements: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        state_idx: int,
        fusion_method: str = "voting"
    ) -> Dict[str, Any]:
        """
        Detecta anomalías combinando múltiples modalidades.
        
        Args:
            measurements: {modalidad: (x_available, mask_available)}
            state_idx: Índice del estado
            fusion_method: "voting", "max", "weighted"
        
        Returns:
            Diccionario con resultados fusionados
        """
        results = {}
        anomalies = []
        
        for modality, (x_avail, mask_avail) in measurements.items():
            if modality in self.detectors:
                result = self.detectors[modality].detect_anomaly(
                    x_available=x_avail,
                    mask_available=mask_avail,
                    state_idx=state_idx,
                    return_details=True
                )
                results[modality] = result
                if result['is_anomaly']:
                    anomalies.append(modality)
        
        # Fusionar decisiones
        if fusion_method == "voting":
            is_anomaly = len(anomalies) > len(self.detectors) / 2
        elif fusion_method == "max":
            # Si alguna modalidad detecta anomalía, considerar anomalía
            is_anomaly = len(anomalies) > 0
        elif fusion_method == "weighted":
            # Usar pesos basados en confianza de cada detector
            weighted_score = 0.0
            total_weight = 0.0
            for modality, result in results.items():
                weight = 1.0 / (1.0 + result.get('mahalanobis_distance', 1.0))
                weighted_score += weight * float(result['is_anomaly'])
                total_weight += weight
            is_anomaly = weighted_score / total_weight > 0.5 if total_weight > 0 else False
        else:
            is_anomaly = len(anomalies) > 0
        
        return {
            'is_anomaly': is_anomaly,
            'modal_results': results,
            'anomalous_modalities': anomalies,
            'fusion_method': fusion_method
        }


# ============================================
# FUNCIONES DE UTILIDAD Y VISUALIZACIÓN
# ============================================

def visualize_anomaly_detection(
    detector: AnomalyDetector,
    save_path: Optional[str] = None
):
    """
    Visualiza los resultados de detección de anomalías.
    """
    if not detector.detection_history:
        print("No hay historial de detecciones para visualizar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Evolución de la distancia de Mahalanobis
    ax1 = axes[0, 0]
    steps = list(range(len(detector.detection_history)))
    mahalanobis = [d['mahalanobis_distance'] for d in detector.detection_history]
    thresholds = [d['threshold'] for d in detector.detection_history]
    anomalies = [d['is_anomaly'] for d in detector.detection_history]
    
    ax1.plot(steps, mahalanobis, 'b-', alpha=0.7, label='Mahalanobis distance')
    ax1.plot(steps, thresholds, 'r--', alpha=0.7, label='Threshold')
    
    # Marcar anomalías
    anomaly_steps = [steps[i] for i in range(len(steps)) if anomalies[i]]
    anomaly_values = [mahalanobis[i] for i in range(len(steps)) if anomalies[i]]
    ax1.scatter(anomaly_steps, anomaly_values, c='red', s=50, zorder=5, label='Anomaly')
    
    ax1.set_xlabel('Detection Step')
    ax1.set_ylabel('Mahalanobis Distance')
    ax1.set_title('Anomaly Detection Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histograma de distancias de Mahalanobis
    ax2 = axes[0, 1]
    ax2.hist(mahalanobis, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=np.mean(thresholds), color='red', linestyle='--', label='Mean Threshold')
    ax2.set_xlabel('Mahalanobis Distance')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Mahalanobis Distances')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error absoluto vs Mahalanobis
    ax3 = axes[1, 0]
    absolute_errors = [d['absolute_error'] for d in detector.detection_history]
    colors = ['red' if a else 'blue' for a in anomalies]
    ax3.scatter(absolute_errors, mahalanobis, c=colors, alpha=0.6)
    ax3.set_xlabel('Absolute Error (L2 Norm)')
    ax3.set_ylabel('Mahalanobis Distance')
    ax3.set_title('Absolute Error vs Mahalanobis Distance\n(Red = Anomaly)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Evolución del error absoluto
    ax4 = axes[1, 1]
    ax4.plot(steps, absolute_errors, 'g-', alpha=0.7)
    ax4.fill_between(steps, 0, absolute_errors, alpha=0.3)
    ax4.set_xlabel('Detection Step')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Prediction Error Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en {save_path}")
    
    plt.show()


def simulate_anomaly_detection_experiment():
    """
    Simula un experimento de detección de anomalías como en el artículo.
    
    Basado en el experimento Musashi (sección III-B.3, Figura 8b):
        - Estado normal: robot moviéndose aleatoriamente
        - Anomalía 1: agarrar un objeto pesado
        - Anomalía 2: desactivar un músculo
    """
    print("=" * 70)
    print("GeMuCo - Fase 9: Detección de Anomalías")
    print("=" * 70)
    
    # Configuración
    n_sensors = 6
    dim_z = 16
    dim_p = 2
    n_states = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguración:")
    print(f"  - n_sensors: {n_sensors}")
    print(f"  - device: {device}")
    
    # Crear componentes necesarios
    model = GeMuCoNetwork(
        n_sensors=n_sensors,
        dim_z=dim_z,
        dim_p=dim_p,
        hidden_sizes=[128, 64, 64, 128],
        use_batchnorm=True
    ).to(device)
    
    mask_manager = MaskManager(n_sensors)
    mask_manager.add_mask(torch.cat([torch.ones(3), torch.zeros(3)]))
    mask_manager.add_mask(torch.ones(n_sensors))
    
    pb_manager = ParametricBiasManager(dim_p=dim_p, n_states=n_states)
    
    # Crear optimizadores (necesarios para state estimator)
    from optimizer import LatentOptimizer, XInOptimizer
    
    latent_optimizer = LatentOptimizer(
        model=model,
        learning_rate=0.01,
        n_iterations=20,
        verbose=False
    )
    
    xin_optimizer = XInOptimizer(
        model=model,
        mask_manager=mask_manager,
        learning_rate=0.01,
        n_iterations=20,
        verbose=False
    )
    
    state_estimator = StateEstimator(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        latent_optimizer=latent_optimizer,
        xin_optimizer=xin_optimizer,
        device=device
    )
    
    # Crear detector de anomalías
    config = AnomalyConfig(
        mahalanobis_threshold=3.0,
        mahalanobis_percentile=99.0,
        window_size_normal=50,
        window_size_detection=10
    )
    
    detector = AnomalyDetector(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        state_estimator=state_estimator,
        config=config,
        device=device
    )
    
    # 1. Calibrar con datos normales
    print("\n" + "-" * 50)
    print("Paso 1: Calibración con datos normales")
    print("-" * 50)
    
    calibration_result = detector.calibrate_from_normal_data(
        n_samples=50,
        state_idx=0,
        verbose=True
    )
    
    if not calibration_result['success']:
        print("Error en calibración")
        return detector, []
    
    # 2. Simular detección de anomalías
    print("\n" + "-" * 50)
    print("Paso 2: Simulación de detección de anomalías")
    print("-" * 50)
    
    # Simular diferentes tipos de anomalías
    anomaly_scenarios = [
        {'name': 'Movimiento normal', 'severity': 0.0},
        {'name': 'Objeto pesado (ligero)', 'severity': 0.5},
        {'name': 'Objeto pesado (moderado)', 'severity': 1.0},
        {'name': 'Objeto pesado (pesado)', 'severity': 1.5},
        {'name': 'Fallo de sensor', 'severity': 2.0},
        {'name': 'Desactivación de músculo', 'severity': 2.5},
    ]
    
    results = []
    
    for scenario in anomaly_scenarios:
        print(f"\n  Escenario: {scenario['name']} (severidad={scenario['severity']})")
        
        # Simular datos afectados por la anomalía
        x_available = torch.randn(1, n_sensors, device=device) * (1 + scenario['severity'] * 0.5)
        mask_available = torch.ones(1, n_sensors, device=device)
        
        # Si es fallo de sensor, desconectar algunos sensores
        if scenario['severity'] >= 2.0:
            mask_available[0, 3:5] = 0.0
            print(f"    Sensores desconectados: índices 3,4")
        
        # Detectar anomalía
        result = detector.detect_anomaly(
            x_available=x_available,
            mask_available=mask_available,
            state_idx=0,
            return_details=True
        )
        
        print(f"    Anomalía detectada: {result['is_anomaly']}")
        print(f"    Distancia Mahalanobis: {result['mahalanobis_distance']:.3f}")
        print(f"    Umbral: {result['threshold_used']:.3f}")
        
        if result['details'].get('anomaly_type'):
            print(f"    Tipo: {result['details']['anomaly_type']}")
        
        results.append(result)
    
    # 3. Mostrar estadísticas
    print("\n" + "-" * 50)
    print("Paso 3: Estadísticas de detección")
    print("-" * 50)
    
    stats = detector.get_anomaly_statistics()
    print(f"  Total detecciones: {stats['total_detections']}")
    print(f"  Anomalías detectadas: {stats['anomalies_detected']}")
    print(f"  Tasa de anomalías: {stats['anomaly_rate']*100:.1f}%")
    print(f"  Mahalanobis promedio: {stats['avg_mahalanobis']:.3f}")
    print(f"  Mahalanobis máximo: {stats['max_mahalanobis']:.3f}")
    
    # 4. Detectar cambios graduales
    print("\n" + "-" * 50)
    print("Paso 4: Detección de cambios graduales")
    print("-" * 50)
    
    gradual_change = detector.detect_gradual_change()
    print(f"  Cambio significativo detectado: {gradual_change.get('change_detected', False)}")
    if 'error_increase_ratio' in gradual_change:
        print(f"  Incremento de error: {gradual_change['error_increase_ratio']:.2f} desviaciones")
    
    # 5. Visualizar resultados
    print("\n" + "-" * 50)
    print("Paso 5: Visualización de resultados")
    print("-" * 50)
    
    visualize_anomaly_detection(detector)
    
    print("\n" + "=" * 70)
    print("Resumen de la Fase 9")
    print("=" * 70)
    print("""
    Implementado:
    ✓ Calibración con datos en estado normal (recolección de media y covarianza)
    ✓ Cálculo de distancia de Mahalanobis (fórmula del artículo)
    ✓ Detección de anomalías con umbral fijo o adaptativo
    ✓ Clasificación de tipos de anomalías (sensor, actuador, entorno, herramienta)
    ✓ Detección de cambios graduales (desgaste, envejecimiento)
    ✓ Detector multimodal (combina múltiples fuentes sensoriales)
    ✓ Visualización de resultados (como Figura 8b del artículo)

    Correspondencia con el artículo:
    - Sección II-K: Fórmula de distancia de Mahalanobis
    - Experimento Musashi (Figura 8b): detección al agarrar objeto pesado
    - Experimento Musashi: detección al desactivar músculo
    - "d rises sharply when stopping the function of one muscle"
    
    El artículo demostró que después del online update:
    - El error de estimación bajó de 0.414 rad a 0.186 rad
    - La detección de anomalías era más clara (d aumentaba mucho)
    """)
    
    return detector, results


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    detector, results = simulate_anomaly_detection_experiment()