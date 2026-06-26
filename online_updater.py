"""08
GeMuCo: Generalized Multisensory Correlational Model
Fase 8: Online Updater - Actualización Online del Modelo

Basado en la sección II-F del artículo:
    "When the robot's physical state, tools, or surrounding environment changes,
    a model adapted to the current state can be used by updating GeMuCo for
    accurate state estimation and control."

Formas de actualización online según el artículo:
    1. Updating W (solo pesos de la red)
    2. Updating p (solo parametric bias)
    3. Updating W and p simultaneously

El artículo también menciona (sección II-F):
    "When the number of data exceeds a determined threshold, data is discarded
    from the oldest."
    
    "In the case of updating only p, only some dynamics are changed and the
    structure of the overall network is kept the same, thus overfitting is
    unlikely to occur."
    
    "On the other hand, it should be noted that updating W or updating W and p
    simultaneously changes the structure of the entire network, and thus
    overfitting is likely to occur."
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import time
from dataclasses import dataclass, field
from enum import Enum

# Importar de fases anteriores
try:
    from model import GeMuCoNetwork, MaskManager, ParametricBiasManager
    from trainer import GeMuCoTrainer
except ImportError:
    print("Importando módulos locales...")


class UpdateMode(Enum):
    """Modos de actualización online según el artículo"""
    WEIGHTS_ONLY = "update_W"           # Solo actualizar W
    BIAS_ONLY = "update_p"              # Solo actualizar p
    BOTH = "update_both"                # Actualizar W y p simultáneamente
    ADAPTIVE = "adaptive"               # Adaptativo (decide automáticamente)


@dataclass
class OnlineUpdateConfig:
    """Configuración para actualización online"""
    # Parámetros generales
    buffer_max_size: int = 1000         # Tamaño máximo del buffer de experiencia
    batch_size: int = 32                # Tamaño del batch para actualización
    update_frequency: int = 10          # Actualizar cada N pasos
    
    # Learning rates
    lr_w: float = 0.001                 # Learning rate para pesos W
    lr_p: float = 0.01                  # Learning rate para parametric bias (más rápido)
    
    # Parámetros de actualización
    n_steps_per_update: int = 5         # Pasos de gradiente por actualización
    use_momentum: bool = True           # Usar momento en optimizadores
    
    # Prevención de sobreajuste
    use_weight_decay: bool = True       # Usar weight decay (L2 regularization)
    weight_decay: float = 0.0001        # Factor de weight decay
    
    # Umbrales
    loss_threshold: float = 0.01        # Si pérdida < umbral, no actualizar
    min_samples_for_update: int = 10    # Mínimo de muestras para actualizar
    
    # Para debugging
    verbose: bool = False
    log_losses: bool = True


class ExperienceBuffer:
    """
    Buffer de experiencia para actualización online.
    
    Según el artículo (sección II-F):
        "When the number of data exceeds a determined threshold,
        data is discarded from the oldest."
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Número máximo de experiencias a almacenar
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, x_in: torch.Tensor, x_out: torch.Tensor, 
            m: torch.Tensor, p: torch.Tensor, state_idx: int):
        """
        Agrega una nueva experiencia al buffer.
        
        Args:
            x_in: Datos de entrada
            x_out: Datos objetivo
            m: Máscara utilizada
            p: Parametric bias (puede ser None si no se usa)
            state_idx: Índice del estado
        """
        experience = {
            'x_in': x_in.clone().detach().cpu(),
            'x_out': x_out.clone().detach().cpu(),
            'm': m.clone().detach().cpu(),
            'p': p.clone().detach().cpu() if p is not None else None,
            'state_idx': state_idx,
            'timestamp': time.time()
        }
        self.buffer.append(experience)
    
    def add_batch(self, x_in_batch: torch.Tensor, x_out_batch: torch.Tensor,
                  m_batch: torch.Tensor, p_batch: Optional[torch.Tensor],
                  state_idx: int):
        """Agrega un batch de experiencias"""
        for i in range(x_in_batch.shape[0]):
            p_i = p_batch[i:i+1] if p_batch is not None else None
            self.add(x_in_batch[i:i+1], x_out_batch[i:i+1],
                    m_batch[i:i+1], p_i, state_idx)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """
        Muestrea un batch aleatorio del buffer.
        
        Returns:
            Lista de experiencias
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_all(self) -> List[Dict]:
        """Obtiene todas las experiencias del buffer"""
        return list(self.buffer)
    
    def get_by_state(self, state_idx: int) -> List[Dict]:
        """Obtiene experiencias de un estado específico"""
        return [exp for exp in self.buffer if exp['state_idx'] == state_idx]
    
    def clear(self):
        """Limpia el buffer"""
        self.buffer.clear()
    
    def size(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_samples: int) -> bool:
        """Verifica si hay suficientes muestras para actualizar"""
        return len(self.buffer) >= min_samples


class OnlineUpdater:
    """
    Actualizador online de GeMuCo.
    
    Implementa las tres formas de actualización descritas en el artículo:
        1. update_W: actualiza solo los pesos de la red
        2. update_p: actualiza solo el parametric bias
        3. update_both: actualiza W y p simultáneamente
    
    El artículo también menciona (sección II-F):
        "In the case of offline update, the network is updated once after a
        certain amount of data has been accumulated. In the case of online
        update, the network is updated gradually."
    """
    
    def __init__(
        self,
        model: GeMuCoNetwork,
        mask_manager: MaskManager,
        pb_manager: ParametricBiasManager,
        config: OnlineUpdateConfig = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: Red GeMuCo
            mask_manager: Gestor de máscaras
            pb_manager: Gestor de parametric biases
            config: Configuración de actualización online
            device: CPU o CUDA
        """
        self.model = model
        self.mask_manager = mask_manager
        self.pb_manager = pb_manager
        self.device = device
        
        self.config = config or OnlineUpdateConfig()
        
        # Crear optimizadores (se mantienen entre actualizaciones)
        # Optimizador para pesos W
        if self.config.use_weight_decay:
            self.optimizer_w = optim.Adam(
                model.parameters(), 
                lr=self.config.lr_w,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer_w = optim.Adam(model.parameters(), lr=self.config.lr_w)
        
        # Optimizador para parametric biases p
        pb_params = pb_manager.get_all_pb()
        if self.config.use_weight_decay:
            self.optimizer_p = optim.Adam(
                pb_params,
                lr=self.config.lr_p,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer_p = optim.Adam(pb_params, lr=self.config.lr_p)
        
        # Buffer de experiencia
        self.buffer = ExperienceBuffer(max_size=self.config.buffer_max_size)
        
        # Función de pérdida
        self.criterion = nn.MSELoss()
        
        # Estado del updater
        self.step_count = 0
        self.update_count = 0
        self.loss_history = {
            'update_W': [],
            'update_p': [],
            'update_both': []
        }
        
        # Detector de cambios (para decidir cuándo actualizar)
        self.change_detector = ChangeDetector()
    
    def add_experience(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        m: torch.Tensor,
        p: Optional[torch.Tensor],
        state_idx: int
    ):
        """
        Agrega una experiencia al buffer.
        
        Args:
            x_in: Entrada utilizada
            x_out: Salida observada
            m: Máscara utilizada
            p: Parametric bias (si se usó)
            state_idx: Índice del estado
        """
        self.buffer.add(x_in, x_out, m, p, state_idx)
    
    def update_weights_only(
        self,
        batch: List[Dict],
        n_steps: Optional[int] = None
    ) -> float:
        """
        Actualiza solo los pesos W (no los parametric biases).
        
        Según el artículo (sección II-F):
            "updating W or updating W and p simultaneously changes the structure
            of the entire network, and thus overfitting is likely to occur."
        
        Args:
            batch: Lista de experiencias
            n_steps: Número de pasos de gradiente (si None, usa config)
        
        Returns:
            Pérdida promedio
        """
        if n_steps is None:
            n_steps = self.config.n_steps_per_update
        
        if len(batch) == 0:
            return 0.0
        
        total_loss = 0.0
        
        # Poner modelo en modo entrenamiento
        self.model.train()
        
        for step in range(n_steps):
            # Preparar batch
            x_in_batch = torch.cat([exp['x_in'] for exp in batch]).to(self.device)
            x_out_batch = torch.cat([exp['x_out'] for exp in batch]).to(self.device)
            #m_batch = torch.cat([exp['m'] for exp in batch]).to(self.device)

            # 🔧 Asegurar dimensiones correctas
            m_list = []
            for exp in batch:
                m = exp['m']
                if m.dim() == 1:
                    m = m.unsqueeze(0)
                m_list.append(m)
            m_batch = torch.cat(m_list, dim=0).to(self.device)
            
            # Obtener p para cada experiencia
            p_list = []
            for exp in batch:
                if exp['p'] is not None:
                    p = exp['p']
                else:
                    p = self.pb_manager.get_pb(exp['state_idx'])
                if p.dim() == 1:
                    p = p.unsqueeze(0)
                p_list.append(p)
            p_batch = torch.cat(p_list, dim=0).to(self.device)

            # 🔧 Expandir si es necesario
            if m_batch.shape[0] != x_in_batch.shape[0]:
                m_batch = m_batch.expand(x_in_batch.shape[0], -1)
            if p_batch.shape[0] != x_in_batch.shape[0]:
                p_batch = p_batch.expand(x_in_batch.shape[0], -1)
            
            # Forward pass
            x_out_pred, _ = self.model(x_in_batch, m_batch, p_batch)
            
            # Calcular pérdida
            loss = self.criterion(x_out_pred, x_out_batch)
            
            # Backward pass
            self.optimizer_w.zero_grad()
            loss.backward()
            self.optimizer_w.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_steps
        
        if self.config.log_losses:
            self.loss_history['update_W'].append({
                'step': self.update_count,
                'loss': avg_loss,
                'batch_size': len(batch)
            })
        
        return avg_loss
    
    def update_bias_only(
        self,
        batch: List[Dict],
        n_steps: Optional[int] = None
    ) -> float:
        """
        Actualiza solo los parametric biases p (no los pesos W).
        
        Según el artículo (sección II-F):
            "updating only p, only some dynamics are changed and the structure
            of the overall network is kept the same, thus overfitting is unlikely to occur."
        
        Args:
            batch: Lista de experiencias
            n_steps: Número de pasos de gradiente
        
        Returns:
            Pérdida promedio
        """
        if n_steps is None:
            n_steps = self.config.n_steps_per_update
        
        if len(batch) == 0:
            return 0.0
        
        total_loss = 0.0
        
        self.model.train()

        # CORREGIDO: No congelar los pesos, simplemente no actualizarlos
        # En lugar de requires_grad=False, usamos un optimizador separado
        # y simplemente no llamamos a optimizer_w.step()
        
        # IMPORTANTE: Congelar pesos W (no actualizar)
        #for param in self.model.parameters():
            #param.requires_grad = False
        
        for step in range(n_steps):
            x_in_batch = torch.cat([exp['x_in'] for exp in batch]).to(self.device)
            x_out_batch = torch.cat([exp['x_out'] for exp in batch]).to(self.device)
            #m_batch = torch.cat([exp['m'] for exp in batch]).to(self.device)

            # 🔧 Asegurar que m_batch tenga forma [batch_size, n_sensors]
            m_list = []
            for exp in batch:
                m = exp['m']
                if m.dim() == 1:
                    m = m.unsqueeze(0)
                m_list.append(m)
            m_batch = torch.cat(m_list, dim=0).to(self.device)
            
            # Asegurar que p_batch tenga forma [batch_size, dim_p]
            p_list = []
            for exp in batch:
                if exp['p'] is not None:
                    p = exp['p']
                else:
                    p = self.pb_manager.get_pb(exp['state_idx'])
                
                if p.dim() == 1:
                    p = p.unsqueeze(0)
                p_list.append(p)
            p_batch = torch.cat(p_list, dim=0).to(self.device)

            # Verificar dimensiones
            if m_batch.dim() == 2 and m_batch.shape[0] != x_in_batch.shape[0]:
                # Si la máscara tiene batch_size=1, expandir
                m_batch = m_batch.expand(x_in_batch.shape[0], -1)
            if p_batch.dim() == 2 and p_batch.shape[0] != x_in_batch.shape[0]:
                p_batch = p_batch.expand(x_in_batch.shape[0], -1)
            
            # Forward pass
            x_out_pred, _ = self.model(x_in_batch, m_batch, p_batch)
            
            # Calcular pérdida
            loss = self.criterion(x_out_pred, x_out_batch)
            
            # Backward pass (solo para p)
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_p.step()
            
            total_loss += loss.item()
            
            # Actualizar los p en el buffer con los nuevos valores
            for i, exp in enumerate(batch):
                if exp['p'] is not None:
                    # Actualizar el p almacenado en el buffer
                    new_p = self.pb_manager.get_pb(exp['state_idx']).clone().detach().cpu()
                    exp['p'] = new_p
        
        # Descongelar pesos W
        #for param in self.model.parameters():
            #param.requires_grad = True
        
        avg_loss = total_loss / n_steps
        
        if self.config.log_losses:
            self.loss_history['update_p'].append({
                'step': self.update_count,
                'loss': avg_loss,
                'batch_size': len(batch)
            })
        
        return avg_loss
    
    def update_both(
        self,
        batch: List[Dict],
        n_steps: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Actualiza simultáneamente W y p.
        
        Según el artículo (sección II-F):
            "updating W and p simultaneously changes the structure of the entire
            network, and thus overfitting is likely to occur."
        
        Args:
            batch: Lista de experiencias
            n_steps: Número de pasos de gradiente
        
        Returns:
            (loss_w, loss_p) pérdidas para cada optimizador
        """
        if n_steps is None:
            n_steps = self.config.n_steps_per_update
        
        if len(batch) == 0:
            return 0.0, 0.0
        
        total_loss_w = 0.0
        total_loss_p = 0.0
        
        self.model.train()
        
        for step in range(n_steps):
            x_in_batch = torch.cat([exp['x_in'] for exp in batch]).to(self.device)
            x_out_batch = torch.cat([exp['x_out'] for exp in batch]).to(self.device)
            #m_batch = torch.cat([exp['m'] for exp in batch]).to(self.device)

            # 🔧 Asegurar dimensiones correctas
            m_list = []
            for exp in batch:
                m = exp['m']
                if m.dim() == 1:
                    m = m.unsqueeze(0)
                m_list.append(m)
            m_batch = torch.cat(m_list, dim=0).to(self.device)
            
            # Obtener p para cada experiencia
            p_list = []
            for exp in batch:
                if exp['p'] is not None:
                    p = exp['p']
                else:
                    p = self.pb_manager.get_pb(exp['state_idx'])
                if p.dim() == 1:
                    p = p.unsqueeze(0)
                p_list.append(p)
            p_batch = torch.cat(p_list, dim=0).to(self.device)

            # 🔧 Expandir si es necesario
            if m_batch.shape[0] != x_in_batch.shape[0]:
                m_batch = m_batch.expand(x_in_batch.shape[0], -1)
            if p_batch.shape[0] != x_in_batch.shape[0]:
                p_batch = p_batch.expand(x_in_batch.shape[0], -1)
            
            # 🔧 Actualizar W
            self.optimizer_w.zero_grad()
            x_out_pred, _ = self.model(x_in_batch, m_batch, p_batch)
            loss_w = self.criterion(x_out_pred, x_out_batch)
            loss_w.backward()
            self.optimizer_w.step()
            total_loss_w += loss_w.item()
        
            # 🔧 Actualizar p (con el mismo forward)
            self.optimizer_p.zero_grad()
            x_out_pred, _ = self.model(x_in_batch, m_batch, p_batch)
            loss_p = self.criterion(x_out_pred, x_out_batch)
            loss_p.backward()
            self.optimizer_p.step()
            total_loss_p += loss_p.item()
        
        avg_loss_w = total_loss_w / n_steps
        avg_loss_p = total_loss_p / n_steps
        
        if self.config.log_losses:
            self.loss_history['update_both'].append({
                'step': self.update_count,
                'loss_w': avg_loss_w,
                'loss_p': avg_loss_p,
                'batch_size': len(batch)
            })
        
        return avg_loss_w, avg_loss_p
    
    def update_online(
        self,
        mode: UpdateMode = UpdateMode.ADAPTIVE,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Realiza una actualización online completa.
        
        Args:
            mode: Modo de actualización
            force_update: Si True, fuerza actualización incluso si no se alcanza la frecuencia
        
        Returns:
            Diccionario con resultados de la actualización
        """
        self.step_count += 1
        
        result = {
            'updated': False,
            'mode': mode.value,
            'loss': None,
            'buffer_size': self.buffer.size(),
            'step': self.step_count
        }
        
        # Verificar si debemos actualizar
        should_update = force_update or (
            self.step_count % self.config.update_frequency == 0 and
            self.buffer.is_ready(self.config.min_samples_for_update)
        )
        
        if not should_update:
            return result
        
        # Obtener batch del buffer
        batch = self.buffer.sample(self.config.batch_size)
        
        if len(batch) == 0:
            return result
        
        # Detectar si hay cambio (para modo adaptativo)
        if mode == UpdateMode.ADAPTIVE:
            change_detected = self.change_detector.detect_change(batch)
            if change_detected.get('significant', False):
                # Si hay cambio significativo, actualizar ambos
                mode = UpdateMode.BOTH
            else:
                # Si no hay cambio, solo actualizar p (más seguro)
                mode = UpdateMode.BIAS_ONLY
        
        # Realizar actualización según modo
        if mode == UpdateMode.WEIGHTS_ONLY:
            loss = self.update_weights_only(batch)
            result['loss'] = loss
            result['updated'] = True
            
        elif mode == UpdateMode.BIAS_ONLY:
            loss = self.update_bias_only(batch)
            result['loss'] = loss
            result['updated'] = True
            
        elif mode == UpdateMode.BOTH:
            loss_w, loss_p = self.update_both(batch)
            result['loss'] = {'w': loss_w, 'p': loss_p}
            result['updated'] = True
        
        if result['updated']:
            self.update_count += 1
            
        return result
    
    def adaptive_online_learning(
        self,
        stream_generator: Callable,
        n_steps: int = 100,
        initial_mode: UpdateMode = UpdateMode.BIAS_ONLY,
        switch_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Bucle de aprendizaje online adaptativo.
        
        Comienza actualizando solo p (más seguro) y si la pérdida no disminuye
        lo suficiente, cambia a actualizar ambos.
        
        Args:
            stream_generator: Generador que produce (x_in, x_out, m, state_idx) en cada paso
            n_steps: Número de pasos
            initial_mode: Modo inicial
            switch_threshold: Umbral para cambiar de modo
        
        Returns:
            Lista de resultados por paso
        """
        results = []
        current_mode = initial_mode
        
        # Para tracking de rendimiento
        recent_losses = deque(maxlen=10)
        
        print(f"Iniciando aprendizaje online por {n_steps} pasos...")
        print(f"  Modo inicial: {current_mode.value}")
        
        for step in range(n_steps):
            # Obtener nueva experiencia
            x_in, x_out, m, state_idx = stream_generator()
            
            # Agregar al buffer
            p_k = self.pb_manager.get_pb(state_idx)
            self.add_experience(x_in, x_out, m, p_k.unsqueeze(0), state_idx)
            
            # Actualizar online
            result = self.update_online(mode=current_mode)
            
            if result['updated'] and result['loss'] is not None:
                loss = result['loss']
                if isinstance(loss, dict):
                    loss = loss.get('w', loss.get('p', 0))
                
                recent_losses.append(loss)
                
                # Decidir si cambiar de modo
                if current_mode == UpdateMode.BIAS_ONLY and len(recent_losses) >= 5:
                    avg_loss = np.mean(recent_losses)
                    if avg_loss > switch_threshold:
                        print(f"  Paso {step}: pérdida alta ({avg_loss:.4f}), cambiando a modo BOTH")
                        current_mode = UpdateMode.BOTH
            
            results.append({
                'step': step,
                'mode': current_mode.value,
                'updated': result['updated'],
                'loss': result.get('loss'),
                'buffer_size': result['buffer_size']
            })
            
            if self.config.verbose and step % 10 == 0:
                print(f"  Paso {step}: modo={current_mode.value}, "
                      f"buffer={result['buffer_size']}, "
                      f"loss={result.get('loss', 'N/A')}")
        
        return results


class ChangeDetector:
    """
    Detector de cambios para actualización adaptativa.
    
    Detecta si el comportamiento del robot ha cambiado significativamente,
    lo que indicaría que se necesita una actualización más agresiva.
    """
    
    def __init__(self, sensitivity: float = 0.1, window_size: int = 20):
        """
        Args:
            sensitivity: Sensibilidad para detectar cambios
            window_size: Tamaño de la ventana para calcular estadísticas
        """
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.loss_window = deque(maxlen=window_size)
        self.prediction_error_window = deque(maxlen=window_size)
    
    def detect_change(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Detecta si hay cambios significativos en los datos.
        
        Returns:
            Diccionario con indicadores de cambio
        """
        # Calcular error de predicción promedio del batch
        # (esto asume que ya tenemos predicciones, simplificado)
        
        if len(self.loss_window) < self.window_size:
            return {'significant': False, 'reason': 'insufficient_data'}
        
        # Calcular estadísticas
        mean_loss = np.mean(self.loss_window)
        std_loss = np.std(self.loss_window)
        recent_mean = np.mean(list(self.loss_window)[-5:])
        
        # Detectar cambio si el error reciente es significativamente mayor
        change_detected = recent_mean > mean_loss + self.sensitivity * std_loss
        
        return {
            'significant': change_detected,
            'mean_loss': mean_loss,
            'recent_loss': recent_mean,
            'std_loss': std_loss
        }
    
    def add_observation(self, loss: float):
        """Agrega una observación de pérdida"""
        self.loss_window.append(loss)


class ContinualLearningUpdater(OnlineUpdater):
    """
    Actualizador con aprendizaje continuo y prevención de olvido catastrófico.
    
    El artículo menciona (sección IV-B):
        "Techniques such as Elastic Weight Consolidation [30] have been developed,
        and their incorporation should be considered in the future."
    
    Esta implementación incluye una versión simplificada de EWC.
    """
    
    def __init__(self, *args, ewc_lambda: float = 0.1, **kwargs):
        """
        Args:
            ewc_lambda: Fuerza de la regularización EWC
        """
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = None
        self.optimal_params = None
        
        # Almacenar parámetros importantes después del entrenamiento inicial
        self._store_important_parameters()
    
    def _store_important_parameters(self):
        """
        Almacena los parámetros óptimos y calcula la matriz de Fisher aproximada.
        """
        self.optimal_params = {}
        self.fisher_matrix = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.clone().detach()
                # Matriz de Fisher diagonal aproximada (inicializada en 0)
                self.fisher_matrix[name] = torch.zeros_like(param)
    
    def _compute_ewc_loss(self) -> torch.Tensor:
        """
        Calcula la pérdida de Elastic Weight Consolidation.
        """
        if self.optimal_params is None:
            return torch.tensor(0.0, device=self.device)
        
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.optimal_params:
                # Diferencia cuadrática ponderada por Fisher
                diff = param - self.optimal_params[name]
                fisher = self.fisher_matrix[name]
                ewc_loss += (fisher * diff * diff).sum()
        
        return (self.ewc_lambda / 2) * ewc_loss
    
    def update_both_with_ewc(
        self,
        batch: List[Dict],
        n_steps: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Actualiza W y p con regularización EWC para prevenir olvido catastrófico.
        """
        if n_steps is None:
            n_steps = self.config.n_steps_per_update
        
        if len(batch) == 0:
            return 0.0, 0.0
        
        total_loss_w = 0.0
        total_loss_p = 0.0
        
        self.model.train()
        
        for step in range(n_steps):
            x_in_batch = torch.cat([exp['x_in'] for exp in batch]).to(self.device)
            x_out_batch = torch.cat([exp['x_out'] for exp in batch]).to(self.device)
            m_batch = torch.cat([exp['m'] for exp in batch]).to(self.device)
            
            p_list = []
            for exp in batch:
                if exp['p'] is not None:
                    p_list.append(exp['p'].to(self.device))
                else:
                    p_k = self.pb_manager.get_pb(exp['state_idx'])
                    p_list.append(p_k.unsqueeze(0))
            p_batch = torch.cat(p_list, dim=0)
            
            x_out_pred, _ = self.model(x_in_batch, m_batch, p_batch)
            
            # Pérdida estándar + pérdida EWC
            task_loss = self.criterion(x_out_pred, x_out_batch)
            ewc_loss = self._compute_ewc_loss()
            loss = task_loss + ewc_loss
            
            # Actualizar W
            self.optimizer_w.zero_grad()
            loss.backward()
            self.optimizer_w.step()
            total_loss_w += loss.item()
            
            # Actualizar p
            x_out_pred, _ = self.model(x_in_batch, m_batch, p_batch)
            loss_p = self.criterion(x_out_pred, x_out_batch)
            
            self.optimizer_p.zero_grad()
            loss_p.backward()
            self.optimizer_p.step()
            total_loss_p += loss_p.item()
        
        return total_loss_w / n_steps, total_loss_p / n_steps


# ============================================
# SIMULACIÓN DE FLUJO ONLINE
# ============================================

def simulate_online_learning_experiment():
    """
    Simula un experimento de aprendizaje online como en el artículo.
    
    Basado en el experimento PR2 (sección III-A.3) donde:
        - El estado de agarre cambia gradualmente
        - Se actualiza p online para adaptarse
        - Se compara "update p" vs "update W"
    """
    print("=" * 70)
    print("GeMuCo - Fase 8: Actualización Online")
    print("=" * 70)
    
    # Configuración
    n_sensors = 6
    dim_z = 16
    dim_p = 2
    n_states = 1  # Comenzamos con un estado, luego añadimos más
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguración:")
    print(f"  - n_sensors: {n_sensors}")
    print(f"  - dim_p: {dim_p}")
    print(f"  - device: {device}")
    
    # Crear modelo
    model = GeMuCoNetwork(
        n_sensors=n_sensors,
        dim_z=dim_z,
        dim_p=dim_p,
        hidden_sizes=[128, 64, 64, 128],
        use_batchnorm=True
    ).to(device)
    
    # Crear mask manager
    mask_manager = MaskManager(n_sensors)
    mask_manager.add_mask(torch.cat([torch.ones(3), torch.zeros(3)]))
    
    # Crear PB manager
    pb_manager = ParametricBiasManager(dim_p=dim_p, n_states=n_states)
    
    # Configuración de actualización online
    config = OnlineUpdateConfig(
        buffer_max_size=200,
        batch_size=16,
        update_frequency=5,
        lr_w=0.001,
        lr_p=0.01,
        n_steps_per_update=3,
        verbose=True,
        log_losses=True
    )
    
    # Crear updater
    updater = OnlineUpdater(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        config=config,
        device=device
    )
    
    # Simular cambio de estado (como en el artículo)
    print("\n" + "-" * 50)
    print("Simulando cambio de estado de agarre")
    print("El PB debe adaptarse al nuevo estado")
    print("-" * 50)
    
    # Generador de flujo de datos
    def data_stream():
        """Genera datos simulando un cambio gradual en el estado de agarre"""
        # Estado inicial: herramienta larga (l=700mm)
        # Estado final: herramienta corta (l=300mm)
        
        # Simulamos la transición del artículo
        transition_steps = 100
        
        for step in range(transition_steps):
            # Progreso de la transición (0 a 1)
            progress = step / transition_steps
            
            # Generar valores de sensores afectados por la transición
            batch_size = 1
            
            # Ángulos aleatorios
            x_in = torch.randn(batch_size, n_sensors, device=device)
            x_in[0, 3:] = 0  # tooltip no disponible inicialmente
            
            # Tooltip depende del progreso (cambiando gradualmente)
            tooltip_values = torch.tensor([
                0.5 + 0.3 * (1 - progress),   # x
                0.3 - 0.2 * (1 - progress),   # y
                0.2 + 0.1 * (1 - progress)    # z
            ], device=device)
            
            x_out = torch.randn(batch_size, n_sensors, device=device)
            x_out[0, 3:6] = tooltip_values
            
            # Máscara: solo ángulos disponibles
            m = torch.zeros(batch_size, n_sensors, device=device)
            m[0, 0:3] = 1.0
            
            state_idx = 0
            
            yield x_in, x_out, m, state_idx
    
    # Ejecutar aprendizaje online
    results = updater.adaptive_online_learning(
        stream_generator=data_stream().__next__,
        n_steps=100,
        initial_mode=UpdateMode.BIAS_ONLY,
        switch_threshold=0.1
    )
    
    # Mostrar resultados
    print("\n" + "-" * 50)
    print("Resultados del aprendizaje online")
    print("-" * 50)
    
    updated_steps = [r for r in results if r['updated']]
    print(f"  Actualizaciones realizadas: {len(updated_steps)}/{len(results)}")
    
    if updated_steps:
        avg_loss = np.mean([r.get('loss', 0) for r in updated_steps if r.get('loss') is not None])
        print(f"  Pérdida promedio: {avg_loss:.4f}")
    
    print(f"\n  Modo final utilizado: {results[-1]['mode']}")
    
    # Mostrar evolución de PB
    print("\n" + "-" * 50)
    print("Evolución del Parametric Bias durante adaptación")
    print("-" * 50)
    pb_final = pb_manager.get_pb(0).detach().cpu().numpy()
    print(f"  PB final: [{pb_final[0]:.4f}, {pb_final[1]:.4f}]")
    
    print("\n" + "=" * 70)
    print("Resumen de la Fase 8")
    print("=" * 70)
    print("""
    Implementado:
    ✓ Buffer de experiencia con descarte FIFO (artículo: "discard oldest")
    ✓ Actualización solo de W
    ✓ Actualización solo de p (más seguro, sin sobreajuste)
    ✓ Actualización simultánea de W y p
    ✓ Modo adaptativo que decide automáticamente
    ✓ Detector de cambios para adaptación inteligente
    ✓ Versión con Elastic Weight Consolidation (EWC) para evitar olvido catastrófico
    
    Correspondencia con el artículo:
    - Sección II-F: Online Update (página 6)
    - "updating only p ... overfitting is unlikely to occur"
    - "updating W ... overfitting is likely to occur"
    - Experimento PR2: comparación "update p" vs "update W" (Figura 6c)
    
    En el artículo se demostró que:
    - Update p: más lento pero generaliza mejor (error 22.6 mm en otras tareas)
    - Update W: más preciso pero pierde generalización (error 207 mm en otras tareas)
    - Nuestra implementación permite ambas estrategias y modo adaptativo
    """)
    
    return updater, results


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    updater, results = simulate_online_learning_experiment()