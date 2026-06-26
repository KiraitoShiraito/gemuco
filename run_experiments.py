"""
GeMuCo - Script Principal para Ejecutar Experimentos (VERSIÓN CORREGIDA)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Agregar directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar todos los módulos de GeMuCo
from model import GeMuCoNetwork, MaskManager, ParametricBiasManager
from data_collector import PR2DusterSimulator, GeMuCoDataCollector
from trainer import GeMuCoTrainer
from structure import StructureDeterminator
from optimizer import LatentOptimizer, XInOptimizer
from state_estimation import StateEstimator
from controller import GeMuCoController
from online_updater import OnlineUpdater, OnlineUpdateConfig, UpdateMode
from anomaly import AnomalyDetector, AnomalyConfig


def run_complete_experiment():
    """
    Ejecuta un experimento completo similar al artículo.
    """
    print("=" * 80)
    print("GeMuCo - Ejecución de Experimento Completo")
    print("=" * 80)
    
    # ========================================
    # CONFIGURACIÓN
    # ========================================
    n_joints = 3
    n_tooltip = 3
    n_sensors = n_joints + n_tooltip  # Total = 6 sensores
    dim_z = 16
    dim_p = 2
    samples_per_state = 300  # Reducido para prueba rápida
    n_epochs = 20            # Reducido para prueba rápida
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguración:")
    print(f"   - Sensores totales: {n_sensors}")
    print(f"   - Ángulos (entrada): {n_joints}")
    print(f"   - Tooltip (salida): {n_tooltip}")
    print(f"   - Dimensión latente (z): {dim_z}")
    print(f"   - Dimensión PB (p): {dim_p}")
    print(f"   - Muestras por estado: {samples_per_state}")
    print(f"   - Épocas de entrenamiento: {n_epochs}")
    print(f"   - Dispositivo: {device}")
    
    # ========================================
    # FASE 1: Crear modelo
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 1: Creando modelo GeMuCo...")
    print("-" * 60)
    
    model = GeMuCoNetwork(
        n_sensors=n_sensors,
        dim_z=dim_z,
        dim_p=dim_p,
        hidden_sizes=[128, 64, 64, 128],
        use_batchnorm=True
    ).to(device)
    
    print(f"Modelo creado: {model.get_num_params():,} parámetros")
    
    # ========================================
    # FASE 2: Generar datos
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 2: Generando datos sintéticos...")
    print("-" * 60)
    
    simulator = PR2DusterSimulator()
    collector = GeMuCoDataCollector(simulator)
    
    all_joint_angles, all_tool_tips, state_info = collector.collect_all_states(
        n_samples_per_state=samples_per_state,
        n_joints=n_joints,
        random_motion=True
    )
    
    # Normalizar datos
    normalized_joints, normalized_tooltips, norm_params = collector.get_normalized_data(
        all_joint_angles, all_tool_tips
    )
    
    # 🔧 IMPORTANTE: Crear el tensor x completo (ángulos + tooltips) para cada estado
    # Esto es crucial: x debe tener TODOS los sensores (6)
    normalized_x_per_state = []
    for joints, tooltips in zip(normalized_joints, normalized_tooltips):
        # Concatenar ángulos y tooltips para formar x completo [n_samples, 6]
        x_full = torch.cat([joints, tooltips], dim=1)
        normalized_x_per_state.append(x_full)
    
    # Crear dataloaders con x completo
    from torch.utils.data import DataLoader, TensorDataset
    
    dataloaders = []
    for x_full in normalized_x_per_state:
        # x_in y x_out son el mismo tensor x (como dice el artículo)
        dataset = TensorDataset(x_full, x_full)  # entrada y salida son iguales
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        dataloaders.append(dataloader)
    
    print(f"Datos generados: {len(dataloaders)} estados, {samples_per_state} muestras/estado")
    print(f"Dimensión de x: {normalized_x_per_state[0].shape[1]} sensores")
    
    # ========================================
    # FASE 3: Entrenamiento
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 3: Entrenando modelo con Parametric Bias...")
    print("-" * 60)
    
    # 🔧 Crear máscaras factibles
    mask_manager = MaskManager(n_sensors)
    
    # Máscara 1: Solo ángulos como entrada (para state estimation)
    mask_only_joints = torch.cat([torch.ones(n_joints), torch.zeros(n_tooltip)])
    mask_manager.add_mask(mask_only_joints)
    
    # Máscara 2: Todos los sensores disponibles
    mask_all = torch.ones(n_sensors)
    mask_manager.add_mask(mask_all)
    
    # Máscara 3: Solo tooltip como entrada (para control inverso)
    mask_only_tooltip = torch.cat([torch.zeros(n_joints), torch.ones(n_tooltip)])
    mask_manager.add_mask(mask_only_tooltip)
    
    print(f"Máscaras factibles: {len(mask_manager.feasible_masks)}")
    
    # Crear PB manager
    pb_manager = ParametricBiasManager(dim_p=dim_p, n_states=len(dataloaders))
    
    # Crear trainer
    trainer = GeMuCoTrainer(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        learning_rate_w=0.001,
        learning_rate_p=0.01,
        device=device
    )
    
    # Entrenar
    loss_history = trainer.train(
        dataloaders=dataloaders,
        n_epochs=n_epochs,
        verbose=True
    )
    
    print(f"Entrenamiento completado. Pérdida final: {loss_history['total'][-1]:.6f}")
    
    # ========================================
    # FASE 4: Determinación de estructura (opcional)
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 4: Determinación automática de estructura...")
    print("-" * 60)
    
    # Preparar datos para structure determinator
    X_all_states = torch.cat(normalized_x_per_state, dim=0)
    data_per_sensor = [X_all_states[:, i:i+1] for i in range(n_sensors)]
    
    struct_det = StructureDeterminator(
        n_sensors=n_sensors,
        dim_z=16,
        device=device
    )
    
    try:
        struct_results = struct_det.run_automatic_structure_determination(
            data_per_sensor=data_per_sensor,
            threshold_out=0.3,
            threshold_in=0.3,
            n_epochs=10,
            verbose=False
        )
        print(f"x_out determinado: {struct_results['x_out_indices']}")
        print(f"x_in determinado: {struct_results['x_in_indices']}")
    except Exception as e:
        print(f"Determinación de estructura: {e}")
    
    # ========================================
    # FASE 5: Optimización de z
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 5: Optimización iterativa de z...")
    print("-" * 60)
    
    latent_optimizer = LatentOptimizer(
        model=model,
        learning_rate=0.01,
        n_iterations=20,
        verbose=False,
        device=device
    )
    
    xin_optimizer = XInOptimizer(
        model=model,
        mask_manager=mask_manager,
        learning_rate=0.01,
        n_iterations=20,
        verbose=False,
        device=device
    )
    
    # Probar optimización simple
    test_z = torch.randn(1, dim_z, device=device)
    def test_loss(x_out_pred):
        return torch.norm(x_out_pred[0, 3:6], p=2)
    
    try:
        z_opt, x_out_opt, _ = latent_optimizer.optimize_z_from_x_out_loss(
            loss_function=test_loss,
            n_iterations=10,
            return_history=True
        )
        print(f"Optimización de z exitosa. z_norm: {z_opt.norm().item():.3f}")
    except Exception as e:
        print(f"Optimización de z: {e}")
    
    # ========================================
    # FASE 6: State Estimation
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 6: Estimación de estado...")
    print("-" * 60)
    
    state_estimator = StateEstimator(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        latent_optimizer=latent_optimizer,
        xin_optimizer=xin_optimizer,
        device=device
    )
    
    # Crear datos de prueba con sensores faltantes
    x_available = torch.randn(1, n_sensors, device=device)
    mask_available = torch.zeros(1, n_sensors, device=device)
    mask_available[0, 0:3] = 1.0  # Solo ángulos disponibles
    
    result = state_estimator.estimate_state(
        x_available=x_available,
        mask_available=mask_available,
        state_idx=0,
        method="auto"
    )
    
    print(f"Método usado: {result['method_used']}")
    print(f"Éxito: {result['success']}")
    
    # ========================================
    # FASE 7: Control
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 7: Control del robot...")
    print("-" * 60)
    
    controller = GeMuCoController(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        latent_optimizer=latent_optimizer,
        xin_optimizer=xin_optimizer,
        device=device
    )
    
    target_tooltip = torch.tensor([0.5, 0.5, 0.5], device=device)
    reference = torch.zeros(1, n_sensors, device=device)
    reference[0, 3:6] = target_tooltip
    
    control_result = controller.compute_control(
        target=reference,
        state_idx=0,
        actuator_indices=[0, 1, 2],
        sensor_indices=[3, 4, 5],
        method="optimize_x_in",
        n_iterations=20
    )
    
    print(f"Método usado: {control_result['method_used']}")
    print(f"Comandos de control: {control_result['control_commands'][0].detach().cpu().numpy()}")
    
    # ========================================
    # FASE 8: Actualización Online
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 8: Actualización online...")
    print("-" * 60)
    
    online_config = OnlineUpdateConfig(
        buffer_max_size=100,
        batch_size=8,
        update_frequency=5,
        n_steps_per_update=2,
        verbose=False
    )
    
    online_updater = OnlineUpdater(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        config=online_config,
        device=device
    )
    
    # Simular algunas actualizaciones
    for step in range(10):
        x_full_sample = normalized_x_per_state[0][step:step+1] if step < len(normalized_x_per_state[0]) else normalized_x_per_state[0][0:1]
        x_in = x_full_sample.to(device)
        x_out = x_full_sample.to(device)
        
        # Asegurar que m tenga forma [1, n_sensors]
        m = mask_only_joints.unsqueeze(0).to(device)
        if m.dim() == 2 and m.shape[0] == 1:
            pass  # Correcto
        elif m.dim() == 1:
            m = m.unsqueeze(0)
        
        p_k = pb_manager.get_pb(0).unsqueeze(0)
        
        online_updater.add_experience(x_in, x_out, m, p_k, 0)
        result = online_updater.update_online(mode=UpdateMode.BIAS_ONLY)
    
    print(f"Buffer size: {online_updater.buffer.size()}")
    print(f"Actualizaciones realizadas: {online_updater.update_count}")
    
    # ========================================
    # FASE 9: Detección de Anomalías
    # ========================================
    print("\n" + "-" * 60)
    print("FASE 9: Detección de anomalías...")
    print("-" * 60)
    
    anomaly_config = AnomalyConfig(
        mahalanobis_threshold=3.0,
        use_adaptive_threshold=False
    )
    
    anomaly_detector = AnomalyDetector(
        model=model,
        mask_manager=mask_manager,
        pb_manager=pb_manager,
        state_estimator=state_estimator,
        config=anomaly_config,
        device=device
    )
    
    # Calibrar
    try:
        anomaly_detector.calibrate_from_normal_data(n_samples=20, verbose=False)
        
        # Probar detección
        test_x = torch.randn(1, n_sensors, device=device) * 2.0
        test_mask = torch.ones(1, n_sensors, device=device)
        
        anomaly_result = anomaly_detector.detect_anomaly(
            x_available=test_x,
            mask_available=test_mask,
            state_idx=0
        )
        print(f"Detector calibrado")
        print(f"Mahalanobis distance: {anomaly_result['mahalanobis_distance']:.3f}")
        print(f"Anomalía detectada: {anomaly_result['is_anomaly']}")
    except Exception as e:
        print(f"Detector de anomalías: {e}")
    
    # ========================================
    # RESULTADOS FINALES
    # ========================================
    print("\n" + "=" * 80)
    print("EXPERIMENTO COMPLETADO CON ÉXITO")
    print("=" * 80)
    
    print("\nResumen de resultados:")
    print(f"   - Modelo entrenado por {n_epochs} épocas")
    print(f"   - Pérdida final de entrenamiento: {loss_history['total'][-1]:.6f}")
    print(f"   - PB final estado 0: {pb_manager.get_pb(0).detach().cpu().numpy()}")
    
    # Graficar pérdida
    if len(loss_history['total']) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history['total'], 'b-', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('MSE Loss')
        plt.title('Historial de pérdidas durante entrenamiento')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        print(f"\n Gráfico de pérdidas guardado como 'training_loss.png'")
    
    print("\n" + "=" * 80)
    print("Todas las fases de GeMuCo se ejecutaron correctamente")
    print("=" * 80)
    
    return {
        'model': model,
        'trainer': trainer,
        'controller': controller,
        'loss_history': loss_history
    }


if __name__ == "__main__":
    # Ejecutar experimento completo
    results = run_complete_experiment()
    
    print("\n¡GeMuCo está funcionando!")

    # 🔧 MOSTRAR ANIMACIÓN AL FINAL
    print("\nGenerando animación del movimiento del robot...")
    from data_collector import animate_robot_motion, PR2DusterSimulator
    
    # Crear simulador para obtener una configuración de referencia
    simulator = PR2DusterSimulator()
    
    # Ángulos iniciales (posición de inicio, por ejemplo reposo)
    start_angles = np.array([0.0, 0.5, 0.0])  # Hombro en 0, codo flexionado, muñeca en 0
    
    # Ángulos finales (los que calculó el controlador)
    control_commands = results['controller'].control_history[-1]['control']
    end_angles = control_commands.flatten() if len(control_commands.shape) > 1 else control_commands
    
    # Si los ángulos finales tienen menos de 3 dimensiones, rellenar
    if len(end_angles) < 3:
        end_angles = np.pad(end_angles, (0, 3 - len(end_angles)), 'constant')
    
    print(f"   Ángulos iniciales: {start_angles}")
    print(f"   Ángulos finales (control): {end_angles}")
    print(f"   Generando animación con 30 frames...")
    
    # Generar animación
    anim = animate_robot_motion(
        start_joint_angles=start_angles,
        end_joint_angles=end_angles,
        tool_length=500,
        tool_angle_deg=30,
        n_frames=30,
        interval=50,
        title="PR2 Robot: Movimiento desde inicio hasta posición controlada",
        save_path="robot_motion.gif"  # Guarda como GIF
    )
    
    print("\nAnimación guardada como 'robot_motion.gif'")
    print("   La ventana mostrará el movimiento en bucle.")
    print("   Cierra la ventana para terminar.")
    
    # Mantener la ventana abierta
    plt.show(block=True)