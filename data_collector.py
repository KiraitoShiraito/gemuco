"""02
GeMuCo: Generalized Multisensory Correlational Model              Fase 2: Data Collector - Generación de datos sintéticos

Basado en el experimento de PR2 con plumero (sección III-A del artículo):
    - Entrada (x_in): ángulos de articulaciones (θ)
    - Salida (x_out): posición de la punta del plumero (x_tool)
    - Relación: x_tool = f(θ, l_tool, φ_tool)  donde l_tool = longitud agarrada, φ_tool = ángulo de agarre
    - Se varía el estado de agarre (grasping state) para usar sesgo paramétrico

Este script genera datos para múltiples estados (diferentes l_tool, φ_tool)
"""

import numpy as np  # Trabajar con vectores, matrices y op mat
import torch  # Para tensores y aprendizaje profundo
from torch.utils.data import DataLoader, TensorDataset  # Organizar datos en lotes de entrenamiento
from typing import Tuple, List, Dict, Optional  # Tipos de datos usados como ayuda para documentar código
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('TkAgg')  # Forza el uso del backend interactivo


class PR2DusterSimulator:  # Simulador del robot PR2 sosteniendo un plumero
    """
    El plumero tiene:
        - Un stick de longitud variable (l_tool) dependiendo de dónde se agarra
        - Un ángulo de agarre (φ_tool) en un grado de libertad
        - La punta del plumero (x_tool) se calcula como:
            x_tool = posición de la muñeca + offset (l_tool) en dirección del stick
            con corrección por gravedad (el plumero cuelga hacia abajo)
    
    En el artículo (página 8):
        "the tool-tip position x_tool is assumed to be 100 mm below the tip of the stick"
        "l_tool = {300, 500, 700} [mm] and φ_tool = {0, 30, 60} [deg]"
    """
    
    def __init__(self): # Método que se ejecuta automáticamente al crear el simulador
        # Parámetros del robot PR2 (simplificados)
        # En un caso real, el PR2 tiene 7 grados de libertad en el brazo
        # Aquí se simula un brazo planar con 2-3 articulaciones para simplificar
        
        # Longitudes de los segmentos del brazo (mm)
        self.shoulder_to_elbow = 300.0  # Distancia del hombro al codo (húmero) en milímetros
        self.elbow_to_wrist = 250.0     # Distancia del codo a la muñeca (cúbito/radio) en milímetros
        self.wrist_offset = 50.0        # Desplazamiento adicional de la muñeca en el eje Z
        
        # Gravedad (afecta cómo cuelga el plumero)
        self.gravity_direction = np.array([0.0, -1.0, 0.0])  # Dirección en la que actúa la gravedad (hacia abajo en Y)
        
        # Offset adicional de la punta del plumero por gravedad (mm)
        self.cloth_droop = 100.0  # # Cuánto cuelga hacia abajo la tela del plumero por a la gravedad, 100 mm hacia abajo (como dice el artículo)
        
        # Estados de agarre posibles (como en el artículo)
        self.tool_lengths = [300, 500, 700]  # Longitudes posibles del palo del plumero, mm
        self.tool_angles = [0, 30, 60]       # Ángulos posibles con los que se sostiene el plumero, grados
        
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:  # Calcula la pos de la muñeca a partir de los áng de las articulaciones
        batch_size = joint_angles.shape[0]  # Obtiene cuántas muestras se están procesando   joint_angles: [batch_size, n_joints] ángulos en radianes
        wrist_pos = np.zeros((batch_size, 3))  # Reserva espacio para guardar las posiciones calculadas      Posición XYZ de la muñeca (mm)
        
        # Modelo cinemático simplificado (brazo planar de 2 grados de libertad para XY, más Z simple)
        for i in range(batch_size):  # Iterará sobre cada muestra una por una
            theta1 = joint_angles[i, 0]  # Obtiene el ángulo del hombro (pitch)
            theta2 = joint_angles[i, 1] if joint_angles.shape[1] > 1 else 0  # Obtiene el ángulo del codo si existe
            
            # Posición del codo
            elbow = np.array([  # Calcula la posición del codo
                self.shoulder_to_elbow * np.cos(theta1),  # Coordenada X del codo
                self.shoulder_to_elbow * np.sin(theta1),  # Coordenada Y del codo
                0.0  # Coordenada Z del codo
            ])
            
            # Posición de la muñeca (relativa al codo)
            wrist = elbow + np.array([  # Calcula la posición de la muñeca sumando el segundo segmento del brazo
                self.elbow_to_wrist * np.cos(theta1 + theta2),  # Coordenada X adicional
                self.elbow_to_wrist * np.sin(theta1 + theta2),  # Coordenada Y adicional
                0.0  # Coordenada Z adicional
            ])
            
            if joint_angles.shape[1] > 2:  # Si hay una tercera articulación (rotación z)
                wrist[2] = self.wrist_offset * np.sin(joint_angles[i, 2])  # Ajusta la coordenada Z de la muñeca
            
            wrist_pos[i] = wrist # Guarda la posición calculada
        
        return wrist_pos # Regresa todas las posiciones de muñeca calculadas
    
    def compute_tool_tip_position(  # Calcula la posición de la punta del plumero
        self,  # Referencia al propio objeto
        wrist_pos: np.ndarray,  # Posiciones de la muñeca [batch_size, 3]
        joint_angles: np.ndarray,  # Ángulos de las articulaciones [batch_size, n_joints]
        tool_length: float,  # Longitud del plumero desde la mano (mm)
        tool_angle_deg: float  # Ángulo con el que se sostiene el plumero (grados)
    ) -> np.ndarray:  # Regresa las posiciones de la punta del plumero
        """
        Según el artículo (página 8):
            "the tool-tip position x_tool is assumed to be 100 mm below the tip of the stick"
            + la herramienta puede tener un ángulo de agarre adicional
        """
        batch_size = wrist_pos.shape[0] # Obtiene el número de muestras
        tool_tip = np.zeros((batch_size, 3)) # Posición de la punta [batch_size, 3]    Reserva espacio para guardar las posiciones finales
        tool_angle_rad = np.deg2rad(tool_angle_deg)  # Convierte grados a radianes
        
        for i in range(batch_size):  # Iterará sobre todas las muestras
            # Dirección del brazo (desde hombro a muñeca)
            arm_direction = wrist_pos[i] - np.array([0, 0, 0])  # Obtiene la dirección desde el hombro en origen hasta la muñeca
            if np.linalg.norm(arm_direction) > 0:  # Verifica que la dirección no sea un vector nulo
                arm_direction = arm_direction / np.linalg.norm(arm_direction)  # Normaliza el vector para que mida 1
            
            # Aplicar rotación adicional por el ángulo de agarre (simplificado: rotación en el plano XY)
            grip_direction = np.array([  # Calcula la dirección del plumero considerando el ángulo de agarre
                arm_direction[0] * np.cos(tool_angle_rad) - arm_direction[1] * np.sin(tool_angle_rad),  # Nueva componente X
                arm_direction[0] * np.sin(tool_angle_rad) + arm_direction[1] * np.cos(tool_angle_rad),  # Nueva componente Y
                arm_direction[2]  # Conserva la componente Z
            ])
            
            stick_tip = wrist_pos[i] + grip_direction * tool_length  # Calcula la posición de la punta del palo (sin considerar el colgajo)
            
            # Aplicar el colgajo del plumero (100 mm hacia abajo por gravedad)
            # Como dice el artículo: "the tool-tip position x_tool is assumed to be 100 mm below the tip"
            tool_tip[i] = stick_tip + self.gravity_direction * self.cloth_droop  # Aplica la caída de la tela por gravedad
        
        return tool_tip  # Regresa las posiciones finales de la punta del plumero
    
    def generate_random_joint_angles(  # Genera ángulos aleatorios para las articulaciones, dentro de rangos realistas
        self,  # Referencia al propio objeto
        n_samples: int,  # No. de muestras a generar
        n_joints: int = 3  # No. de articulaciones
    ) -> np.ndarray:  # Regresa una matriz de ángulos
        """
        El artículo dice (página 8):
            "the joint angle is moved randomly"
        
        Rangos típicos para brazo de robot:
            - Hombro pitch: -90° a 90° (-π/2 a π/2)
            - Codo pitch: 0° a 150° (0 a 2.6 rad)
            - Muñeca yaw: -90° a 90°
        """
        angles = np.zeros((n_samples, n_joints))  # Crea una matriz vacía para almacenar los ángulos
        
        angles[:, 0] = np.random.uniform(-np.pi/2, np.pi/2, n_samples) # Genera áng aleat para el hombro
        
        angles[:, 1] = np.random.uniform(0, 2.6, n_samples) # Genera áng aleat para el codo
        
        # Muñeca (yaw) - opcional
        if n_joints > 2:  # Si hay una tercera articulación
            angles[:, 2] = np.random.uniform(-np.pi/2, np.pi/2, n_samples)  # Genera áng aleat para la muñeca
        
        return angles  # Regresa todos los ángulos generados
    
    def generate_trajectory_joint_angles(  # Genera una trayectoria suave de ángulos para simular movimiento para las articulaciones
        self,  # Referencia al propio objeto
        n_steps: int,  # Cantidad de pasos de la trayectoria
        n_joints: int = 3  # No. de articulaciones
    ) -> np.ndarray:  # Regresa una secuencia de ángulos
        t = np.linspace(0, 2*np.pi, n_steps)  # Genera valores distribuidos uniformemente entre 0 y 2π
        angles = np.zeros((n_steps, n_joints))  # Crea una matriz vacía para guardar los ángulos
        
        angles[:, 0] = 0.5 * np.sin(t)  # Movimiento sinusoidal suave para el hombro
        angles[:, 1] = 0.3 * np.sin(1.5*t) + 0.5  # Movimiento sinusoidal suave para el codo con desplazamiento
        if n_joints > 2:  # Si hay una tercera articulación
            angles[:, 2] = 0.2 * np.sin(2*t)  # Movimiento sinusoidal para la muñeca
        
        return angles  # Regresa la trayectoria generada
    
    def collect_data_for_state(  # Genera y recolecta datos para una configuración/estado específica de agarre del plumero
        self,  # Referencia al propio objeto
        tool_length: float,  # Longitud del plumero
        tool_angle_deg: float,  # Ángulo del plumero
        n_samples: int,  # No. de muestras a generar
        n_joints: int = 3,  # No. de articulaciones
        random_motion: bool = True  # Indica si se usarán movimientos aleatorios
    ) -> Tuple[np.ndarray, np.ndarray]:  # Regresa ángulos y posiciones finales
        """
        Regresa:
            - joint_angles: [n_samples, n_joints]
            - tool_tip_pos: [n_samples, 3]
        
        Según el artículo (página 8):
            "we set x = [x_tool, θ]. x_tool is the tool-tip position and θ is the 7-dimensional joint angle"
        """
        if random_motion:  # Si se eligió movimiento aleatorio...
            joint_angles = self.generate_random_joint_angles(n_samples, n_joints)  # Genera ángulos aleatorios
        else:  # Si se eligió movimiento siguiendo una trayectoria...
            joint_angles = self.generate_trajectory_joint_angles(n_samples, n_joints)  # Genera trayectoria suave
        
        wrist_pos = self.forward_kinematics(joint_angles)  # Calcula posiciones de la muñeca
        
        tool_tip_pos = self.compute_tool_tip_position(  # Calcula posiciones de la punta del plumero
            wrist_pos, joint_angles, tool_length, tool_angle_deg  # Datos necesarios para el cálculo
        )
        
        return joint_angles, tool_tip_pos  # Regresa los ángulos utilizados y las posiciones obtenidas


class GeMuCoDataCollector:  # Recolectar y preparar los datos que usará la red GeMuCo para entrenamiento
    """
    Según el artículo (sección II-D):
        "we take data while changing the state of the body, tools, and environment.
        Let D_k = {x_1, x_2, ..., x_{T_k}} (1 ≤ k ≤ K) where K is the total number of states
        and T_k is the number of data in the state k."
    
    También (sección III-A.2):
        "1000 data points per grasping state are collected, which amounts to 9000 data points in total"
    """
    
    def __init__(self, simulator: PR2DusterSimulator):  # Método que se ejecuta al crear el recolector
        self.simulator = simulator  # Guarda una referencia al simulador PR2 con plumero
        self.data_states = []  # Lista de diccionarios donde se almacenarán los datos recolectados para cada estado
    
    def collect_all_states(  # Recolecta datos para todas las combinaciones posibles de estados de agarre de plumero
        self,  # Referencia al propio objeto
        n_samples_per_state: int = 1000,  # Cantidad de muestras que se generarán para cada estado
        n_joints: int = 3,  # No. de articulaciones del brazo
        random_motion: bool = True  # Indica si se usarán movimientos aleatorios
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:  # Regresa ángulos, posiciones y datos descriptivos
        all_joint_angles = []  # Lista de arrays donde se guardarán todos los ángulos generados por estado
        all_tool_tips = []  # Lista de arrays donde se guardarán todas las posiciones de la punta por estado del plumero
        state_info = []  # Lista de diccionarios donde se guardará info descriptiva de cada estado
        
        state_id = 0  # ID único para cada estado
        for l_tool in self.simulator.tool_lengths:  # Iterará sobre todas las longitudes posibles del plumero
            for phi_tool in self.simulator.tool_angles:  # Iterará sobre todos los ángulos posibles del plumero
                print(f"Recolectando datos para estado {state_id}: l_tool={l_tool}mm, φ_tool={phi_tool}°")  # Muestra info del estado actual
                
                joint_angles, tool_tip_pos = self.simulator.collect_data_for_state(  # Solicita al simulador generar datos
                    tool_length=l_tool,  # Longitud actual del plumero
                    tool_angle_deg=phi_tool,  # Ángulo actual del plumero
                    n_samples=n_samples_per_state,  # Cantidad de muestras a generar
                    n_joints=n_joints,  # No. de articulaciones
                    random_motion=random_motion  # Tipo de movimiento a usar
                )
                
                all_joint_angles.append(joint_angles)  # Guarda los ángulos generados para este estado
                all_tool_tips.append(tool_tip_pos)  # Guarda las posiciones de la punta para este estado
                state_info.append({  # Guarda info descriptiva del estado
                    'state_id': state_id,  # ID del estado
                    'tool_length': l_tool,  # Longitud utilizada
                    'tool_angle_deg': phi_tool,  # Ángulo utilizado
                    'n_samples': n_samples_per_state  # Cantidad de muestras generadas
                })
                
                state_id += 1  # Incrementa el ID para el siguiente estado
        
        self.data_states = list(zip(all_joint_angles, all_tool_tips, state_info))  # Agrupa toda la info de cada estado
        
        return all_joint_angles, all_tool_tips, state_info  # Regresa todos los datos recolectados
    
    def get_normalized_data(  # Normaliza los datos para que tengan escalas similares y los convierte a tensores de PyTorch
        self,  # Referencia al propio objeto
        all_joint_angles: List[np.ndarray],  # Lista de ángulos articulares
        all_tool_tips: List[np.ndarray]  # Lista de posiciones de la punta del plumero
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:  # Regresa datos normalizados y parámetros de normalización
        """
        El artículo menciona (página 3):
            "Note that x is assumed to be normalized using all the data obtained."
        
        Regresa:
            - normalized_joints: tensor [total_samples, n_joints] normalizado
            - normalized_tooltips: tensor [total_samples, 3] normalizado
            - norm_params: diccionario con medias y stds para desnormalizar después
        """
        # Concatenar todos los datos
        all_joints_concat = np.vstack(all_joint_angles)  # Une todos los ángulos en una sola matriz grande
        all_tooltips_concat = np.vstack(all_tool_tips)  # Une todas las posiciones en una sola matriz grande
        
        # Calcular medias y stds
        joint_mean = np.mean(all_joints_concat, axis=0)  # Calcula el promedio de cada ángulo
        joint_std = np.std(all_joints_concat, axis=0) + 1e-8  # Calcula la STD evitando división por cero
        tooltip_mean = np.mean(all_tooltips_concat, axis=0)  # Calcula el promedio de cada coordenada de posición
        tooltip_std = np.std(all_tooltips_concat, axis=0) + 1e-8  # Calcula la STD evitando división por cero
        
        # Normalizar
        normalized_joints_list = []  # Almacenarán los ángulos normalizados
        normalized_tooltips_list = []  # Almacenarán las posiciones normalizadas
        
        for joints, tooltips in zip(all_joint_angles, all_tool_tips):  # Iterará simultáneamente sobre cada conjunto de datos
            norm_joints = (joints - joint_mean) / joint_std  # Convierte los ángulos a una escala normalizada
            norm_tooltips = (tooltips - tooltip_mean) / tooltip_std  # Convierte las posiciones a una escala normalizada
            
            normalized_joints_list.append(torch.tensor(norm_joints, dtype=torch.float32))  # Convierte los ángulos normalizados a tensor
            normalized_tooltips_list.append(torch.tensor(norm_tooltips, dtype=torch.float32))  # Convierte las posiciones normalizadas a tensor
        
        norm_params = {  # Diccionario con la info necesaria para desnormalizar después
            'joint_mean': joint_mean,  # Promedios de los ángulos
            'joint_std': joint_std,  # STD de los ángulos
            'tooltip_mean': tooltip_mean,  # Promedios de las posiciones
            'tooltip_std': tooltip_std  # STD de las posiciones
        }
        
        return normalized_joints_list, normalized_tooltips_list, norm_params  # Regresa datos normalizados y estadísticas
    
    def create_dataloaders(  # Convierte los datos en DataLoaders de PyTorch para entrenamiento eficiente de cada estado (cada state tiene su p)
        self,  # Referencia al propio objeto
        normalized_joints: List[torch.Tensor],  # Ángulos normalizados
        normalized_tooltips: List[torch.Tensor],  # Posiciones normalizadas
        batch_size: int = 64,  # Cantidad de muestras que se entregarán juntas
        shuffle: bool = True  # Indica si los datos deben mezclarse aleatoriamente
    ) -> List[DataLoader]:  # Regresa una lista de DataLoaders
        dataloaders = []  # Guardará todos los DataLoaders
        
        for joints, tooltips in zip(normalized_joints, normalized_tooltips):  # Iterará sobre cada conjunto de datos
            dataset = TensorDataset(joints, tooltips)  # Crea un conjunto de datos que empareja entradas y salidas
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # Crea un cargador de datos
            dataloaders.append(dataloader)  # Guarda el DataLoader creado
        
        return dataloaders  # Regresa todos los DataLoaders listos para entrenamiento


class OnlineDataBuffer:  # Funciona como una memoria temporal para almacenar datos nuevos durante la ejecución para la actualización online
    """
    El artículo menciona:
        "When the number of data exceeds a determined threshold,
        data is discarded from the oldest."
    """

    from collections import deque
    
    def __init__(self, max_size: int = 1000):  # Método que se ejecuta al crear el buffer
        self.max_size = max_size  # Guarda la cantidad máxima de elementos que se pueden almacenar
        self.buffer_x = deque(maxlen=max_size)  # Cola que almacena las entradas y elimina automáticamente las más antiguas cuando se llena
        self.buffer_y = deque(maxlen=max_size)  # Cola que almacena las salidas correspondientes
    
    def add_data(self, x: torch.Tensor, y: torch.Tensor):  # Agrega una sola muestra al buffer
        self.buffer_x.append(x.clone())  # Guarda una copia de la entrada
        self.buffer_y.append(y.clone())  # Guarda una copia de la salida
    
    def add_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor):  # Agrega varias muestras al mismo tiempo
        for i in range(x_batch.shape[0]):  # Iterará sobre cada elemento del lote
            self.buffer_x.append(x_batch[i].clone())  # Guarda una copia de la entrada actual
            self.buffer_y.append(y_batch[i].clone())  # Guarda una copia de la salida actual
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:  # Obtiene todos los datos almacenados
        if len(self.buffer_x) == 0:  # Si el buffer está vacío
            return torch.tensor([]), torch.tensor([])  # Regresa tensores vacíos
        
        x_tensor = torch.stack(list(self.buffer_x))  # Une todas las entradas en un único tensor
        y_tensor = torch.stack(list(self.buffer_y))  # Une todas las salidas en un único tensor
        return x_tensor, y_tensor # Regresa todas las entradas y salidas almacenadas
    
    def get_random_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:  # Obtiene un lote aleatorio de muestras
        if len(self.buffer_x) < batch_size:  # Si hay menos datos de los solicitados
            return self.get_all_data()  # Regresa todos los datos disponibles
        
        indices = np.random.choice(len(self.buffer_x), batch_size, replace=False)  # Selecciona índices aleatorios sin repetir
        x_batch = torch.stack([self.buffer_x[i] for i in indices])  # Construye el lote de entradas
        y_batch = torch.stack([self.buffer_y[i] for i in indices])  # Construye el lote de salidas
        return x_batch, y_batch  # Regresa el lote aleatorio
    
    def clear(self):  # Vacía y limpia completamente el buffer
        self.buffer_x.clear()  # Elimina todas las entradas almacenadas
        self.buffer_y.clear()  # Elimina todas las salidas almacenadas
    
    def size(self) -> int:  # Obtiene la cantidad de elementos almacenados
        return len(self.buffer_x)  # Regresa el no. de muestras guardadas

# ============================================
# VISUALIZACIÓN DE DATOS
# ============================================

def visualize_tool_tip_positions(  # Func. para visualizar las posiciones de la punta del plumero para diferentes estados en 3D
    all_tool_tips: List[np.ndarray],  # Lista con todas las posiciones calculadas
    state_info: List[Dict],  # Info descriptiva de cada estado
    title: str = "Tool-tip positions for different grasping states"
):
    """
    Esto ayuda a entender cómo varía x_tool con l_tool y φ_tool.
    """
    fig = plt.figure(figsize=(12, 8))  # Crea una figura
    ax = fig.add_subplot(111, projection='3d')  # Agrega un sistema de ejes tridimensional
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_tool_tips)))  # Genera colores distintos para cada estado
    
    for i, (tooltips, info) in enumerate(zip(all_tool_tips, state_info)):  # Iterará simultáneamente sobre los datos y descripción
        ax.scatter(  # Traza los puntos correspondientes a este estado
            tooltips[:, 0], tooltips[:, 1], tooltips[:, 2],  # Coordenadas X, Y y Z
            c=[colors[i]], marker='.', alpha=0.3, s=1,  # Configuración visual de los puntos
            label=f"l={info['tool_length']}mm, φ={info['tool_angle_deg']}°"  # Texto que aparecerá en la leyenda
        )
    
    ax.set_xlabel('X (mm)')  # Nombre del eje X
    ax.set_ylabel('Y (mm)')  # Nombre del eje Y
    ax.set_zlabel('Z (mm)')  # Nombre del eje Z
    ax.set_title(title)  # Coloca el título de la gráfica
    ax.legend(markerscale=5, fontsize=8)  # Muestra la leyenda con el significado de cada color
    plt.show()


def visualize_joint_angles_distribution(  # Func para visualizar cómo se distribuyen los ángulos de las articulaciones
    all_joint_angles: List[np.ndarray],  # Lista con todos los ángulos generados
    state_info: List[Dict]  # Info descriptiva de cada estado
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # Crea una figura con tres gráficas colocadas horizontalmente
    
    for i, (joints, info) in enumerate(zip(all_joint_angles, state_info)):  # Iterará simultáneamente sobre los datos y descripción
        for j in range(min(3, joints.shape[1])):  # Iterará como máximo sobre las tres primeras articulaciones
            axes[j].hist(joints[:, j], alpha=0.3, bins=30, # Traza un histograma para la articulación actual
                        label=f"State {i}: l={info['tool_length']}, φ={info['tool_angle_deg']}°")
    
    axes[0].set_title('Joint 0 (Shoulder)')  # Título para la articulación del hombro
    axes[1].set_title('Joint 1 (Elbow)')  # Articulación del codo
    axes[2].set_title('Joint 2 (Wrist)')  # La de la muñeca
    
    for ax in axes:  # Iterará sobre cada gráfica individual
        ax.set_xlabel('Angle (rad)')  # Nombre del eje horizontal
        ax.set_ylabel('Frequency')  # Nombre del eje vertical
        ax.legend(fontsize=7)  # Muestra la leyenda correspondiente
    
    plt.tight_layout()  # Ajusta automáticamente los espacios para evitar superposiciones
    plt.show()

def visualize_robot_3d(
    joint_angles: np.ndarray,
    tool_tip: np.ndarray,
    title: str = "PR2 Robot with Duster"
):
    """
    Visualización 3D simple del robot con plumero usando matplotlib.
    
    Args:
        joint_angles: Ángulos de articulaciones [n_joints] o [batch, n_joints]
        tool_tip: Posición de la punta [x, y, z]
        title: Título del gráfico
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Cinemática directa simplificada (igual que en el simulador)
    shoulder = np.array([0, 0, 0])
    theta1 = joint_angles[0] if len(joint_angles) > 0 else 0
    theta2 = joint_angles[1] if len(joint_angles) > 1 else 0
    
    elbow = np.array([
        300 * np.cos(theta1),
        300 * np.sin(theta1),
        0
    ])
    
    wrist = elbow + np.array([
        250 * np.cos(theta1 + theta2),
        250 * np.sin(theta1 + theta2),
        0
    ])
    
    # Graficar segmentos del brazo
    ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], [shoulder[2], elbow[2]], 
            'b-', linewidth=4, label='Upper arm')
    ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], [elbow[2], wrist[2]], 
            'g-', linewidth=4, label='Forearm')
    
    # Graficar el plumero (desde muñeca hasta punta)
    ax.plot([wrist[0], tool_tip[0]], [wrist[1], tool_tip[1]], [wrist[2], tool_tip[2]], 
            'r-', linewidth=3, label='Duster (stick + cloth)')
    
    # Graficar articulaciones como puntos
    ax.scatter(*shoulder, s=100, c='black', marker='o', label='Shoulder')
    ax.scatter(*elbow, s=80, c='blue', marker='o', label='Elbow')
    ax.scatter(*wrist, s=80, c='green', marker='o', label='Wrist')
    ax.scatter(*tool_tip, s=150, c='red', marker='^', label='Tool tip')
    
    # Configurar gráfico
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([-400, 800])
    ax.set_ylim([-400, 800])
    ax.set_zlim([-200, 400])
    
    # Ajustar ángulo de vista
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show()

def animate_robot_motion(
    start_joint_angles: np.ndarray,
    end_joint_angles: np.ndarray,
    tool_length: float = 500,
    tool_angle_deg: float = 30,
    n_frames: int = 50,
    interval: int = 50,
    title: str = "PR2 Robot Motion Simulation",
    save_path: Optional[str] = None,
    show_info: bool = True,
    show_target: bool = True
):
    """
    Versión mejorada de la animación con información en tiempo real.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    
    # Crear figura con dos subplots
    if show_info:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121, projection='3d')
        ax_info = fig.add_subplot(122)
        ax_info.axis('off')
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Paleta de colores
    COLORS = {
        'upper_arm': '#2196F3',
        'forearm': '#4CAF50',
        'duster': '#F44336',
        'shoulder': '#1A1A2E',
        'elbow': '#1565C0',
        'wrist': '#2E7D32',
        'tip': '#D32F2F',
        'target': '#FFD700',
        'trajectory': '#FF6F00',
        'grid': '#E0E0E0'
    }
    
    # Configuración del brazo
    shoulder_to_elbow = 300.0
    elbow_to_wrist = 250.0
    cloth_droop = 100.0
    shoulder = np.array([0, 0, 0])
    tool_angle_rad = np.deg2rad(tool_angle_deg)
    
    # Calcular trayectoria
    frames = np.linspace(0, 1, n_frames)
    joint_trajectory = []
    tool_tip_trajectory = []
    all_angles = []
    
    for t in frames:
        current_angles = start_joint_angles + t * (end_joint_angles - start_joint_angles)
        theta1 = current_angles[0]
        theta2 = current_angles[1] if len(current_angles) > 1 else 0
        theta3 = current_angles[2] if len(current_angles) > 2 else 0
        
        elbow = np.array([
            shoulder_to_elbow * np.cos(theta1),
            shoulder_to_elbow * np.sin(theta1),
            0
        ])
        
        wrist = elbow + np.array([
            elbow_to_wrist * np.cos(theta1 + theta2),
            elbow_to_wrist * np.sin(theta1 + theta2),
            0
        ])
        
        # Añadir efecto de muñeca si existe tercera articulación
        if len(current_angles) > 2:
            wrist[2] = 50 * np.sin(theta3)
        
        arm_direction = wrist / (np.linalg.norm(wrist) + 1e-8)
        grip_direction = np.array([
            arm_direction[0] * np.cos(tool_angle_rad) - arm_direction[1] * np.sin(tool_angle_rad),
            arm_direction[0] * np.sin(tool_angle_rad) + arm_direction[1] * np.cos(tool_angle_rad),
            arm_direction[2]
        ])
        
        stick_tip = wrist + grip_direction * tool_length
        tool_tip = stick_tip + np.array([0.0, -cloth_droop, 0.0])
        
        joint_trajectory.append({
            'shoulder': shoulder.copy(),
            'elbow': elbow,
            'wrist': wrist,
            'theta1': theta1,
            'theta2': theta2,
            'theta3': theta3
        })
        tool_tip_trajectory.append(tool_tip)
        all_angles.append(current_angles)
    
    # Configurar ejes
    ax.set_xlim([-400, 800])
    ax.set_ylim([-400, 800])
    ax.set_zlim([-200, 400])
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.view_init(elev=20, azim=-60)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Agregar suelo
    xx, yy = np.meshgrid(np.linspace(-400, 800, 5), np.linspace(-400, 800, 5))
    zz = np.full_like(xx, -200)
    ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')
    
    # Elementos de la animación
    arm_line, = ax.plot([], [], [], color=COLORS['upper_arm'], linewidth=5, label='Upper arm')
    forearm_line, = ax.plot([], [], [], color=COLORS['forearm'], linewidth=5, label='Forearm')
    duster_line, = ax.plot([], [], [], color=COLORS['duster'], linewidth=4, label='Duster')
    
    shoulder_point, = ax.plot([], [], [], 'o', color=COLORS['shoulder'], markersize=12, label='Shoulder')
    elbow_point, = ax.plot([], [], [], 'o', color=COLORS['elbow'], markersize=10, label='Elbow')
    wrist_point, = ax.plot([], [], [], 'o', color=COLORS['wrist'], markersize=10, label='Wrist')
    tip_point, = ax.plot([], [], [], '^', color=COLORS['tip'], markersize=14, label='Tool tip')
    
    tip_trajectory_line, = ax.plot([], [], [], color=COLORS['trajectory'], linewidth=2, 
                                   linestyle='--', alpha=0.7, label='Tip trajectory')
    
    # Esfera objetivo
    if show_target:
        target_pos = tool_tip_trajectory[-1]
        target_sphere = ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], 
                                   s=500, c=COLORS['target'], alpha=0.2, marker='o', label='Target')
    
    # Texto de información (en la ventana 3D)
    info_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, 
                         fontsize=10, verticalalignment='top', 
                         bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    # Leyenda personalizada
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['upper_arm'], lw=4, label='Brazo superior (húmero)'),
        plt.Line2D([0], [0], color=COLORS['forearm'], lw=4, label='Antebrazo (cúbito/radio)'),
        plt.Line2D([0], [0], color=COLORS['duster'], lw=3, label='Plumero (stick + tela)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['shoulder'], 
                   markersize=10, label='Articulaciones'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['tip'], 
                   markersize=12, label='Punta del plumero'),
    ]
    if show_target:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['target'], 
                       markersize=12, alpha=0.3, label='Objetivo')
        )
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    # Función de inicialización
    def init():
        arm_line.set_data([], [])
        arm_line.set_3d_properties([])
        forearm_line.set_data([], [])
        forearm_line.set_3d_properties([])
        duster_line.set_data([], [])
        duster_line.set_3d_properties([])
        shoulder_point.set_data([], [])
        shoulder_point.set_3d_properties([])
        elbow_point.set_data([], [])
        elbow_point.set_3d_properties([])
        wrist_point.set_data([], [])
        wrist_point.set_3d_properties([])
        tip_point.set_data([], [])
        tip_point.set_3d_properties([])
        tip_trajectory_line.set_data([], [])
        tip_trajectory_line.set_3d_properties([])
        info_text.set_text("")
        return arm_line, forearm_line, duster_line, shoulder_point, elbow_point, wrist_point, tip_point, tip_trajectory_line, info_text
    
    # Función de actualización
    def update(frame):
        joints = joint_trajectory[frame]
        tip_pos = tool_tip_trajectory[frame]
        
        # Actualizar líneas
        arm_line.set_data([shoulder[0], joints['elbow'][0]], [shoulder[1], joints['elbow'][1]])
        arm_line.set_3d_properties([shoulder[2], joints['elbow'][2]])
        
        forearm_line.set_data([joints['elbow'][0], joints['wrist'][0]], [joints['elbow'][1], joints['wrist'][1]])
        forearm_line.set_3d_properties([joints['elbow'][2], joints['wrist'][2]])
        
        duster_line.set_data([joints['wrist'][0], tip_pos[0]], [joints['wrist'][1], tip_pos[1]])
        duster_line.set_3d_properties([joints['wrist'][2], tip_pos[2]])
        
        # Actualizar puntos
        shoulder_point.set_data([shoulder[0]], [shoulder[1]])
        shoulder_point.set_3d_properties([shoulder[2]])
        elbow_point.set_data([joints['elbow'][0]], [joints['elbow'][1]])
        elbow_point.set_3d_properties([joints['elbow'][2]])
        wrist_point.set_data([joints['wrist'][0]], [joints['wrist'][1]])
        wrist_point.set_3d_properties([joints['wrist'][2]])
        tip_point.set_data([tip_pos[0]], [tip_pos[1]])
        tip_point.set_3d_properties([tip_pos[2]])
        
        # Trayectoria
        trajectory_points = np.array(tool_tip_trajectory[:frame+1])
        tip_trajectory_line.set_data(trajectory_points[:, 0], trajectory_points[:, 1])
        tip_trajectory_line.set_3d_properties(trajectory_points[:, 2])
        
        # Información en tiempo real
        progress = frame / n_frames * 100
        info_str = (
            f"Progreso: {progress:.1f}%\n"
            f"θ₁: {joints['theta1']:.3f} rad\n"
            f"θ₂: {joints['theta2']:.3f} rad\n"
            f"θ₃: {joints['theta3']:.3f} rad\n"
            f"Tip: ({tip_pos[0]:.1f}, {tip_pos[1]:.1f}, {tip_pos[2]:.1f})"
        )
        info_text.set_text(info_str)
        
        # Si hay panel de información
        if show_info:
            ax_info.clear()
            ax_info.axis('off')
            progress_bar = '█' * int(progress / 5) + '░' * (20 - int(progress / 5))
            info_panel = f"""
┌──────────────────────────────────────────────────┐
│                  DATOS DEL ROBOT                  │
├──────────────────────────────────────────────────┤
│ Ángulos articulares:                             │
│   • Hombro (θ₁):  {joints['theta1']:>7.3f} rad    │
│   • Codo (θ₂):    {joints['theta2']:>7.3f} rad    │
│   • Muñeca (θ₃):  {joints['theta3']:>7.3f} rad    │
├──────────────────────────────────────────────────┤
│ Posiciones:                                      │
│   • Codo:   ({joints['elbow'][0]:>6.1f}, {joints['elbow'][1]:>6.1f}, {joints['elbow'][2]:>6.1f}) │
│   • Muñeca: ({joints['wrist'][0]:>6.1f}, {joints['wrist'][1]:>6.1f}, {joints['wrist'][2]:>6.1f}) │
│   • Punta:  ({tip_pos[0]:>6.1f}, {tip_pos[1]:>6.1f}, {tip_pos[2]:>6.1f}) │
├──────────────────────────────────────────────────┤
│ Progreso: [{progress_bar}] {progress:.1f}%      │
│ Frame: {frame:>3d}/{n_frames}                    │
└──────────────────────────────────────────────────┘
"""
            ax_info.text(0.5, 0.5, info_panel, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace')
        
        # Actualizar esfera objetivo
        if show_target:
            target_pos = tool_tip_trajectory[-1]
            target_sphere._offsets3d = ([target_pos[0]], [target_pos[1]], [target_pos[2]])
        
        return arm_line, forearm_line, duster_line, shoulder_point, elbow_point, wrist_point, tip_point, tip_trajectory_line, info_text
    
    # Crear animación
    anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, interval=interval,
        blit=True, repeat=True
    )
    
    # Guardar
    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"📹 Animación guardada en {save_path}")
    
    plt.tight_layout()
    plt.show()
    return anim

# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":  # Solo se ejecuta cuando el archivo se corre directamente
    print("=" * 60)
    print("Testing GeMuCo Data Collector - Fase 2")
    print("=" * 60)
    
    # Crear simulador
    simulator = PR2DusterSimulator()  # Crea una instancia del simulador del robot PR2 con plumero
    print(f"\nSimulador PR2 con plumero creado:")
    print(f"  - Tool lengths: {simulator.tool_lengths} mm")  # Muestra las longitudes disponibles del plumero
    print(f"  - Tool angles: {simulator.tool_angles}°")  # Muestra los ángulos disponibles del plumero
    print(f"  - Cloth droop: {simulator.cloth_droop} mm")  # Muestra cuánto cae la tela por efecto de la gravedad
    
    collector = GeMuCoDataCollector(simulator)  # Crea el recolector de datos usando el simulador
    
    # Recolectar datos para todos los estados (como en el artículo: 1000 por estado)
    n_samples_per_state = 500  # No. de ej que se generarán para cada estado durante la prueba. 500 para prueba rápida (el artículo usa 1000)
    print(f"\nRecolectando {n_samples_per_state} muestras por estado...")
    
    all_joint_angles, all_tool_tips, state_info = collector.collect_all_states(  # Genera datos para todos los estados posibles
        n_samples_per_state=n_samples_per_state,  # Cantidad de muestras por estado
        n_joints=3,  # No. de articulaciones simuladas
        random_motion=True  # Usa movimientos aleatorios
    )
    
    print(f"\nDatos recolectados:")  # Muestra un encabezado informativo
    print(f"  - Total estados: {len(all_joint_angles)}")  # Cuántos estados fueron generados
    print(f"  - Muestras por estado: {all_joint_angles[0].shape[0]}")  # Cuántas muestras tiene cada estado
    print(f"  - Dimensión de ángulos: {all_joint_angles[0].shape[1]}")  # Cuántas articulaciones tiene cada muestra
    print(f"  - Dimensión de tool-tip: {all_tool_tips[0].shape[1]}")  # Cuántas coordenadas tiene cada posición del plumero
    
    normalized_joints, normalized_tooltips, norm_params = collector.get_normalized_data(  # Normaliza todos los datos recolectados
        all_joint_angles, all_tool_tips  # Datos originales que serán normalizados
    )
    
    print(f"\nNormalización completada:")
    print(f"  - Joint mean: {norm_params['joint_mean']}")  # Muestra el promedio de los ángulos articulares
    print(f"  - Joint std: {norm_params['joint_std']}")  # La STD de los ángulos articulares
    print(f"  - Tooltip mean: {norm_params['tooltip_mean']}")  # El promedio de las posiciones de la punta del plumero
    print(f"  - Tooltip std: {norm_params['tooltip_std']}")  # La STD de las posiciones de la punta
    
    dataloaders = collector.create_dataloaders(  # Crea DataLoaders para facilitar el entrenamiento posterior
        normalized_joints, normalized_tooltips, batch_size=32
    )
    print(f"\nDataLoaders creados: {len(dataloaders)}")
    
    # Visualizar datos
    print("\nGenerando visualizaciones...")
    visualize_tool_tip_positions(all_tool_tips, state_info)  # Muestra una gráfica 3D de las posiciones de la punta del plumero
    visualize_joint_angles_distribution(all_joint_angles, state_info)  # Muestra histogramas con la distribución de los ángulos articulares

    # Después de recolectar datos, visualizar una muestra
    simulator = PR2DusterSimulator()
    angles, tip = simulator.collect_data_for_state(
        tool_length=500, tool_angle_deg=30, n_samples=1
    )
    visualize_robot_3d(angles[0], tip[0], "PR2 with Duster (l=500mm, φ=30°)")

    # Probar OnlineDataBuffer
    buffer = OnlineDataBuffer(max_size=100)  # Crea un buffer capaz de almacenar hasta 100 muestras
    for i in range(50):  # Repite el proceso 50 veces
        x_sample = torch.randn(3)  # Genera una muestra de entrada aleatoria de tamaño 3
        y_sample = torch.randn(3)  # Genera una muestra de salida aleatoria de tamaño 3
        buffer.add_data(x_sample, y_sample)  # Guarda ambas muestras dentro del buffer
    
    print(f"\nOnlineDataBuffer:")
    print(f"  - Buffer size: {buffer.size()}/{buffer.max_size}")  # Muestra cuántos elementos contiene actualmente el buffer
    x_all, y_all = buffer.get_all_data()  # Recupera todos los datos almacenados en el buffer
    print(f"  - All data shape: {x_all.shape}")  # Muestra las dim de todas las entradas almacenadas

    x_batch, y_batch = buffer.get_random_batch(10)  # Obtiene un lote aleatorio de 10 muestras
    print(f"  - Random batch shape: {x_batch.shape}")  # Muestra las dim del lote obtenido
    
    print("\nFase 2 completada con éxito!")
    print("\nResumen de datos generados:")
    print("  - Simula el experimento PR2 con plumero")
    print("  - 9 estados (3 longitudes × 3 ángulos)")
    print("  - Datos normalizados listos para entrenamiento")
    print("  - Buffer para actualización online implementado")