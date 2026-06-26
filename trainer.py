"""03
GeMuCo: Generalized Multisensory Correlational Model              Fase 3: Trainer - Entrenamiento offline con sesgo paramétrico

Basado en la sección II-D del artículo:
    "When parametric bias p is used as input, it is necessary to add one more step to the training method.
    Let D_k = {x_1, x_2, ..., x_{T_k}} (1 ≤ k ≤ K) where K is the total number of states
    and T_k is the number of data in the state k.
    Using this data D, we simultaneously update the network weight W and parametric bias p_k."

También:
    "p_k is trained with an initial value of 0."
    "We simultaneously update the network weight W and parametric bias p_k."
"""

import torch  # Para tensores y redes neuronales
import torch.nn as nn  # Contiene capas y componentes para construir redes neuronales
import torch.optim as optim  # Contiene algoritmos de optimización para entrenar redes neuronales
from torch.utils.data import DataLoader  # Para entregar datos por lotes durante el entrenamiento
import numpy as np  # Para cálculos matemáticos y manejo de arreglos numéricos
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional  # Tipos utilizados para documentar entradas y salidas
from collections import defaultdict  # Diccionario que crea listas vacías automáticamente
import time  # Medir tiempos de ejecución

# Importar los módulos de las fases anteriores (deben estar en el mismo directorio)
try:  # Los intenta importar...
    from model import GeMuCoNetwork, MaskManager, ParametricBiasManager  # Clases creadas en la Fase 1
    from data_collector import GeMuCoDataCollector, PR2DusterSimulator  # Clases creadas en la Fase 2
except ImportError:
    print("Importando módulos locales...")
    # Si no se pueden importar, las definiciones están más abajo (en un entorno real, usaríamos los imports)

class GeMuCoTrainer:  # Entrenar la red neuronal GeMuCo. Maneja train offline (con múltiples estados y p) y el online (actualización incremental).
    """    
    Según el artículo (sección II-D):
        "We simultaneously update the network weight W and parametric bias p_k."
        "p_k is trained with an initial value of 0."
    """
    
    def __init__(  # Método que se ejecuta al crear el entrenador
        self,  # Referencia al propio objeto
        model: GeMuCoNetwork,  # Modelo de red neuronal a entrenar
        mask_manager: MaskManager,  # Gestor de máscaras (para obtenerlas durante el train)
        pb_manager: ParametricBiasManager,  # Gestor de sesgo paramétrico (p_k para cada estado)
        learning_rate_w: float = 0.001,  # Velocidad de aprendizaje para los pesos W de la red
        learning_rate_p: float = 0.01,  # Velocidad de aprendizaje para los sesgos paramétricos
        device: str = "cuda" if torch.cuda.is_available() else "cpu"  # GPU si existe, sino CPU
    ):
        self.model = model.to(device)  # Mueve la red neuronal al dispositivo seleccionado
        self.mask_manager = mask_manager  # Guarda referencia al gestor de máscaras
        self.pb_manager = pb_manager  # Guarda referencia al gestor de sesgo paramétrico
        self.device = device  # Guarda el dispositivo que se usará
        
        # Optimizadores separados como sugiere el artículo (W y p se actualizan a la vez, pero pueden tener diferentes tazas de aprendizaje)
        self.optimizer_w = optim.Adam(model.parameters(), lr=learning_rate_w)  # Crea el optimizador que ajustará los pesos de la red
        
        # Los p son parámetros entrenables dentro de pb_manager
        self.optimizer_p = optim.Adam(pb_manager.get_all_pb(), lr=learning_rate_p)  # Crea el optimizador que ajustará los sesgos paramétricos
        
        self.criterion = nn.MSELoss() # Func de pérdida: MSE (como dice el artículo)
        
        self.loss_history = defaultdict(list) # Dicc donde se almacenará el historial de pérdidas. Para tracking del training

    
    def train_epoch(  # Entrena la red durante una época completa con todos los estados
        self,  # Referencia al propio objeto
        dataloaders: List[DataLoader],  # Lista de DataLoaders, uno por estado
        masks_per_state: Optional[List[torch.Tensor]] = None,  # Posibles máscaras específicas para cada estado (opcional)
        epoch: int = 0  # No. de época actual
    ) -> Dict[str, float]:  # Regresa estadísticas del entrenamiento
        """        
        Para cada estado k:
            1. Obtiene el sesgo paramétrico p_k correspondiente
            2. Para cada batch, aplica máscaras aleatorias
            3. Se calcula pérdida y actualiza W y p_k
        
        Regresa:
            Diccionario con pérdidas promedio por estado y total
        """
        total_loss = 0.0  # Acumulador para la pérdida total
        total_samples = 0  # Contador total de muestras procesadas
        per_state_loss = {}  # Diccionario para guardar la pérdida de cada estado
        
        for state_idx, dataloader in enumerate(dataloaders):  # Iterará sobre todos los estados disponibles (c/u tiene su Dataloader)
            state_loss = 0.0  # Acumulador de pérdida para este estado
            state_samples = 0  # Contador de muestras para este estado
            
            p_k = self.pb_manager.get_pb(state_idx)  # Obtiene el sesgo paramétrico para ese estado actual
            # Expandir a batch_size más adelante
            
            for batch_idx, (x_in, x_out_target) in enumerate(dataloader):  # Iterará sobre todos los lotes del estado
                x_in = x_in.to(self.device)  # Mueve las entradas al dispositivo seleccionado
                x_out_target = x_out_target.to(self.device)  # Mueve las salidas objetivo al dispositivo seleccionado
                batch_size = x_in.shape[0]  # Obtiene cuántas muestras tiene este lote
                
                p_batch = p_k.unsqueeze(0).expand(batch_size, -1)  # Copia y expande el sesgo paramétrico para cada muestra del lote
                
                # (como dice el artículo: "m is randomly selected from M each time")
                m = self.mask_manager.get_random_mask(batch_size).to(self.device)  # Obtiene y genera máscaras aleatorias para las muestras
                
                x_out_pred, _ = self.model(x_in, m, p_batch)  # Ejecuta la red neuronal y obtiene la pred (Forward pass)
                
                loss = self.criterion(x_out_pred, x_out_target)  # Calcula el error entre pred y resultado esperado (pérdida)
                
                # Backward pass (actualizar W y p_k a la vez)
                self.optimizer_w.zero_grad()  # Borra gradientes anteriores de los pesos
                self.optimizer_p.zero_grad()  # Borra gradientes anteriores de los sesgos paramétricos
                loss.backward()  # Calcula cómo deben ajustarse los parámetros para reducir el error
                self.optimizer_w.step()  # Actualiza los pesos de la red neuronal
                self.optimizer_p.step()  # Actualiza los sesgos paramétricos
                
                state_loss += loss.item() * batch_size  # Acumula la pérdida ponderada por cantidad de muestras
                state_samples += batch_size  # Acumula muestras procesadas del estado
                total_loss += loss.item() * batch_size  # Acumula pérdida global
                total_samples += batch_size  # Acumula muestras globales
            
            per_state_loss[state_idx] = state_loss / state_samples if state_samples > 0 else 0.0  # Calcula la pérdida promedio del estado
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0  # Calcula la pérdida promedio global
        
        # Guardar historial
        self.loss_history['total'].append(avg_loss)  # Guarda la pérdida global en el historial
        for state_idx, loss_val in per_state_loss.items():  # Iterará sobre las pérdidas de todos los estados
            self.loss_history[f'state_{state_idx}'].append(loss_val)  # Guarda cada pérdida individual
        
        return {  # Regresa estadísticas del entrenamiento
            'total_loss': avg_loss,  # Pérdida promedio total
            'per_state_loss': per_state_loss  # Pérdida promedio por estado
        }
    
    def train(  # Método principal de entrenamiento completo offline
        self,  # Referencia al propio objeto
        dataloaders: List[DataLoader],  # Lista de conjuntos de datos organizados por estado
        n_epochs: int = 100,  # No. de épocas a entrenar
        verbose: bool = True,  # Si se mostrará progreso
        save_path: Optional[str] = None  # Ruta opcional para guardar el modelo entrenado
    ) -> Dict[str, List[float]]:  # Regresa el historial de pérdidas
        """
        Según el artículo: "we prepare a set of feasible masks M. Then, for each x_in we use each m in M to mask a part of the corresponding x_in"
        """
        print(f"Iniciando entrenamiento por {n_epochs} épocas...")
        print(f"  - Estados: {len(dataloaders)}")  # Muestra cuántos estados existen
        print(f"  - Máscaras factibles: {len(self.mask_manager.feasible_masks)}")  # Cuántas máscaras pueden usarse
        print(f"  - Device: {self.device}")  # Y el dispositivo usado
        print(f"  - Learning rates: W={self.optimizer_w.param_groups[0]['lr']}, " # Info de las tazas de aprendizaje
              f"p={self.optimizer_p.param_groups[0]['lr']}")
        
        start_time = time.time()  # Guarda el instante de inicio para medir duración
        
        for epoch in range(n_epochs):  # Iterará sobre todas las épocas de entrenamiento
            self.model.train() # Pone la red en modo entrenamiento
            
            metrics = self.train_epoch(dataloaders, epoch=epoch) # Ejecuta y entrena una época completa
            
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):  # Decide cuándo mostrar info
                elapsed = time.time() - start_time  # Calcula cuánto tiempo ha pasado
                print(f"Epoch {epoch:3d}/{n_epochs} | " # Progreso del entrenamiento
                      f"Loss: {metrics['total_loss']:.6f} | "
                      f"Tiempo: {elapsed:.1f}s")
        
        if save_path:  # Si hay una ruta de guardado...
            self.save_model(save_path)  # Guarda el modelo entrenado
            print(f"Modelo guardado en {save_path}")
        
        print(f"\nEntrenamiento completado en {time.time() - start_time:.1f}s")  # Muestra duración total
        print(f"Pérdida final: {self.loss_history['total'][-1]:.6f}")  # Muestra la última pérdida obtenida

        return self.loss_history  # Regresa todo el historial de pérdidas
    
    def train_online_update_w(  # Actualiza solo los pesos W de la red neuronal durante aprendizaje online
        self,  # Referencia al propio objeto
        x_in: torch.Tensor,  # Datos de entrada
        x_out_target: torch.Tensor,  # Resultado esperado
        m: torch.Tensor,  # Máscara de sensores que indica qué info está disponible
        p: torch.Tensor,  # Sesgo paramétrico usado para representar el estado actual
        n_steps: int = 1,  # No. de actualizaciones consecutivas a realizar
        learning_rate: Optional[float] = None  # Velocidad de aprendizaje temporal opcional
    ) -> float:  # Regresa el valor final de la pérdida
        """
        El artículo menciona: "updating W or updating W and p simultaneously changes the structure of the entire network, and thus overfitting is likely to occur."
        
        Regresa: Pérdida final
        """
        if learning_rate is not None: # Si se especificó una tasa de aprendizaje temporal...
            original_lr = self.optimizer_w.param_groups[0]['lr']  # Guarda la velocidad original
            for param_group in self.optimizer_w.param_groups:  # Iterará sobre los grupos de parámetros del optimizador
                param_group['lr'] = learning_rate  # Sustituye temporalmente la taza de aprendizaje
        
        self.model.train()  # Coloca la red en modo entrenamiento
        
        for _ in range(n_steps):  # Repite el proceso el no. de veces indicado
            self.optimizer_w.zero_grad()  # Borra gradientes calculados anteriormente
            x_out_pred, _ = self.model(x_in, m, p)  # Ejecuta la red para obtener una predicción
            loss = self.criterion(x_out_pred, x_out_target)  # Calcula el error entre predicción y valor esperado
            loss.backward()  # Calcula cómo deben modificarse los pesos para reducir el error
            self.optimizer_w.step()  # Actualiza los pesos de la red
        
        if learning_rate is not None: # Si se había cambiado temporalmente la taza de aprendizaje
            for param_group in self.optimizer_w.param_groups:  # Recorre nuevamente los grupos de parámetros
                param_group['lr'] = original_lr  # Restaura la taza original
        
        return loss.item() # Regresa el valor numérico de la pérdida final
    
    def train_online_update_p(  # Actualiza sólo los sesgos paramétricos p_k durante aprendizaje online
        self,  # Referencia al propio objeto
        x_in: torch.Tensor,  # Datos de entrada
        x_out_target: torch.Tensor,  # Resultado esperado
        m: torch.Tensor,  # Máscara de sensores
        state_idx: int,  # Índice del estado cuyo sesgo paramétrico p_k se obtendrá y será ajustado
        n_steps: int = 1,  # No. de actualizaciones consecutivas
        learning_rate: Optional[float] = None  # Taza de aprendizaje temporal opcional
    ) -> float:  # Regresa el valor final de la pérdida
        """
        El artículo menciona: "updating only p, only some dynamics are changed and the structure of the overall network is kept the same, thus overfitting is unlikely to occur."
        
        Regresa: Pérdida final
        """
        if learning_rate is not None:  # Si se indicó una taza temporal...
            original_lr = self.optimizer_p.param_groups[0]['lr']  # Guarda la taza original
            for param_group in self.optimizer_p.param_groups:  # Recorre los grupos del optimizador
                param_group['lr'] = learning_rate  # Sustituye temporalmente la taza
        
        self.model.train()  # Coloca la red en modo entrenamiento
        
        p_k = self.pb_manager.get_pb(state_idx)  # Obtiene el sesgo paramétrico a actualizar, correspondiente al estado
        batch_size = x_in.shape[0]  # Obtiene la cantidad de muestras del lote
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1)  # Duplica el sesgo paramétrico para todas las muestras

        for _ in range(n_steps):  # Repite el proceso de actualización
            self.optimizer_p.zero_grad()  # Borra gradientes anteriores de los sesgos paramétricos
            x_out_pred, _ = self.model(x_in, m, p_batch)  # Genera una predicción usando la red
            loss = self.criterion(x_out_pred, x_out_target)  # Calcula el error de predicción
            loss.backward()  # Calcula cómo deben ajustarse los sesgos paramétricos
            self.optimizer_p.step()  # Actualiza únicamente los sesgos paramétricos
        
        if learning_rate is not None:  # Si se modificó temporalmente la taza de aprendizaje
            for param_group in self.optimizer_p.param_groups:  # Recorre los grupos del optimizador
                param_group['lr'] = original_lr  # Restaura la taza original
        
        return loss.item() # Regresa el valor numérico de la pérdida final
    
    def train_online_update_both(  # Actualización online simultánea de los pesos W de la red y los sesgos paramétricos p_k
        self,  # Referencia al propio objeto
        x_in: torch.Tensor,  # Datos de entrada
        x_out_target: torch.Tensor,  # Resultado esperado
        m: torch.Tensor,  # Máscara de sensores
        state_idx: int,  # Índice del estado asociado al sesgo paramétrico
        n_steps: int = 1  # No. de actualizaciones consecutivas
    ) -> Tuple[float, float]:  # Regresa las pérdidas finales obtenidas para ambos optimizadores (loss_w, loss_p)
        self.model.train()  # Coloca la red en modo entrenamiento
        
        p_k = self.pb_manager.get_pb(state_idx)  # Obtiene el sesgo paramétrico del estado actual
        batch_size = x_in.shape[0]  # Obtiene cuántas muestras hay en el lote
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1)  # Replica el sesgo paramétrico para todas las muestras
        
        for _ in range(n_steps):  # Repite el proceso las veces indicadas
            self.optimizer_w.zero_grad()  # Borra gradientes anteriores de los pesos
            x_out_pred, _ = self.model(x_in, m, p_batch)  # Ejecuta la red
            loss_w = self.criterion(x_out_pred, x_out_target)  # Calcula el error para actualizar pesos
            loss_w.backward()  # Calcula gradientes para los pesos
            self.optimizer_w.step()  # Actualiza los pesos W de la red
            
            self.optimizer_p.zero_grad()  # Borra gradientes anteriores de los sesgos paramétricos
            x_out_pred, _ = self.model(x_in, m, p_batch)  # Ejecuta nuevamente la red
            loss_p = self.criterion(x_out_pred, x_out_target)  # Calcula el error para actualizar sesgos paramétricos
            loss_p.backward()  # Calcula gradientes para los sesgos paramétricos
            self.optimizer_p.step()  # Actualiza los sesgos paramétricos
        
        return loss_w.item(), loss_p.item()  # Regresa ambas pérdidas finales
    
    def save_model(self, path: str):  # Guarda el modelo entrenado y los sesgos paramétricos
        torch.save({  # Guarda la info importante en un diccionario
            'model_state_dict': self.model.state_dict(),  # Guarda los pesos W de la red neuronal
            'pb_state_dict': self.pb_manager.get_all_pb(),  # Guarda todos los sesgos paramétricos
            'loss_history': dict(self.loss_history),  # Guarda el historial de pérdidas
            'n_states': self.pb_manager.n_states,  # Guarda la cantidad de estados existentes
            'dim_z': self.model.dim_z,  # Guarda el tamaño del espacio latente
            'dim_p': self.model.dim_p,  # Guarda la dim del sesgo paramétrico
            'n_sensors': self.model.n_sensors  # Guarda la cantidad de sensores usados
        }, path)  # Escribe toda la info en el archivo indicado
        print(f"Modelo guardado en {path}")  # Informa dónde se almacenó el modelo
    
    def load_model(self, path: str):  # Carga el modelo guardado y sus sesgos paramétricos
        checkpoint = torch.load(path, map_location=self.device)  # Lee el archivo y carga su contenido
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Restaura los pesos W de la red neuronal
        
        # Cargar sesgos paramétricos (asumiendo que ya están inicializados)
        for i, pb_param in enumerate(checkpoint['pb_state_dict']):  # Iterará sobre todos los sesgos paramétricos guardados
            if i < len(self.pb_manager.get_all_pb()):  # Verifica que exista un espacio correspondiente
                self.pb_manager.get_all_pb()[i].data.copy_(pb_param.data)  # Copia los valores almacenados
        
        self.loss_history = defaultdict(list, checkpoint.get('loss_history', {}))  # Recupera el historial de pérdidas
        print(f"Modelo cargado desde {path}")
    
    def plot_loss_history(self, figsize: tuple = (10, 6)):  # Visualiza la evolución del error durante el entrenamiento
        plt.figure(figsize=figsize)  # Crea una nueva figura
        
        plt.plot(self.loss_history['total'], label='Total', linewidth=2, color='black') # Traza la curva de pérdida total
        
        # Pérdidas por estado (muestra algunas para no saturar)
        n_states_to_show = min(5, len([k for k in self.loss_history.keys() if k.startswith('state_')]))
        colors = plt.cm.viridis(np.linspace(0, 1, n_states_to_show))
        
        for i, state_key in enumerate([k for k in self.loss_history.keys() if k.startswith('state_')][:n_states_to_show]): # Sobre los state mostrados
            plt.plot(self.loss_history[state_key], label=state_key, alpha=0.6, # Traza la curva correspondiente a un estado específico
                    linestyle='--', color=colors[i])
        
        plt.xlabel('Época')  # Etiqueta del eje horizontal
        plt.ylabel('MSE Loss')  # Etiqueta del eje vertical
        plt.title('Historial de pérdidas durante entrenamiento')
        plt.legend()
        plt.grid(True, alpha=0.3)  # Cuadrícula suave para facilitar la lectura
        plt.yscale('log')  # Usa escala logarítmica para visualizar mejor cambios pequeños y grandes
        plt.tight_layout()  # Ajusta automáticamente los espacios de la figura
        plt.show()
    
    def visualize_pb_online(  # Visualiza la trayectoria de los sesgos paramétricos en la actualización online en un plano bidimensional
        self,  # Referencia al propio objeto
        trajectory_states: List[int],  # Lista de índices estados recorridos durante una trayectoria, cuyos p se actualizaron online
        title: str = "Evolución de Parametric Bias durante actualización online"
    ):
        if self.pb_manager.dim_p > 2:  # Si los sesgos paramétricos tienen más de dos dimensiones
            # Reducir a 2D con PCA para visualización
            pb_matrix = self.pb_manager.get_pb_matrix().detach().cpu().numpy() # Obtiene todos los sesgos paramétricos como matriz NumPy
            
            pb_matrix = pb_matrix + np.random.randn(*pb_matrix.shape) * 1e-6 # Añade un poco de ruido para evitar problemas numéricos
            
            pca = PCA(n_components=2)  # PCA para reducir dimensiones a dos
            pb_2d = pca.fit_transform(pb_matrix)  # Convierte los sesgos paramétricos a coordenadas bidimensionales
            explained_variance = pca.explained_variance_ratio_  # Obtiene el % de info conservada en cada eje
        else:  # Si ya tienen dos dimensiones o menos
            pb_2d = self.pb_manager.get_pb_matrix().detach().cpu().numpy()  # Usa los valores originales
            explained_variance = [1.0, 1.0]  # Indica que se conserva toda la info
        
        plt.figure(figsize=(10, 8)) # Se crea una figura
        
        # Dibujar PB entrenados (estáticos)
        colors = plt.cm.viridis(np.linspace(0, 1, pb_2d.shape[0])) # Genera una serie de colores diferentes, uno por estado
        for i, (x, y) in enumerate(pb_2d): # Iterará sobre cada punto bidimensional
            plt.scatter(x, y, c=[colors[i]], s=100, edgecolors='black', zorder=3)
            plt.annotate(f"State {i}", (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        # Dibujar trayectorias (si se proporcionan)
        # Nota: Para trayectorias reales, necesitaríamos pasar los históricos
        # Por ahora, solo mostramos los puntos finales
        
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')  # Etiqueta del 1er componente principal
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')  # Etiqueta del 2do componente principal
        plt.title(title)
        plt.grid(True, alpha=0.3)  # Activa una cuadrícula ligera
        plt.tight_layout()  # Ajusta automáticamente los espacios
        plt.show()

# ============================================
# FUNCIONES DE UTILIDAD PARA PREPARAR DATOS
# ============================================

def prepare_masks_for_pr2_experiment(  # Prepara las máscaras de sensores para el experimento PR2
    n_joints: int = 3,  # No. de sensores de articulaciones
    n_tooltip: int = 3  # No. de sensores de posición de la punta del plumero
) -> MaskManager:  # Regresa gestor de máscaras
    """
    En el experimento (sección III-A.2), la red automáticamente determinó:
        - Entrada: solo θ (ángulos de articulaciones)
        - Salida: solo x_tool (posición de la punta)
    
    Por lo tanto, las máscaras factibles son aquellas que permiten inferir x_tool a partir de θ.
    """
    n_sensors = n_joints + n_tooltip  # Calcula el no. total de sensores (6)
    mask_manager = MaskManager(n_sensors)  # Crea un administrador para ese no. de sensores
    
    # Máscara para usar solo ángulos como entrada
    # [1,1,1,0,0,0] (3 ángulos, 3 tooltip)
    mask_only_joints = torch.cat([  # Construye una máscara que deja visibles solo los sensores articulares
        torch.ones(n_joints),  # Activa los sensores de articulaciones
        torch.zeros(n_tooltip)  # Desactiva los sensores de posición de la punta
    ])
    mask_manager.add_mask(mask_only_joints)  # Agrega la máscara al gestor
    
    # También se puede agregar máscaras con tooltip disponible (para state estimation)
    mask_all = torch.ones(n_sensors)  # Crea una máscara donde todos los sensores están activos
    mask_manager.add_mask(mask_all)  # Agrega la máscara al gestor
    
    # Máscara con solo tooltip (para control inverso)
    mask_only_tooltip = torch.cat([  # Construye una máscara que deja visibles solo los sensores de la punta
        torch.zeros(n_joints),  # Desactiva los sensores articulares
        torch.ones(n_tooltip)  # Activa los sensores de la punta
    ])
    mask_manager.add_mask(mask_only_tooltip)  # Agrega la máscara al gestor
    
    print(f"Máscaras preparadas para PR2 ({n_sensors} sensores totales):")  # Muestra info general
    for i, m in enumerate(mask_manager.feasible_masks):  # Iterará sobre todas las máscaras registradas
        print(f"  m{i}: {m.int().tolist()}")  # Muestra cada máscara como lista de ceros y unos

    return mask_manager  # Regresa el gestor configurado


def validate_training_results(  # Comprueba y valida qué tan bien funciona la red después del entrenamiento. Que tan bien predice x_tool desde θ
    model: GeMuCoNetwork,  # Modelo de red neuronal ya entrenada
    pb_manager: ParametricBiasManager,  # Gestor de sesgos paramétricos entrenados
    dataloaders: List[DataLoader],  # Conjuntos de datos usados para la evaluación
    device: str  # Dispositivo donde se ejecutará el cálculo (CPU o GPU)
):
    model.eval()  # Pone la red en modo evaluación para desactivar comportamientos exclusivos del entrenamiento
    m_only_joints = torch.cat([  # Crea una máscara que solo deja visibles los sensores articulares
        torch.ones(3),   # Activa los 3 sensores de articulaciones (ángulos)
        torch.zeros(3)   # Desactiva los 3 sensores de posición de la punta del plumero
    ]).to(device) # Mueve la máscara al dispositivo
    
    print("\n" + "=" * 60)
    print("Validación de resultados del entrenamiento")
    print("=" * 60)
    
    for state_idx, dataloader in enumerate(dataloaders):  # Iterará sobre todos los estados disponibles
        p_k = pb_manager.get_pb(state_idx) # Obtiene el sesgo paramétrico para este estado
        
        total_error = 0.0  # Acumulador para el error total del estado
        n_samples = 0  # Contador de muestras procesadas
        
        with torch.no_grad():  # Desactiva el cálculo de gradientes porque solo se está evaluando
            for x_in, x_out_target in dataloader:  # Iterará sobre todos los lotes de datos del estado
                x_in = x_in.to(device)  # Mueve las entradas al dispositivo
                x_out_target = x_out_target.to(device)  # Mueve las salidas objetivo al dispositivo
                batch_size = x_in.shape[0]  # Obtiene cuántas muestras hay en este lote
                
                p_batch = p_k.unsqueeze(0).expand(batch_size, -1)  # Duplica el sesgo paramétrico para cada muestra
                m_batch = m_only_joints.unsqueeze(0).expand(batch_size, -1)  # Duplica la máscara para cada muestra
                
                x_out_pred, _ = model(x_in, m_batch, p_batch)  # Genera una predicción usando la red entrenada
                
                # Error de predicción (en espacio original, no normalizado)
                error = torch.mean(torch.abs(x_out_pred - x_out_target)).item() # Calcula el error medio absoluto del lote
                total_error += error * batch_size  # Acumula el error ponderado por no. de muestras
                n_samples += batch_size  # Acumula la cantidad de muestras procesadas
        
        avg_error = total_error / n_samples if n_samples > 0 else 0.0  # Calcula el error promedio del estado
        print(f"Estado {state_idx}: Error medio absoluto = {avg_error:.4f}")

# ============================================
# EJEMPLO DE USO COMPLETO
# ============================================

def run_full_training_example(): # Ejecuta un ejemplo completo de entrenamiento con datos sintéticos
    print("=" * 70)
    print("GeMuCo - Fase 3: Entrenamiento offline con sesgo paramétrico")
    print("=" * 70)
    
    # 1. Configuración:
    n_joints = 3  # No. de sensores asociados a articulaciones
    n_tooltip = 3  # No. de sensores asociados a la posición de la punta del plumero
    n_sensors = n_joints + n_tooltip  # No. total de sensores
    dim_z = 16  # Tamaño del espacio latente interno z de la red
    dim_p = 2  # Dim de los vectores de sesgo paramétrico
    n_epochs = 50  # Cantidad de épocas de entrenamiento
    samples_per_state = 500  # Cantidad de ejemplos generados para cada estado (original son 1000)
    batch_size = 64  # Cantidad de muestras procesadas simultáneamente
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Selecciona GPU si existe, si no usa CPU
    
    print(f"\nConfiguración:")
    print(f"  - Sensores: {n_sensors} ({n_joints} ángulos, {n_tooltip} tooltip)")  # Muestra la configuración de sensores
    print(f"  - dim_z: {dim_z}")  # El tamaño del espacio latente
    print(f"  - dim_p: {dim_p}")  # La dim de sesgo paramétrico
    print(f"  - Device: {device}")
    
    # 2. Crear modelo
    model = GeMuCoNetwork(  # Crea una nueva red neuronal GeMuCo
        n_sensors=n_sensors,  # No. total de sensores
        dim_z=dim_z,  # Tamaño del espacio latente z
        dim_p=dim_p,  # Tamaño del sesgo paramétrico
        hidden_sizes=[128, 64, 64, 128],  # Tamaños de las capas ocultas
        use_batchnorm=True  # Activa Batch Normalization
    )
    print(f"\nModelo creado: {model.get_num_params():,} parámetros")  # Muestra cuántos parámetros tiene la red
    
    mask_manager = prepare_masks_for_pr2_experiment(n_joints, n_tooltip) # Crea las máscaras necesarias
    
    # 4. Generar datos
    print("\nGenerando datos sintéticos...")
    simulator = PR2DusterSimulator()  # Crea el simulador del robot con plumero
    collector = GeMuCoDataCollector(simulator)  # Crea el recolector de datos usando el simulador
    
    all_joint_angles, all_tool_tips, state_info = collector.collect_all_states(  # Genera datos para todos los estados
        n_samples_per_state=samples_per_state,  # Cantidad de muestras por estado
        n_joints=n_joints,  # No. de articulaciones simuladas
        random_motion=True  # Usa movimientos aleatorios
    )
    
    normalized_joints, normalized_tooltips, norm_params = collector.get_normalized_data(  # Normaliza todos los datos generados
        all_joint_angles, all_tool_tips  # Ángulos originales y posiciones originales
    )
    
    # Crear dataloaders (uno por estado)
    dataloaders = collector.create_dataloaders(  # Convierte los datos normalizados en DataLoaders
        normalized_joints, normalized_tooltips, batch_size=batch_size  # Entradas normalizadas, salidas normalizadas, tamaño de lote usado
    )
    print(f"  - Estados: {len(dataloaders)}")  # Muestra cuántos estados fueron generados
    print(f"  - Muestras por estado: {samples_per_state}")  # Cuántas muestras tiene cada estado
    print(f"  - Total muestras: {len(dataloaders) * samples_per_state}")  # Cantidad total de muestras generadas
    
    # 5. Crea gestor de sesgo paramétrico
    pb_manager = ParametricBiasManager(dim_p=dim_p, n_states=len(dataloaders)) # Dim de cada sesgo, no. de estados existentes
    print(f"\nParametric Bias Manager: {pb_manager.n_states} estados, dim={dim_p}")
    
    # 6. Crear trainer y entrenar
    trainer = GeMuCoTrainer(  # Crea el entrenador principal
        model=model,  # Red neuronal que será entrenada
        mask_manager=mask_manager,  # Gestor de máscaras
        pb_manager=pb_manager,  # Gestor de sesgo paramétrico
        learning_rate_w=0.001,  # Tasa de aprendizaje para pesos W de la red
        learning_rate_p=0.01,  # Tasa de aprendizaje para sesgo paramétrico p
        device=device  # Dispositivo
    )
    
    # Entrenar
    loss_history = trainer.train(  # Inicia el entrenamiento principal de la red neuronal
        dataloaders=dataloaders,  # Datos de entrenamiento organizados por estados
        n_epochs=n_epochs,  # Cantidad de épocas que se entrenará la red
        verbose=True,  # Mensajes de progreso durante el entrenamiento
        save_path="gemuco_pr2_trained.pt"  # Nombre del archivo donde se guardará el modelo entrenado
    )
    
    trainer.plot_loss_history()  # Gráfica con la evolución del error durante el entrenamiento
    
    trainer.visualize_pb_online([])  # Visualización de los sesgos paramétricos aprendidos
    
    validate_training_results(model, pb_manager, dataloaders, device) # Evalúa qué tan bien funciona el modelo entrenado
    
    # 10. Mostrar PB finales
    print("\nParametric Biases finales entrenados:")
    pb_matrix = pb_manager.get_pb_matrix().detach().cpu().numpy()  # Obtiene todos los sesgos como matriz NumPy
    for i, (pb, info) in enumerate(zip(pb_matrix, state_info)):  # Itera simultáneamente sobre los sesgos y la info de cada estado
        print(f"  Estado {i}: {pb} | l={info['tool_length']}mm, φ={info['tool_angle_deg']}°") 
    
    print("\nFase 3 completada con éxito!")
    
    return trainer, dataloaders, state_info # Regresa los objetos principales

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__": # Se ejecuta sólo cuando este archivo se corre directamente
    trainer, dataloaders, state_info = run_full_training_example() # Ejecuta todo el ejemplo completo y guarda los resultados
    
    print("\n" + "=" * 70)
    print("Resumen de la Fase 3")
    print("=" * 70)
    print("""
    Implementado:
    ✓ Entrenamiento offline con múltiples estados
    ✓ Actualización simultánea de W y p_k
    ✓ Máscaras aleatorias durante entrenamiento
    ✓ Optimizadores separados para W y p (con diferentes LRs)
    ✓ Guardado y carga de modelos
    ✓ Visualización de pérdidas
    ✓ Visualización de PB con PCA
    ✓ Validación de resultados
    
    Pendiente para próximas fases:
    - Determinación automática de estructura (Fase 4)
    - Optimización iterativa de z (Fase 5)
    - State estimation (Fase 6)
    - Control (Fase 7)
    - Actualización online (Fase 8, ya tenemos los métodos)
    - Anomaly detection (Fase 9)
    """)