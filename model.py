"""01
GeMuCo: Generalized Multisensory Correlational Model.     Fase 1: Red neuronal (Encoder-Decoder con máscara y sesgo paramétrico)

Estructura general:
    Entrada: [x_in (sensores/actuadores), m (máscara), p (sesgo paramétrico)]
        ↓
    Encoder (capas lineales + ReLU + BatchNorm)
        ↓
    z (estado latente)
        ↓
    Decoder (capas lineales + ReLU + BatchNorm)
        ↓
    Salida: x_out (predicción de todos los sensores/actuadores)
"""

import torch  # Trabajar con tensores y redes neuronales
import torch.nn as nn  # Contiene capas y componentes para crear redes neuronales
import torch.nn.functional as F  # Func. para op. mat. en redes neuronales
from typing import Tuple, Optional, List  # Indicar tipos de datos esperados


class GeMuCoNetwork(nn.Module): # Define una red neuronal completa con encoder y decoder, "GeMuCoNetwork"
    def __init__(  # Método, se ejecuta al crear un objeto de esta clase
        self,  # Referencia al propio objeto
        n_sensors: int,  # No. total de sensores/actuadores de entrada (dim. de x)
        dim_z: int = 32,  # Dim. del espacio latente interno z
        dim_p: int = 2,  # Dim. del vector de sesgo paramétrico p
        hidden_sizes: List[int] = [128, 64, 64, 128],  # No. de neuronas en las capas ocultas: [encoder_h1, encoder_h2, decoder_h1, decoder_h2]
        use_batchnorm: bool = True,  # Se usará normalización por lotes
        activation: nn.Module = nn.ReLU,  # Func. de activación utilizada en la red
        device: str = "cpu"
    ):
        super(GeMuCoNetwork, self).__init__()  # Inicializa la clase base nn.Module
        
        self.n_sensors = n_sensors  # Guarda el no. de sensores
        self.dim_z = dim_z  # Guarda el tamaño del espacio latente z
        self.dim_p = dim_p  # Guarda el tamaño del vector p
        self.use_batchnorm = use_batchnorm  # Guarda si se utilizará BatchNorm
        self.device = device
        
        input_dim = n_sensors + n_sensors + dim_p # La entrada total es: x_in (n_sensors) + m (n_sensors) + p (dim_p)
        
        # ============================================
        # ENCODER: input -> z
        # ============================================
        encoder_layers = [] # Guardar las capas del codificador
        
        # Capa 1: input -> hidden_size[0]
        encoder_layers.append(nn.Linear(input_dim, hidden_sizes[0]))  # Crea la 1ra capa lineal del codificador
        if use_batchnorm:  # Si BatchNorm está activado
            encoder_layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Agrega normalización para estabilizar el aprendizaje
        encoder_layers.append(activation())  # Agrega la func. de activación
        
        # Capa 2: hidden_size[0] -> hidden_size[1]
        encoder_layers.append(nn.Linear(hidden_sizes[0], hidden_sizes[1]))  # 2da capa lineal del codificador
        if use_batchnorm:  # Si BatchNorm está activado
            encoder_layers.append(nn.BatchNorm1d(hidden_sizes[1]))  # Agrega normalización
        encoder_layers.append(activation())  # Agrega la func. de activación
        
        # Capa 3 (latente): hidden_size[1] -> dim_z (sin activación, es el embedding)
        encoder_layers.append(nn.Linear(hidden_sizes[1], dim_z))  # Última capa que produce la representación latente
        # Nota: No se pone activación en la capa latente para que z sea libre
        
        self.encoder = nn.Sequential(*encoder_layers)  # Une todas las capas del codificador en secuencia
        
        # ============================================
        # DECODER: z -> x_out
        # ============================================
        decoder_layers = []  # Guardará las capas del decodificador
        
        # Capa 4: dim_z -> hidden_sizes[2]
        decoder_layers.append(nn.Linear(dim_z, hidden_sizes[2]))  # 1ra capa lineal del decodificador
        if use_batchnorm:  # Si BatchNorm está activado
            decoder_layers.append(nn.BatchNorm1d(hidden_sizes[2]))  # Agrega normalización
        decoder_layers.append(activation())  # Agrega func. de activación
        
        # Capa 5: hidden_sizes[2] -> hidden_sizes[3]
        decoder_layers.append(nn.Linear(hidden_sizes[2], hidden_sizes[3]))  # 2da capa lineal del decodificador
        if use_batchnorm:  # Si BatchNorm está activado
            decoder_layers.append(nn.BatchNorm1d(hidden_sizes[3]))  # Agrega normalización
        decoder_layers.append(activation())  # Agrega func. de activación
        
        # Capa 6: hidden_sizes[3] -> n_sensors (salida, lineal para valores reales)
        decoder_layers.append(nn.Linear(hidden_sizes[3], n_sensors))  # Capa final que genera la salida
        # Nota: Salida lineal para que pueda predecir cualquier rango de valores
        
        self.decoder = nn.Sequential(*decoder_layers)  # Une todas las capas del decodificador
        
        self._initialize_weights() # Inicialización de pesos de la red (Kaiming He para ReLU)
    
    def _initialize_weights(self):  # Método para asignar valores iniciales a pesos y sesgos usando Kaiming He para capas con ReLU
        for module in self.modules():  # Iterará sobre todos los módulos de la red
            if isinstance(module, nn.Linear):  # Si el módulo es una capa lineal
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')  # Inicializa pesos con método Kaiming
                if module.bias is not None:  # Y si hay sesgo
                    nn.init.constant_(module.bias, 0)  # Inicializa el sesgo en cero
            elif isinstance(module, nn.BatchNorm1d):  # Si el módulo no es una capa lineal, sino BatchNorm
                nn.init.constant_(module.weight, 1)  # Inicializa pesos en uno
                nn.init.constant_(module.bias, 0)  # Inicializa sesgos en cero
    
    def forward(  # Define cómo fluye la info a través de la red
        self,  # Referencia al propio objeto
        x_in: torch.Tensor,  # Datos de entrada de sensores/actuadores [batch_size, n_sensors]
        m: torch.Tensor,  # Máscara [batch_size, n_sensors] que indica qué sensores están activos (0 o 1)
        p: torch.Tensor  # Vector de sesgo paramétrico [batch_size, dim_p]
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Regresa dos tensores
        """
        Forward pass de GeMuCo.
        
        Returns:
            x_out (torch.Tensor): Pred de salida [batch_size, n_sensors]
            z (torch.Tensor): Estado latente [batch_size, dim_z]
        
        Nota: Si m[i] = 0, el valor correspondiente en x_in se enmascara a 0
        """

        # 1. Aplicar la máscara a x_in (enmascarar entradas)
        x_masked = x_in * m  # Apaga los sensores inactivos multiplicándolos elemento a elemento por cero
        
        # 2. Concatenar: [x_masked, m, p]
        # Sigue la ecuación (1) del artículo: z = h_enc(x_in, m, p)
        encoder_input = torch.cat([x_masked, m, p], dim=1) # Une datos, máscara y sesgo en un solo vector
        
        # 3. Encoder: Obtener la representación del estado latente z
        z = self.encoder(encoder_input)
        
        # 4. Decoder: Obtener predicción x_out
        x_out = self.decoder(z)  # Reconstruye o predice la salida desde el espacio latente
        
        return x_out, z  # Regresa la salida y el vector latente
    
    def encode(self, x_in: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor: # Convierte una entrada al espacio latente
        """
        Solo el encoder: obtiene z a partir de x_in, m, p. Útil para state estimation y control.
        """
        x_masked = x_in * m  # Aplica la máscara a los sensores
        encoder_input = torch.cat([x_masked, m, p], dim=1)  # Une todos los datos necesarios
        return self.encoder(encoder_input)  # Regresa la representación latente
    
    def decode(self, z: torch.Tensor) -> torch.Tensor: # Convierte un vector latente a una salida
        """
        Solo el decoder: obtiene x_out a partir de z. Útil para simulación.
        """
        return self.decoder(z) # Regresa la salida reconstruida
    
    def get_latent_size(self) -> int:  # Obtiene el tamaño del espacio latente
        return self.dim_z # Regresa la dim del espacio latente z
    
    def get_num_params(self) -> int:  # Cuenta cuántos parámetros entrenables tiene la red
        return sum(p.numel() for p in self.parameters() if p.requires_grad) # Suma todos los parámetros entrenables y regresa el no. total

# ============================================
# CLASE PARA MANEJAR MÁSCARAS
# ============================================

class MaskManager: # Clase para administrar máscaras. Mantiene el conjunto de máscaras factibles M y proporciona utilidades.
    """
    Según el artículo (sección II-A.1):
    "m ∈ {0,1}^{N_sensor} is a variable that masks the input x"
    "it is necessary to maintain a set of feasible masks M"
    """
    
    def __init__(self, n_sensors: int):  # Constructor
        self.n_sensors = n_sensors  # Guarda el no. de sensores
        self.feasible_masks = []  # Lista de máscaras válidas (como tensores)
    
    def add_mask(self, mask: torch.Tensor): # Agrega una nueva máscara al conjunto factible M
        assert mask.shape[0] == self.n_sensors, f"Mask debe tener dimensión {self.n_sensors}"  # Verifica tamaño correcto
        assert torch.all((mask == 0) | (mask == 1)), "Mask debe contener solo 0s y 1s"  # Verifica que solo tenga 0 y 1
        self.feasible_masks.append(mask.clone())  # Guarda una copia de la máscara
    
    def add_masks_from_list(self, masks_list: List[List[int]]):  # Agrega varias máscaras desde una lista de listas
        for mask in masks_list:  # Iterará sobre cada máscara
            self.add_mask(torch.tensor(mask, dtype=torch.float32))  # Convierte a tensor y la agrega
    
    def get_random_mask(self, batch_size: int = 1) -> torch.Tensor:  # Obtiene y regresa máscaras aleatorias del conjunto factible
        """
        Útil durante el entrenamiento (sección II-D).
        """
        if not self.feasible_masks:  # Si no hay máscaras registradas
            raise ValueError("No hay máscaras factibles. Agrega máscaras primero.")
        idx = torch.randint(0, len(self.feasible_masks), (batch_size,))  # Sino, selecciona índices aleatorios
        masks = torch.stack([self.feasible_masks[i] for i in idx])  # Agrupa las máscaras seleccionadas
        return masks  # Regresa las máscaras
    
    def get_all_masks(self) -> List[torch.Tensor]:  # Obtiene y regresa todas las máscaras
        return self.feasible_masks.copy()  # Devuelve una copia de la lista
    
    def mask_is_feasible(self, mask: torch.Tensor) -> bool:  # Comprueba si una máscara existe
        for feasible in self.feasible_masks:  # Iterará sobre todas las máscaras válidas
            if torch.equal(mask, feasible):  # Si son iguales...
                return True
        return False
    
    def generate_all_masks(self, exclude_zero: bool = True) -> List[torch.Tensor]:
        """
        Genera todas las combinaciones de máscaras posibles (2^n_sensors combinaciones).
        Luego se pueden filtrar para obtener M (conjunto factible).
        
        Args:
            exclude_zero: Si True, excluye la máscara de todos ceros
        """
        all_masks = []  # Almacenar resultados
        for i in range(1 if exclude_zero else 0, 2 ** self.n_sensors):  # Iterará sobre todas las combinaciones binarias
            mask = torch.tensor(  # Crea una nueva máscara
                [int(b) for b in format(i, f'0{self.n_sensors}b')],  # Convierte el núm a binario
                dtype=torch.float32  # Usa números decimales de PyTorch
            )
            all_masks.append(mask)  # Guarda la máscara
        return all_masks  # Regresa todas las máscaras
    
    def mask_to_tuple(self, mask: torch.Tensor) -> tuple:  # Convierte una máscara a tupla para usar como clave en diccionarios
        return tuple(mask.int().tolist())  # Convierte el tensor a tupla

# ============================================
# CLASE PARA MANEJAR PARAMETRIC BIAS (PB)
# ============================================

class ParametricBiasManager:  # Clase para administrar los vectores de sesgo paramétrico p
    """
    Según el artículo (sección II-D):
    "p_k is the parametric bias for the state k, which is a variable with a common value
    for the data D_k but a different value for different data."
    
    "p_k is trained with an initial value of 0."
    """
    
    def __init__(self, dim_p: int, n_states: int = 0): # Constructor
        self.dim_p = dim_p  # Guarda la dim. de cada p
        self.n_states = n_states  # Guarda el no. de estados (situaciones) diferentes
        
        # Inicializar todos los p en 0 (como dice el artículo)
        self.p_biases = nn.ParameterList()  # Lista especial de parámetros entrenables
        for _ in range(n_states):  # Repite para cada estado inicial
            self.p_biases.append(nn.Parameter(torch.zeros(dim_p)))  # Crea un p lleno de ceros
    
    def add_state(self) -> int:  # Agrega un nuevo estado (nuevo p) y retorna su índice
        self.p_biases.append(nn.Parameter(torch.zeros(self.dim_p)))  # Agrega un nuevo p
        self.n_states += 1  # Incrementa el contador de estados
        return self.n_states - 1  # Devuelve el índice del nuevo estado
    
    def get_pb(self, state_idx: int) -> torch.Tensor:  # Obtiene el p de un estado específico
        return self.p_biases[state_idx]  # Regresa el parámetro solicitado
    
    def get_all_pb(self) -> nn.ParameterList:  # Obtiene todos los p
        return self.p_biases  # Regresa la lista completa
    
    def update_pb(self, state_idx: int, new_pb: torch.Tensor):  # Actualiza un p de un estado existente (para online update)
        with torch.no_grad():  # Evita que esta operación afecte el entrenamiento
            self.p_biases[state_idx].copy_(new_pb)  # Copia los nuevos valores
    
    def get_pb_matrix(self) -> torch.Tensor:  # Convierte todos los p y lo regresa en una matriz (filas = estados, columnas = dim_p)
        return torch.stack([pb for pb in self.p_biases])  # Une todos los p
    
    def visualize_pb_2d(self): # Muestra una visualización de los p en 2D con PCA (como en las Figuras 6 y 10 del artículo)
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        if self.n_states == 0:  # Si no existen estados
            print("No hay PB para visualizar")
            return  # Sale del método
        
        pb_matrix = self.get_pb_matrix().detach().cpu().numpy() # Convierte la matriz a formato compatible con gráficos
        
        if self.dim_p > 2:  # Si tiene más de 2 dim
            pca = PCA(n_components=2)  # Reduce a 2 dim
            pb_2d = pca.fit_transform(pb_matrix)  # Realiza la reducción
        else:  # Si ya tiene 2 o menos dim
            pb_2d = pb_matrix  # Usa los datos originales
        
        plt.figure(figsize=(8, 6))  # Crea una figura
        plt.scatter(pb_2d[:, 0], pb_2d[:, 1], c=range(self.n_states), cmap='viridis')  # Traza los puntos
        for i in range(self.n_states):  # Iterará sobre cada estado
            plt.annotate(str(i), (pb_2d[i, 0], pb_2d[i, 1]))  # Escribe la etiqueta del estado
        plt.xlabel('PC1')  # Nombre del eje horizontal
        plt.ylabel('PC2')  # Nombre del eje vertical
        plt.title('Parametric Bias Visualization (PCA)')
        plt.colorbar(label='State Index')  # Agrega barra de colores
        plt.show()  # Imprime la gráfica

# ============================================
# EJEMPLO DE USO RÁPIDO (para probar que funciona)   CREO QUE ESTO ES UNA SIMULACIÓN!!!!!!!!!!!!!! REVISAR!!!!!!!!!!!!!!!!!!!!!!!!
# ============================================

if __name__ == "__main__":  # Se ejecuta solo si este archivo se corre directamente
    print("=" * 60)
    print("Testing GeMuCoNetwork - Fase 1")
    print("=" * 60)
    
    # Parámetros de ejemplo (Simulando el experimento PR2 con plumero):
    n_sensors = 2  # No. de sonsores. Ej: [ángulo_joint, posición_tool_tip]
    dim_z = 16  # Tamaño del espacio latente z
    dim_p = 2  # Tamaño del vector p
    
    # Se crea la red
    model = GeMuCoNetwork(  # Se crea una instancia del modelo
        n_sensors=n_sensors,  # Usa el no. de sensores definido
        dim_z=dim_z,  # Usa el tamaño latente definido
        dim_p=dim_p,  # Usa el tamaño p definido
        hidden_sizes=[128, 64, 64, 128],  # Configuración de capas ocultas
        use_batchnorm=True  # Activa BatchNorm
    )
    
    print(f"Modelo creado correctamente")
    print(f"  - n_sensors: {n_sensors}")  # Muestra el no. de sensores
    print(f"  - dim_z: {dim_z}")  # Muestra el tamaño latente
    print(f"  - dim_p: {dim_p}")  # Muestra el tamaño p
    print(f"  - Parámetros totales: {model.get_num_params():,}")  # Muestra cuántos parámetros tiene la red
    
    # Probar forward pass:
    batch_size = 8  # No. de ejemplos procesados al mismo tiempo
    x_in = torch.randn(batch_size, n_sensors)  # Genera datos aleatorios de entrada
    m = torch.ones(batch_size, n_sensors)  # Crea una máscara donde todos los sensores están activos
    p = torch.randn(batch_size, dim_p)  # Genera p aleatorios
    
    x_out, z = model(x_in, m, p)  # Ejecuta el modelo
    
    print(f"\nForward pass exitoso:")
    print(f"  - x_in shape: {x_in.shape}")  # Muestra dim de entrada
    print(f"  - m shape: {m.shape}")  # Muestra dim de la máscara
    print(f"  - p shape: {p.shape}")  # Muestra dim de p
    print(f"  - z shape: {z.shape} (latente)")  # Muestra dim del vector latente
    print(f"  - x_out shape: {x_out.shape} (predicción)")  # Muestra dim de salida
    
    # Probar MaskManager:
    mask_mgr = MaskManager(n_sensors=n_sensors)  # Crea el administrador de máscaras
    mask_mgr.add_masks_from_list([  # Agrega varias máscaras
        [1, 1],  # ambos sensores disponibles
        [1, 0],  # solo 1er sensor activo (0)
        [0, 1],  # solo 2do sensor activo (1)
    ])
    
    print(f"\nMaskManager:")
    print(f"  - Máscaras factibles: {len(mask_mgr.feasible_masks)}")  # Muestra cuántas máscaras existen
    random_mask = mask_mgr.get_random_mask(batch_size=2)  # Obtiene 2 máscaras aleatorias
    print(f"  - Máscara aleatoria (batch=2): {random_mask.tolist()}")  # Muestra las máscaras obtenidas
    
    # Probar ParametricBiasManager:
    pb_mgr = ParametricBiasManager(dim_p=dim_p, n_states=3)  # Crea administrador p con tres estados
    print(f"\nParametricBiasManager:")
    print(f"  - dim_p: {dim_p}")  # Muestra dim p
    print(f"  - n_states: {pb_mgr.n_states}")  # Muestra no. de estados
    print(f"  - PB matriz shape: {pb_mgr.get_pb_matrix().shape}")  # Muestra tamaño de la matriz p
    
    print("\nFase 1 completada con éxito!")