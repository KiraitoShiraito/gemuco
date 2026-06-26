"""04
GeMuCo: Generalized Multisensory Correlational Model    Fase 4: Structure Determination - Determinación automática de la estructura de la red

Basado en la sección II-E del artículo: "We describe a method for automatically determining the network structure of GeMuCo. Specifically,
we determine x_in, x_out, and a set of feasible masks M."

Pasos (según el artículo y Figura 4):
    1. Determinar x_out: Sensores que pueden ser inferidos de otros (L_i < C_thre_out)
    2. Determinar x_in y M: Combinaciones de sensores que permiten inferir x_out (L_m < C_thre_in)

El artículo también menciona (página 5): "Note that the number of layers and units of the network are given externally by humans, and these
are not automatically determined (there are various mechanisms such as NAS for these [12])."
"""

import torch  # Trabajar con tensores y redes neuronales
import torch.nn as nn  # Capas y componentes de redes neuronales
import torch.optim as optim  # Algoritmos para entrenar redes neuronales
from torch.utils.data import DataLoader, TensorDataset  # Organizar datos en lotes de entrenamiento
import numpy as np  # Cálculos matemáticos y arreglos numéricos
from typing import List, Tuple, Set, Dict, Optional  # Tipos de datos usados para documentación
from itertools import combinations  # Generar combinaciones posibles de elementos
import matplotlib.pyplot as plt
from collections import defaultdict  # Diccionario para crear valores por defecto automáticamente

# Importar de fases anteriores
try:
    from model import GeMuCoNetwork, MaskManager  # Importa la red GeMuCo y el gestor de máscaras
    from data_collector import GeMuCoDataCollector, PR2DusterSimulator  # Importa el simulador y recolector de datos
except ImportError:  # Si no encuentra esos módulos...
    print("Importando módulos locales...")

class StructureDeterminator:  # Descubre automáticamente qué sensores deben ser entradas y salidas, determinador de estructura
    """
    Según el artículo (sección II-E):
        "This operation mainly consists of determining network outputs that can be inferred
        from the latent space, and determining combinations of network inputs and masks
        that can infer the latent space."
    """
    
    def __init__(  # Constructor
        self,  # Referencia al propio objeto
        n_sensors: int,  # No. total de sensores/actuadores disponibles
        dim_z: int = 16,  # Tamaño del espacio latente z usado para la red temporal
        hidden_sizes: List[int] = [64, 32, 32, 64],  # Tamaños de las capas ocultas de las redes auxiliares
        device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Usa GPU si está disponible, sino CPU
    ):
        self.n_sensors = n_sensors  # Guarda la cantidad total de sensores
        self.dim_z = dim_z  # El tamaño del espacio latente z
        self.hidden_sizes = hidden_sizes  # La configuración de capas ocultas
        self.device = device  # El dispositivo
        
        # Para almacenar resultados
        self.x_out_indices = None  # Los índices de sensores que son salidas
        self.x_in_indices = None  # Los índices de sensores que son entradas
        self.feasible_masks = []  # Lista de máscaras M consideradas válidas
        
        # Almacenar errores calculados
        self.L_i_errors = {}  # Diccionario para guardar los errores L_i de cada sensor para determinar x_out
        self.L_m_errors = {}  # Diccionario para guardar los errores L_m de cada máscara para determinar x_in y M
    
    def compute_L_i(  # Calcula el error L_i de inferencia para cada sensor i
        self,  # Referencia al objeto
        data_per_sensor: List[torch.Tensor],  # Lista de tensores [n_samples, 1] para cada sensor, datos separados por sensor
        n_epochs: int = 50,  # No. de épocas de entrenamiento para cada predictor (red)
        batch_size: int = 32,  # Cantidad de muestras procesadas simultáneamente
        learning_rate: float = 0.001,  # Tasa de aprendizaje
        verbose: bool = True
    ) -> Dict[int, float]:  # Regresa un diccionario con errores por sensor    {sensor_idx: L_i_error}
        """
        Según el artículo (sección II-E.2 y Figura 4b):
            "A value x_i is inferred from other values x_j and its inference error is L_i.
            We collect only x_i for which L_i < C_thre_out, and construct x_out using them."
        
        Para cada sensor i:
            - Entrenar una red para predecir x_i a partir de los otros sensores (x_j, j≠i)
            - El error de predicción (MSE) es L_i
            - Si L_i es pequeño, significa que x_i está correlacionado con otros sensores y debe ser parte de x_out
        """
        print("\n" + "=" * 60)
        print("Paso 1: Determinando x_out (sensores de salida)")
        print("=" * 60)
        print(f"Calculando L_i para {self.n_sensors} sensores...")
        
        # Preparar datos combinados (todos los sensores). Asumimos que data_per_sensor[i] tiene forma [n_samples, 1]
        n_samples = data_per_sensor[0].shape[0]  # Obtiene la cantidad total de muestras
        X_all = torch.cat(data_per_sensor, dim=1)  # Une todos los sensores en una sola matriz [n_samples, n_sensors]
        
        for target_idx in range(self.n_sensors):  # Iterará sobre cada sensor para analizarlo individualmente
            if verbose:
                print(f"\n  Sensor {target_idx}: entrenando predictor...")
            
            # Datos de entrada
            input_indices = [i for i in range(self.n_sensors) if i != target_idx]  # Selecciona todos los sensores excepto el actual
            X_input = X_all[:, input_indices]  # Usa los demás sensores como entradas
            Y_target = X_all[:, target_idx:target_idx+1]  # Usa el sensor actual como objetivo a predecir
            
            predictor = SimplePredictor(  # Crea una pequeña red para realizar la predicción
                input_dim=self.n_sensors - 1,  # No. de entradas igual al total de sensores menos uno
                hidden_sizes=[64, 32],  # Dos capas ocultas
                output_dim=1  # Produce una sola salida porque predice un único sensor
            ).to(self.device)  # Mueve la red al dispositivo
            
            optimizer = optim.Adam(predictor.parameters(), lr=learning_rate) # Crea optimizador Adam
            criterion = nn.MSELoss() # Utiliza Error Cuadrático Medio como medida de error
            
            dataset = TensorDataset(X_input, Y_target) # Crea un conjunto de datos de entrada y salida
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Crea un cargador de datos
            
            predictor.train()  # Coloca la red en modo entrenamiento
            for epoch in range(n_epochs):  # Repite el entrenamiento durante varias épocas
                epoch_loss = 0.0  # Acumulador del error de la época
                for batch_x, batch_y in dataloader:  # Recorre todos los lotes de datos
                    batch_x = batch_x.to(self.device)  # Mueve las entradas al dispositivo
                    batch_y = batch_y.to(self.device)  # Mueve las salidas al dispositivo
                    
                    optimizer.zero_grad()  # Borra gradientes anteriores
                    pred = predictor(batch_x)  # Realiza una predicción
                    loss = criterion(pred, batch_y)  # Calcula el error de la predicción
                    loss.backward()  # Calcula cómo corregir los parámetros
                    optimizer.step()  # Actualiza los parámetros de la red
                    
                    epoch_loss += loss.item()  # Acumula el error del lote
                
                if verbose and epoch % 20 == 0 and epoch > 0: # Cada 20 épocas muestra progreso
                    print(f"    Epoch {epoch}: loss = {epoch_loss/len(dataloader):.6f}") # Muestra el error promedio de la época
            
            # Evaluar error final L_i
            predictor.eval()  # Cambia la red a modo evaluación
            total_error = 0.0  # Acumulador del error total
            with torch.no_grad():  # Desactiva el cálculo de gradientes porque ya no se entrenará
                for batch_x, batch_y in dataloader:  # Recorre nuevamente todos los datos
                    batch_x = batch_x.to(self.device)  # Mueve las entradas al dispositivo
                    batch_y = batch_y.to(self.device)  # Mueve las salidas al dispositivo
                    pred = predictor(batch_x)  # Genera predicciones
                    total_error += criterion(pred, batch_y).item() * batch_x.shape[0] # Acumula el error total
            
            L_i = total_error / n_samples  # Calcula el error promedio del sensor actual
            self.L_i_errors[target_idx] = L_i  # Guarda el error asociado a este sensor
            
            if verbose:
                print(f"    L_{target_idx} = {L_i:.6f}") # Muestra el valor final del error L_i
        
        return self.L_i_errors # Regresa todos los errores calculados
    
    def determine_x_out(self, threshold: float = 0.15) -> List[int]: # Decide qué sensores serán salidas (x_out) por L_i < threshold
        """
        Según el artículo (sección II-E.2):
            "We collect only x_i for which L_i < C_thre_out, and construct x_out using them."
            "Sensor values not adopted here are not utilized as part of the network output."
        
        Los autores usan C_thre_out = 0.15 en los experimentos (página 7).
        Args: threshold: Umbral C_thre_out (default 0.15)
        
        Regresa: Lista de índices de sensores que forman x_out
        """
        print(f"\nDeterminando x_out con umbral C_thre_out = {threshold}") # Muestra el valor del umbral usado
        
        self.x_out_indices = [ # Crea una lista con los sensores seleccionados como salida
            idx for idx, error in self.L_i_errors.items() # Recorre cada sensor y su error L_i
            if error < threshold # Conserva únicamente los sensores cuyo error es menor al umbral
        ]
        
        # También considerar que si un sensor tiene error muy pequeño, debe ser salida. El artículo sugiere que si no es deducible, no debe ser salida
        
        print(f"  Sensores en x_out: {self.x_out_indices}") # Muestra los sensores seleccionados
        for idx in self.x_out_indices: # Recorre cada sensor seleccionado
            print(f"    - Sensor {idx}: L_i = {self.L_i_errors[idx]:.6f} < {threshold}") # Muestra su error y confirma que cumple el criterio
        
        if not self.x_out_indices: # Si ningún sensor cumple el umbral
            print("  ADVERTENCIA: No hay sensores que cumplan el umbral.")
            print("  Usando todos los sensores como salida por defecto.")
            self.x_out_indices = list(range(self.n_sensors)) # Usa todos los sensores como salida para evitar quedarse sin resultados
        
        return self.x_out_indices # Regresa la lista final de sensores de salida
    
    def compute_L_m( # Calcula el error L_m (error de inferencia) para cada máscara posible
        self,
        data_per_sensor: List[torch.Tensor], # Datos organizados por sensor. Lista de tensores [n_samples, 1] por sensor
        x_out_indices: List[int], # Índices de sensores que fueron elegidos como salida
        n_epochs: int = 50, # Número de épocas de entrenamiento por red
        batch_size: int = 32, # Tamaño de lote usado durante entrenamiento
        learning_rate: float = 0.001, # Tasa de aprendizaje
        max_masks_to_test: Optional[int] = None, # Límite máximo opcional de máscaras a evaluar (None es probar todas)
        verbose: bool = True # Mostrar progreso
    ) -> Dict[tuple, float]: # Regresa un diccionario máscara → error   {mask_tuple: L_m_error}
        """
        Según el artículo (sección II-E.3 y Figura 4c):
            "We calculate the inference error L_m of x_out for all m."
            "We collect m and the corresponding x_i for which L_m < C_thre_in, and denote their union set as M and x_in, respectively."
        
        Para cada máscara m (combinación de sensores de entrada):
            - Entrenar una red GeMuCo temporal que use solo esos sensores como entrada
            - Predecir x_out (sensores determinados en paso anterior)
            - El error de predicción es L_m
            - Si L_m es pequeño, esa máscara es factible
        """
        print("\n" + "=" * 60)
        print("Paso 2: Determinando x_in y máscaras factibles M")
        print("=" * 60)
        
        # Preparar datos: X_all y Y_out
        n_samples = data_per_sensor[0].shape[0] # Obtiene el no. total de muestras
        X_all = torch.cat(data_per_sensor, dim=1) # Une todos los sensores en una sola matriz [n_samples, n_sensors]
        
        # Y_out son solo los sensores en x_out_indices
        Y_out = X_all[:, x_out_indices] # Extrae únicamente los sensores elegidos como salida [n_samples, len(x_out_indices)]
        
        # El artículo dice: "all 2^{N_sensor} - 1 combinations excluding masks that are all zero"
        all_masks = self._generate_all_masks(exclude_zero=True) # Genera todas las máscaras posibles excepto la vacía
        
        if max_masks_to_test and len(all_masks) > max_masks_to_test:  # Si hay demasiadas máscaras y se definió un límite
            print(f"  Limitando a {max_masks_to_test} máscaras de {len(all_masks)} totales")  # Informa la reducción
            # Estrategia: Priorizar máscaras con pocos unos (más simples)
            all_masks.sort(key=lambda m: sum(m)) # Ordena las máscaras por cantidad de sensores activos
            all_masks = all_masks[:max_masks_to_test] # Conserva únicamente las primeras máscaras
        
        print(f"  Probando {len(all_masks)} máscaras...") # Muestra cuántas serán evaluadas
        
        # Para cada máscara, entrenar una red y calcular error
        for mask_idx, mask_tuple in enumerate(all_masks): # Recorre todas las máscaras posibles
            mask_tensor = torch.tensor(mask_tuple, dtype=torch.float32) # Convierte la máscara a tensor
            active_sensors = [i for i, v in enumerate(mask_tuple) if v == 1] # Obtiene los sensores activos en esta máscara
            
            if verbose and mask_idx % 50 == 0: # Cada 50 máscaras evaluadas
                print(f"  Máscara {mask_idx}/{len(all_masks)}: {mask_tuple}") # Muestra cuál se está procesando
            
            # Datos de entrada: solo sensores activos
            if not active_sensors: # Si la máscara no tiene sensores activos
                continue  # Máscara vacía no es válida
            
            X_in = X_all[:, active_sensors] # Selecciona sólo los sensores activos como entrada [n_samples, len(active_sensors)]
            
            # Crear red GeMuCo temporal (sin PB por ahora)
            temp_model = GeMuCoNetwork( # Crea una red temporal
                n_sensors=len(x_out_indices),  # No. de sensores de salida x_out
                dim_z=self.dim_z, # Tamaño espacio latente z
                dim_p=0,  # Sin sesgo para este cálculo
                hidden_sizes=self.hidden_sizes,
                use_batchnorm=True # Normalización por lotes
            ).to(self.device) # Mueve el modelo al dispositivo
            
            # Necesitamos adaptar: la entrada es X_in, la salida es Y_out
            # Pero GeMuCo espera entrada de tamaño n_sensors + mask + p
            # Aquí simplificamos: usamos un predictor directo X_in -> Y_out
            
            predictor = SimplePredictor( # Crea una red simple que intentará predecir las salidas
                input_dim=len(active_sensors), # No. de sensores activos usados como entrada
                hidden_sizes=[64, 64, 32], # Tamaños de las capas ocultas
                output_dim=len(x_out_indices) # No. de sensores que se quieren predecir
            ).to(self.device) # Mueve la red al dispositivo
            
            optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            dataset = TensorDataset(X_in, Y_out) # Empaqueta entradas y salidas en un conjunto de datos
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Divide los datos en lotes y los mezcla
            
            # Entrenar
            predictor.train() # Pone la red en modo entrenamiento
            for epoch in range(n_epochs): # Repite el entrenamiento durante varias épocas
                epoch_loss = 0.0 # Reinicia el acumulador de error de esta época
                for batch_x, batch_y in dataloader: # Recorre cada lote de datos
                    batch_x = batch_x.to(self.device) # Mueve las entradas al dispositivo
                    batch_y = batch_y.to(self.device) # Mueve las salidas reales al dispositivo
                    
                    optimizer.zero_grad() # Borra gradientes anteriores
                    pred = predictor(batch_x) # Genera una predicción usando la red
                    loss = criterion(pred, batch_y) # Calcula el error entre predicción y valor real
                    loss.backward() # Calcula cómo deben ajustarse los pesos
                    optimizer.step() # Actualiza los pesos para reducir el error
                    
                    epoch_loss += loss.item() # Acumula el error de este lote
            
            # Evaluar error L_m
            predictor.eval() # Cambia la red a modo evaluación
            total_error = 0.0 # Inicializa el acumulador de error total
            with torch.no_grad(): # Desactiva el cálculo de gradientes
                for batch_x, batch_y in dataloader: # Recorre nuevamente todos los lotes
                    batch_x = batch_x.to(self.device) # Mueve entradas al dispositivo
                    batch_y = batch_y.to(self.device) # Mueve salidas reales al dispositivo
                    pred = predictor(batch_x) # Obtiene las predicciones de la red
                    total_error += criterion(pred, batch_y).item() * batch_x.shape[0] # Acumula el error ponderado por cantidad de muestras
            
            L_m = total_error / n_samples # Calcula el error promedio para esta máscara
            self.L_m_errors[mask_tuple] = L_m # Guarda el error asociado a esta máscara
            
            if verbose and mask_idx % 50 == 0: # Cada 50 máscaras evaluadas
                print(f"    L_m = {L_m:.6f}") # Muestra el error obtenido para la máscara actual
        
        return self.L_m_errors # Regresa el diccionario con todas las máscaras y sus errores
    
    def determine_x_in_and_masks( # Decide qué sensores pueden usarse como entrada (x_in) y qué máscaras M son válidas
        self,
        threshold: float = 0.15, # Umbral C_thre_in. Error máximo permitido para considerar válida una máscara
        x_out_indices: Optional[List[int]] = None # Lista de índices de sensores en x_out que fueron elegidos como salida (si None, usar self.x_out_indices)
    ) -> Tuple[List[int], List[torch.Tensor]]: # Regresa sensores de entrada y máscaras factibles
        """
        Según el artículo (sección II-E.3):
            "We collect m and the corresponding x_i for which L_m < C_thre_in, and denote their union set as M and x_in, respectively."
        
        También importante:
            "it is obvious that x_out can be inferred by the mask m corresponding to the set of sensors X such that X_out ⊆ X."
            Por lo tanto, esas máscaras se excluyen del cálculo.
        
        Regresa:
            - x_in_indices: Lista de índices de sensores que forman x_in
            - feasible_masks: Lista de tensores de máscaras factibles
        """
        print(f"\nDeterminando x_in y máscaras factibles con umbral C_thre_in = {threshold}") # Muestra el umbral usado
        
        if x_out_indices is None: # Si no se proporcionó una lista de sensores de salida
            x_out_indices = self.x_out_indices # Usa la lista calculada anteriormente
        
        x_out_set = set(x_out_indices) # Convierte la lista en conjunto para búsquedas más rápidas
        
        # Encontrar máscaras que cumplen L_m < threshold
        feasible_mask_tuples = [] # Lista donde se guardarán las máscaras válidas
        for mask_tuple, L_m in self.L_m_errors.items(): # Iterará sobre cada máscara evaluada junto con su error
            # Excluir máscaras que ya contienen todos los x_out
            active_sensors = {i for i, v in enumerate(mask_tuple) if v == 1} # Obtiene los sensores activos en esa máscara
            
            # "if X_out ⊆ X, we exclude from calculation"
            if x_out_set.issubset(active_sensors): # Comprueba si todos los sensores de salida están visibles
                continue # Si están visibles, descarta esta máscara
            
            if L_m < threshold: # Si el error de inferencia es suficientemente pequeño
                feasible_mask_tuples.append(mask_tuple) # Guarda la máscara como válida
        
        # La unión de todos los sensores en máscaras factibles es x_in
        all_active_sensors = set() # Conjunto para reunir todos los sensores usados
        for mask_tuple in feasible_mask_tuples: # Recorre las máscaras válidas
            for i, v in enumerate(mask_tuple): # Recorre cada posición de la máscara
                if v == 1: # Si ese sensor está activo
                    all_active_sensors.add(i) # Lo añade al conjunto
        
        self.x_in_indices = sorted(list(all_active_sensors)) # Convierte el conjunto en lista ordenada
        self.feasible_masks = [ # Convierte cada máscara válida a tensor de PyTorch
            torch.tensor(mask_tuple, dtype=torch.float32)
            for mask_tuple in feasible_mask_tuples
        ]
        
        print(f"  Sensores en x_in: {self.x_in_indices}") # Muestra los sensores elegidos como entrada
        print(f"  Máscaras factibles: {len(self.feasible_masks)}") # Muestra cuántas máscaras válidas existen
        for i, mask in enumerate(self.feasible_masks[:10]):  # Muestra primeras 10
            print(f"    m{i}: {mask.int().tolist()}") # Convierte la máscara a enteros para verla claramente
        if len(self.feasible_masks) > 10: # Si hay más de 10 máscaras
            print(f"    ... y {len(self.feasible_masks) - 10} más") # Informa cuántas faltan por mostrar
        
        return self.x_in_indices, self.feasible_masks # Regresa sensores de entrada y máscaras válidas
    
    def _generate_all_masks(self, exclude_zero: bool = True) -> List[tuple]: # Genera todas las combinaciones posibles de máscaras (2^n)
        """
        Args: exclude_zero: Si True, excluye la máscara de todos ceros
        """
        all_masks = [] # Donde se almacenarán las máscaras
        start = 1 if exclude_zero else 0 # Decide si se excluye la máscara completamente vacía
        for i in range(start, 2 ** self.n_sensors): # Recorre todos los números binarios posibles
            mask = tuple( # Convierte el número en una máscara
                int((i >> j) & 1) for j in range(self.n_sensors) # Extrae cada bit individualmente
            )
            all_masks.append(mask) # Guarda la máscara generada
        return all_masks # Regresa la lista completa de máscaras
    
    def run_automatic_structure_determination( # Ejecuta todo el proceso automático de descubrimiento de estructura
        self,
        data_per_sensor: List[torch.Tensor], # Datos separados por sensor. Lista de tensores [n_samples, 1] para cada sensor
        threshold_out: float = 0.15, # C_thre_out. Umbral para elegir sensores de salida x_out
        threshold_in: float = 0.15, # C_thre_in. Umbral para elegir sensores de entrada (máscaras factibles)
        n_epochs: int = 50, # No. de épocas de entrenamiento para redes temporales
        verbose: bool = True # Si se mostrará progreso
    ) -> Dict: # Regresa un diccionario con todos los resultados
        print("\n" + "=" * 70)
        print("GeMuCo - Determinación automática de estructura")
        print("=" * 70)
        print(f"Umbrales: C_thre_out = {threshold_out}, C_thre_in = {threshold_in}") # Muestra los umbrales usados
        print(f"Número de sensores: {self.n_sensors}") # Muestra la cantidad de sensores
        
        # Paso 1: Calcular L_i y determinar x_out
        self.compute_L_i(data_per_sensor, n_epochs=n_epochs, verbose=verbose)  # Calcula errores individuales
        x_out_indices = self.determine_x_out(threshold=threshold_out)  # Determina qué sensores serán salida
        
        # Paso 2: Calcular L_m y determinar x_in y M
        self.compute_L_m( # Calcula errores para todas las máscaras
            data_per_sensor,
            x_out_indices,
            n_epochs=n_epochs,
            verbose=verbose,
            max_masks_to_test=200  # Se limita para velocidad
        )
        x_in_indices, feasible_masks = self.determine_x_in_and_masks( # Determina sensores de entrada y máscaras válidas
            threshold=threshold_in,
            x_out_indices=x_out_indices
        )
        
        # Resultados
        results = {
            'x_out_indices': x_out_indices, # Sensores elegidos como salida
            'x_in_indices': x_in_indices, # Sensores elegidos como entrada
            'feasible_masks': feasible_masks, # Máscaras factibles encontradas
            'L_i_errors': self.L_i_errors, # Errores individuales por sensor
            'L_m_errors': self.L_m_errors, # Errores por máscara
            'threshold_out': threshold_out, # Umbral usado para salida
            'threshold_in': threshold_in, # Umbral usado para entrada
            'n_sensors': self.n_sensors # No. total de sensores
        }
        
        self._print_structure_summary(results) # Muestra un resumen final
        
        return results # Regresa todos los resultados obtenidos
    
    def _print_structure_summary(self, results: Dict): # Imprime un resumen de la estructura descubierta
        print("\n" + "=" * 70)
        print("Resumen de la Estructura Determinada")
        print("=" * 70)
        print(f"""
Estructura de la red GeMuCo:

  x_out (salidas): {results['x_out_indices']}
    - Estos sensores pueden ser inferidos de otros
    - Serán la salida de la red

  x_in (entradas): {results['x_in_indices']}
    - Estos sensores son necesarios como entrada
    - Son la unión de todos los sensores en máscaras factibles

  |M| (máscaras factibles): {len(results['feasible_masks'])}
    - Combinaciones de sensores que permiten inferir x_out

Interpretación para el experimento PR2:
  - Si x_out = [3,4,5] (tool-tip) y x_in = [0,1,2] (ángulos)
    → La red aprenderá θ → x_tool (como en el artículo)
  - Si x_out incluye también ángulos
    → La red aprenderá relaciones más complejas (como en Musashi o KXR)
        """)

class SimplePredictor(nn.Module): # Define una red sencilla para hacer predicciones en la determinación automática de estructura para L_i y L_m    
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int): # Constructor
        super().__init__() # Inicializa la clase base nn.Module
        
        layers = [] # Donde se construirán las capas de la red
        prev_dim = input_dim # Guarda el tamaño de la capa actual, comenzando por la entrada
        for h in hidden_sizes: # Recorre cada tamaño definido para las capas ocultas
            layers.append(nn.Linear(prev_dim, h)) # Añade una capa totalmente conectada
            layers.append(nn.ReLU()) # Añade una func de activación ReLU
            layers.append(nn.BatchNorm1d(h)) # Añade normalización para estabilizar el entrenamiento
            prev_dim = h # Actualiza el tamaño de entrada para la siguiente capa
        
        layers.append(nn.Linear(prev_dim, output_dim)) # Añade la capa final que genera la salida
        self.net = nn.Sequential(*layers) # Une todas las capas en una sola red secuencial
    
    def forward(self, x): # Define cómo fluye la info por la red
        return self.net(x) # Pasa los datos por todas las capas y devuelve la predicción

def create_data_per_sensor_from_collector( # Separa y convierte los datos completos recolectados en una lista de tensores, una por sensor
    all_data: List[torch.Tensor], # Datos provenientes del recolector. Lista de tensores [n_samples_total, n_sensors] (puede ser múltiples estados)
    n_sensors: int # No. total de sensores
) -> List[torch.Tensor]: # Regresa una lista de tensores [n_samples_total, 1] con los datos de cada sensor por separado
    # Concatenar todos los datos si hay múltiples estados
    if isinstance(all_data, list) and len(all_data) > 0: # Verifica si los datos vienen en forma de lista y no está vacía
        if all_data[0].dim() == 2: # Comprueba si cada elemento es una matriz de dos dimensiones
            X_concatenated = torch.cat(all_data, dim=0) # Une todos los datos verticalmente en una sola matriz
        else: # Si no son matrices bidimensionales...
            X_concatenated = all_data # Usa los datos tal como llegaron
    else: # Si no es una lista válida
        X_concatenated = all_data # Usa directamente el contenido recibido
    
    # Separar por sensor
    data_per_sensor = [] # Donde se almacenarán los datos separados por sensor
    for i in range(n_sensors): # Iterará sobre todos los sensores
        data_per_sensor.append(X_concatenated[:, i:i+1]) # Extrae la columna correspondiente al sensor y la guarda
    
    return data_per_sensor # Regresa la lista de sensores separados

# ============================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================

def visualize_structure_determination( # Visualizar los resultados de la determinación de estructura
    determinator: StructureDeterminator, # Objeto que realizó la determinación de estructura
    results: Dict # Diccionario con los resultados
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # Crea una figura con dos gráficas lado a lado
    
    # Gráfico 1: Errores L_i por sensor
    ax1 = axes[0] # Selecciona la primera gráfica
    sensors = list(determinator.L_i_errors.keys()) # Obtiene la lista de sensores evaluados
    errors = [determinator.L_i_errors[i] for i in sensors] # Obtiene el error L_i correspondiente a cada sensor
    colors_out = ['red' if i in results['x_out_indices'] else 'blue' for i in sensors] # Colorea de rojo los sensores elegidos como salida
    
    ax1.bar(sensors, errors, color=colors_out, alpha=0.7) # Traza un gráfico de barras con los errores
    ax1.axhline(y=results['threshold_out'], color='green', linestyle='--', # Traza una línea horizontal indicando el umbral
                label=f'Umbral (C_thre_out = {results["threshold_out"]})')
    ax1.set_xlabel('Sensor Index') # Etiqueta del eje horizontal
    ax1.set_ylabel('L_i (Inference Error)') # Etiqueta del eje vertical
    ax1.set_title('Determinación de x_out\n(Rojo = seleccionado como salida)')
    ax1.set_yscale('log') # Escala logarítmica para visualizar mejor los errores
    ax1.legend() # Muestra la leyenda
    ax1.grid(True, alpha=0.3) # Activa una cuadrícula suave
    
    # Gráfico 2: Distribución de errores L_m
    ax2 = axes[1] # Selecciona la segunda gráfica
    if determinator.L_m_errors: # Comprueba si existen errores L_m calculados
        errors_m = list(determinator.L_m_errors.values()) # Obtiene todos los errores L_m
        ax2.hist(errors_m, bins=30, alpha=0.7, color='purple') # Traza un histograma con la distribución de errores
        ax2.axvline(x=results['threshold_in'], color='green', linestyle='--', # Traza una línea vertical indicando el umbral
                    label=f'Umbral (C_thre_in = {results["threshold_in"]})')
        ax2.set_xlabel('L_m (Inference Error)') # Etiqueta del eje horizontal
        ax2.set_ylabel('Frecuencia') # Etiqueta del eje vertical
        ax2.set_title('Distribución de errores L_m para máscaras')
        ax2.legend() # Muestra la leyenda
        ax2.grid(True, alpha=0.3) # Activa una cuadrícula suave
    
    plt.tight_layout() # Ajusta automáticamente los espacios entre elementos
    plt.show()
    
    # Imprimir matriz de máscaras factibles
    if results['feasible_masks']: # Si hay máscaras factibles
        print("\nMatriz de máscaras factibles (primeras 10):")
        mask_matrix = torch.stack(results['feasible_masks'][:10])  # Une las primeras 10 máscaras en una matriz
        print(mask_matrix.int()) # Muestra la matriz usando valores enteros
        print("(1 = sensor disponible como entrada, 0 = sensor tapado)")  # Explica el significado de los valores

# ============================================
# EJEMPLO DE USO COMPLETO
# ============================================

def run_structure_determination_example(): # Ejecuta un ejemplo completo de determinación automática de estructura, usando datos del PR2
    print("=" * 70)
    print("GeMuCo - Fase 4: Determinación Automática de Estructura")
    print("=" * 70)
    
    # 1. Configuración
    n_joints = 3 # No. de articulaciones del brazo simulado
    n_tooltip = 3 # No. de coordenadas del extremo de la herramienta (X,Y,Z)
    n_sensors = n_joints + n_tooltip # No. total de sensores disponibles
    samples_per_state = 500 # Cantidad de muestras que se generarán para cada estado
    
    # 2. Generar datos sintéticos
    print("\nGenerando datos sintéticos...")
    simulator = PR2DusterSimulator() # Crea el simulador del robot PR2 con plumero
    collector = GeMuCoDataCollector(simulator) # Crea el recolector de datos usando el simulador
    
    all_joint_angles, all_tool_tips, state_info = collector.collect_all_states( # Genera datos para todos los estados posibles
        n_samples_per_state=samples_per_state,
        n_joints=n_joints,
        random_motion=True
    )
    
    # 3. Normalizar y combinar datos
    normalized_joints, normalized_tooltips, norm_params = collector.get_normalized_data( # Normaliza los datos para facilitar el entrenamiento
        all_joint_angles, all_tool_tips
    )
    
    # Combinar joint angles y tooltips en un solo tensor por estado
    all_data_per_state = [] # Donde se guardarán los datos combinados de cada estado
    for joints, tooltips in zip(normalized_joints, normalized_tooltips): # Recorre los ángulos y posiciones normalizados
        # joints: [n_samples, 3], tooltips: [n_samples, 3]
        combined = torch.cat([joints, tooltips], dim=1) # Une ángulos y posiciones en una sola matriz [n_samples, 6]
        all_data_per_state.append(combined) # Guarda la matriz combinada
    
    X_all_states = torch.cat(all_data_per_state, dim=0) # Une todos los estados en una única matriz [total_samples, 6]
    
    # Crear data_per_sensor
    data_per_sensor = [] # Donde se almacenará la info sensor por sensor
    for i in range(n_sensors): # Recorre todos los sensores
        data_per_sensor.append(X_all_states[:, i:i+1]) # Extrae una columna y la guarda como sensor independiente
    
    print(f"\nDatos preparados:")
    print(f"  - Sensores: {n_sensors}") # Cantidad total de sensores
    print(f"  - Muestras totales: {X_all_states.shape[0]}") # Cuántas muestras existen en total
    print(f"  - Sensor 0 (shoulder angle): media={data_per_sensor[0].mean():.3f}, std={data_per_sensor[0].std():.3f}") # Estadísticas del sensor 0
    print(f"  - Sensor 3 (tooltip x): media={data_per_sensor[3].mean():.3f}, std={data_per_sensor[3].std():.3f}") # Estadísticas del sensor 3
    
    determinator = StructureDeterminator( # Crea el objeto encargado de descubrir automáticamente la estructura
        n_sensors=n_sensors,
        dim_z=16,
        hidden_sizes=[64, 32, 32, 64],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results = determinator.run_automatic_structure_determination( # Ejecuta todo el proceso de determinación automática
        data_per_sensor=data_per_sensor,
        threshold_out=0.15,
        threshold_in=0.15,
        n_epochs=30,  # Para prueba rápida
        verbose=True
    )
    
    visualize_structure_determination(determinator, results) # Visualizar los resultados
    
    print("\n" + "=" * 70)
    print("Interpretación de Resultados")
    print("=" * 70)
    
    if set(results['x_out_indices']) == {3, 4, 5}: # Comprueba si los sensores de salida encontrados son los esperados
        print("✓ RESULTADO ESPERADO: x_out = [3,4,5] (tool-tip position)") # Informa que el resultado coincide con lo esperado
        print("  Esto coincide con el artículo: la red aprende θ → x_tool")
    else: # Si el resultado no coincide
        print(f"  x_out obtenido: {results['x_out_indices']}") # Muestra los sensores encontrados
        print("  Esto podría indicar relaciones más complejas entre sensores")
    
    if set(results['x_in_indices']) == {0, 1, 2}: # Comprueba si los sensores de entrada encontrados son los esperados
        print("✓ RESULTADO ESPERADO: x_in = [0,1,2] (joint angles)")  # Informa que coincide con la expectativa
    else: # Si no coincide
        print(f"  x_in obtenido: {results['x_in_indices']}") # Muestra los sensores encontrados
    
    print(f"\nNúmero de máscaras factibles: {len(results['feasible_masks'])}")
    
    return determinator, results # Regresa el objeto determinador y los resultados obtenidos

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__": # Comprueba si este archivo se está ejecutando directamente
    determinator, results = run_structure_determination_example()  # Ejecuta el ejemplo completo y guarda los resultados
    
    print("\n" + "=" * 70)
    print("Resumen de la Fase 4")
    print("=" * 70)
    print("""
    Implementado:
    ✓ Cálculo de L_i para determinar qué sensores pueden ser inferidos (x_out)
    ✓ Cálculo de L_m para determinar máscaras factibles (M) y x_in
    ✓ Generación automática de todas las máscaras posibles
    ✓ Entrenamiento de redes temporales para calcular errores
    ✓ Visualización de resultados
    ✓ Interpretación para el experimento PR2

    Correspondencia con el artículo:
    - Sección II-E.1: Network Training with all masks
    - Sección II-E.2: Determination of Network Output (Figura 4b)
    - Sección II-E.3: Determination of Network Input (Figura 4c)
    - Umbrales: C_thre_out = 0.15, C_thre_in = 0.15 (página 7)
    
    La determinación automática permite que el robot descubra:
    - Qué sensores están correlacionados (deben ser salida)
    - Qué sensores son suficientes como entrada
    - Qué combinaciones de sensores (máscaras) son útiles
    """)