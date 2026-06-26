"""06
GeMuCo: Generalized Multisensory Correlational Model                         Fase 6: Estimación de estado

Basado en la sección II-H del artículo:
"In state estimation, the sensor values that are currently unavailable are estimated from the network. For this purpose, (a), (b), (e), and (f) in Fig. 3 are used."

Casos de state estimation según el artículo:
    a) Usar máscara factible m para inferir datos no disponibles directamente
    b) La red tiene todas las entradas necesarias
    e) Optimizar z cuando no hay máscara factible y x_out contiene el valor a estimar
    f) Optimizar x_in cuando el valor a estimar no está en x_out
"""

import torch # Trabajar con tensores y redes neuronales
import torch.nn as nn # Contiene capas y herramientas para construir redes
from typing import Optional, List, Tuple, Dict, Any, Union # Tipos de datos para documentar parámetros y retornos
import numpy as np # Op numéricas y arreglos
from collections import defaultdict # Diccionario que crea valores por defecto automáticamente

# Importar de fases anteriores
try:
    from model import GeMuCoNetwork, MaskManager, ParametricBiasManager # Importa clases principales de archivos anteriores
    from optimizer import LatentOptimizer, XInOptimizer # Importa los optimizadores de la Fase 5
except ImportError:
    print("Importando módulos locales...")


class StateEstimator: # Clase para estimar sensores faltantes usando GeMuCo
    """
    Implementa los cuatro casos descritos en el artículo (sección II-H):
    - Caso (a): Máscara factible disponible
    - Caso (b): Red con todas las entradas necesarias
    - Caso (e): Optimización de z (valores a estimar están en x_out)
    - Caso (f): Optimización de x_in (valores a estimar NO están en x_out)
    
    El artículo también menciona:
    "If x_out contains the value to be estimated, we consider the execution of (a) or (b), and if (a) and (b) are not possible, 
    we consider the execution of (e). If x_out does not contain the value to be estimated, we consider the execution of (f)."
    """
    
    def __init__( # Constructor
        self, # Referencia al propio objeto
        model: GeMuCoNetwork, # Modelo red GeMuCo entrenado
        mask_manager: MaskManager, # Gestor de máscaras válidas M
        pb_manager: ParametricBiasManager, # Gestor de sesgos paramétricos
        latent_optimizer: LatentOptimizer, # Optimizador del espacio latente z (caso e)
        xin_optimizer: XInOptimizer, # Optimizador de entradas x_in (caso f)
        device: str = "cuda" if torch.cuda.is_available() else "cpu" # Dispositivo
    ):
        self.model = model # Guarda el modelo recibido
        self.mask_manager = mask_manager # El gestor de máscaras
        self.pb_manager = pb_manager # El gestor de sesgos paramétricos
        self.latent_optimizer = latent_optimizer # El optimizador de z
        self.xin_optimizer = xin_optimizer # El optimizador de x_in
        self.device = device # Y el dispositivo
        
        self.estimation_history = [] # Donde se almacenará el historial de estimaciones
    
    def estimate_with_mask( # Caso a) y b): Estima sensores faltantes usando máscaras factibles directamente
        self, # Referencia al objeto actual
        x_available: torch.Tensor, # Tensor de valores de sensores disponibles [batch_size, n_sensors]
        mask_available: torch.Tensor, # Máscara que indica qué sensores están disponibles [batch_size, n_sensors] (1=disponible, 0=no disponible)
        state_idx: int, # Índice del estado actual del sistema (para obtener p_k)
        sensors_to_estimate: Optional[List[int]] = None # Lista de sensores a estimar (si None, estima todos los no disponibles)
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]: # Regresa Tensor con sensores estimados, máscara usada y se encontró una factible
        """
        Según el artículo (sección II-H):
        "In (a), we consider a mask m, which is set to 0 for the unavailable data and 1 for the available data. If this mask m is included in the set of
        feasible masks M, then by inputting this m and x_in_masked with 0 for the unavailable data into the network, we can estimate the currently unavailable data."
            
        "Similarly, in (b), if the network has all necessary inputs, the remaining data can be estimated directly."
        """
        batch_size = x_available.shape[0] # Obtiene cuántas muestras hay en el lote
        n_sensors = self.model.n_sensors # Y el no. total de sensores del modelo
        
        mask_available = (mask_available > 0.5).float() # Convierte la máscara a valores binarios

        #CORREGIDO: Asegurar que mask_available tenga forma [batch_size, n_sensors]
        if mask_available.dim() == 3:
            mask_available = mask_available.squeeze(0)
        if mask_available.dim() == 1:
            mask_available = mask_available.unsqueeze(0)
        
        #m_candidate = mask_available.clone() # Crea una copia de la máscara disponible: 1 para sensores disponibles, 0 para no disponibles
        m_candidate = mask_available[0].clone() if batch_size > 0 else mask_available.clone()
        
        #is_feasible = self.mask_manager.mask_is_feasible(m_candidate[0]) # Verifica si la máscara es válida según GeMuCo
        is_feasible = self.mask_manager.mask_is_feasible(m_candidate)
        
        if not is_feasible: # Si no es factible...
            #m_candidate = self._find_best_feasible_mask(mask_available[0]) # Busca la máscara válida más parecida
            m_candidate = self._find_best_feasible_mask(m_candidate)
            if m_candidate is None: # Si no se encontró ninguna máscara válida
                return x_available, torch.zeros_like(x_available), False  # Regresa los datos originales e indica fallo
            
        # 🔧 CORREGIDO: Asegurar que m_candidate tenga la forma correcta [n_sensors]
        if m_candidate.dim() > 1:
            m_candidate = m_candidate.squeeze()
        
        m_batch = m_candidate.unsqueeze(0).expand(batch_size, -1).to(self.device) # Replica la máscara para todas las muestras (batch)
        
        p_k = self.pb_manager.get_pb(state_idx) # Obtiene el vector paramétrico asociado al estado actual
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1) # Replica el vector paramétrico para todas las muestras
        
        x_in_masked = x_available * mask_available # Aplicar máscara a x_available (Conserva sólo los sensores disponibles y oculta los demás)
        
        # Forward pass
        with torch.no_grad(): # Desactiva cálculo de gradientes porque sólo se va a inferir
            x_out_pred, _ = self.model(x_in_masked, m_batch, p_batch) # Ejecuta el modelo para estimar el estado completo
        
        # Para los sensores que ya estaban disponibles, podemos mantener el valor original o usar la pred (el artículo sugiere usar la predicción para consistencia)
        #x_estimated = x_out_pred # Guarda la estimación obtenida
        x_estimated = x_out_pred.clone()
        
        if sensors_to_estimate is not None: # Si se especificaron sensores especificos a estimar...
            for i in range(batch_size): # Recorre todas las muestras
                for sensor_idx in sensors_to_estimate: # Recorre cada sensor solicitado
                    if mask_available[i, sensor_idx] == 0: # Si ese sensor estaba oculto
                        x_estimated[i, sensor_idx] = x_out_pred[i, sensor_idx] # Sustituye por el valor estimado
        
        return x_estimated, m_batch, True # Regresa la estimación, máscara utilizada y éxito
    
    def _find_best_feasible_mask(self, mask_available: torch.Tensor) -> Optional[torch.Tensor]: # Encuentea la mejor máscara válida
        """
        Prioriza máscaras con mayor número de sensores disponibles.
        """
        available_indices = set(torch.where(mask_available > 0.5)[0].tolist()) # Obtiene índices de sensores disponibles
        
        best_mask = None # Inicializa la mejor máscara encontrada
        best_score = -1 # Inicializa la mejor puntuación
        
        for mask in self.mask_manager.get_all_masks(): # Recorre todas las máscaras factibles registradas
            mask_indices = set(torch.where(mask > 0.5)[0].tolist()) # Obtiene los sensores activos en esa máscara
            
            if mask_indices.issubset(available_indices): # Verifica si la máscara usa sólo sensores disponibles
                score = len(mask_indices) # Cuenta cuántos sensores aprovecha (priorizar máscaras con más sensores)
                if score > best_score: # Si utiliza más sensores que la mejor hasta ahora...
                    best_score = score # Actualiza la mejor puntuación
                    best_mask = mask # Guarda esta máscara como la mejor encontrada
        
        return best_mask # Regresa la mejor máscara válida encontrada
    
    def estimate_by_optimizing_z( # Caso e): Estima sensores faltantes optimizando la variable latente z
        self, # Referencia al objeto actual
        x_available: torch.Tensor, # Sensores disponibles observados [batch_size, n_sensors]
        mask_available: torch.Tensor, # Máscara que indica qué sensores están disponibles [batch_size, n_sensors]
        state_idx: int, # Índice del estado actual del sistema
        sensors_to_estimate: List[int], # Lista de sensores a estimar (deben estar en x_out)
        n_iterations: int = 50, # Número de iteraciones de optimización
        learning_rate: float = 0.01, # Tasa de aprendizaje usada en la optimización
        verbose: bool = False # Si mostrar progreso
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]: # Regresa estimación completa [batch_size, n_sensors], z optimizada e historial de pérdidas
        """
        Según el artículo (sección II-H):
        "If there is no feasible m in the form of (a) and (b), state estimation is performed in the form of (e). This corresponds to the case where the
        loss function is set as follows in Section II-G: h_loss(x_out_pred, x_out_data) = || m_x_out ⊙ (x_out_pred - x_out_data) ||_2"
        
        Este caso se usa cuando:
        - Los valores a estimar están en x_out
        - No hay una máscara factible directa
        """
        batch_size = x_available.shape[0] # Obtiene el número de muestras del lote
        n_sensors = self.model.n_sensors # Y el número total de sensores del modelo
        
        p_k = self.pb_manager.get_pb(state_idx) # Obtiene el sesgo paramétrico correspondiente al estado
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1) # Replica ese sesgo para todas las muestras del lote
        
        mask_x_out = mask_available.clone() # Crea una copia de la máscara de sensores disponibles para x_out (solo considera los disponibles)
        
        # Para cada muestra en el batch, optimizar z
        x_estimated_list = [] # Donde se almacenarán las estimaciones
        z_opt_list = [] # Las z optimizadas
        all_loss_history = [] # Y los historiales de pérdida
        
        for i in range(batch_size): # Recorre cada muestra del lote
            x_avail_i = x_available[i:i+1] # Extrae una sola muestra de sensores disponibles
            mask_i = mask_x_out[i:i+1] # Extrae la máscara correspondiente a esa muestra
            
            if verbose:
                print(f"\nOptimizando z para muestra {i}") # Muestra que se está optimizando
            
            # Optimiza z para que la predicción coincida con valores disponibles (reconstruye los sensores faltantes)
            z_opt, x_out_pred, loss_history = self.latent_optimizer.optimize_z_from_measurements(
                x_out_measured=x_avail_i, # Valores observados de sensores
                mask_x_out=mask_i, # Máscara de sensores visibles
                z_init=None, # Comienza desde una z inicial automática
                n_iterations=n_iterations, # Número de iteraciones de optimización
                learning_rate=learning_rate, # Tasa de aprendizaje
                return_history=True # Solicita guardar el historial de pérdidas
            )
            
            # La predicción completa es la estimación
            x_estimated_list.append(x_out_pred) # Guarda la estimación obtenida
            z_opt_list.append(z_opt) # La z encontrada
            all_loss_history.append(loss_history) # Y la evolución de la pérdida
        
        x_estimated = torch.cat(x_estimated_list, dim=0) # Une todas las estimaciones en un solo tensor
        z_opt = torch.cat(z_opt_list, dim=0) # Une todas las variables latentes optimizadas
        
        return x_estimated, z_opt, all_loss_history # Regresa los resultados completos
    
    def estimate_by_optimizing_x_in( # Caso f): Estima sensores faltantes optimizando directamente x_in
        self, # Referencia al objeto actual
        x_available: torch.Tensor, # Sensores disponibles observados [batch_size, n_sensors]
        mask_available: torch.Tensor, # Máscara de sensores disponibles [batch_size, n_sensors]
        state_idx: int, # Índice del estado actual del sistema
        sensors_to_estimate: List[int], # Lista de sensores que se desean estimar (reconstruir) (no en x_out)
        n_iterations: int = 50, # Número de iteraciones de optimización
        learning_rate: float = 0.01, # Tasa de aprendizaje
        verbose: bool = False # Progreso
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]: # Regresa estimación [batch_size, n_sensors], x_in optimizada e historial de pérdidas
        """
        Según el artículo (sección II-H):
        "If x_out does not contain the value to be estimated, state estimation is performed in the form (f). This corresponds to the case in Section II-G
        where the variable to be optimized z_opt and its initial value z_init are changed to x_in_opt and x_in_init, respectively."
        """
        batch_size = x_available.shape[0] # Obtiene cuántas muestras hay en el lote
        
        p_k = self.pb_manager.get_pb(state_idx) # Obtiene el sesgo paramétrico asociado al estado
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1) # Replica el sesgo para todas las muestras
        
        # Crea máscara para x_out (solo sensores disponibles en x_out)
        # Nota: x_out puede ser un subconjunto de n_sensors
        mask_x_out = mask_available.clone() # Crea una copia de la máscara de sensores visibles
        
        # Para la optimización de x_in, se necesita una máscara m que determine qué sensores de entrada se optimizan
        # Usa una máscara que permite usar cualquier sensor como entrada
        m = torch.ones(1, self.model.n_sensors, device=self.device) # Crea una máscara donde todos los sensores están habilitados
        
        x_estimated_list = [] # Almacenar estimaciones
        x_in_opt_list = [] # x_in optimizadas
        all_loss_history = [] # E historiales de pérdida
        
        for i in range(batch_size): # Recorre cada muestra del lote
            x_avail_i = x_available[i:i+1] # Extrae una muestra individual
            mask_i = mask_x_out[i:i+1] # Y la máscara de esa muestra
            
            if verbose:
                print(f"\nOptimizando x_in para muestra {i}")
            
            # Optimizar x_in
            x_in_opt, x_out_pred, loss_history = self.xin_optimizer.optimize_x_in_from_measurements(  # Optimiza directamente los sensores de entrada
                x_out_measured=x_avail_i, # Valores observados
                m=m, # Máscara utilizada durante la optimización
                p=p_batch[i:i+1], # Sesgo paramétrico correspondiente
                mask_x_out=mask_i, # Máscara de sensores visibles
                x_in_init=None, # Usa inicialización automática
                n_iterations=n_iterations, # Número de iteraciones
                learning_rate=learning_rate, # Tasa de aprendizaje
                return_history=True # Solicita historial de pérdidas
            )
            
            # La salida predicha contiene la estimación
            x_estimated_list.append(x_out_pred) # Guarda la estimación obtenida
            x_in_opt_list.append(x_in_opt) # La entrada optimizada
            all_loss_history.append(loss_history) # Y el historial de pérdidas
        
        x_estimated = torch.cat(x_estimated_list, dim=0) # Une todas las estimaciones en un único tensor
        x_in_opt = torch.cat(x_in_opt_list, dim=0) # Une todas las entradas optimizadas
        
        return x_estimated, x_in_opt, all_loss_history # Regresa los resultados finales
    
    def estimate_state( # Func principal, decide cómo estimar sensores faltantes. Decide auto qué caso usar
        self, # Referencia al objeto actual
        x_available: torch.Tensor, # Valores de sensores que sí están disponibles [batch_size, n_sensors]
        mask_available: torch.Tensor, # Máscara que indica qué sensores están disponibles [batch_size, n_sensors]
        state_idx: int, # Índice del estado actual del sistema
        sensors_to_estimate: Optional[List[int]] = None, # Lista de sensores a estimar (si None, estima los no disponibles)
        method: str = "auto", # Método a usar ("auto", "mask", "optimize_z" u "optimize_x_in")
        **kwargs # Parámetros adicionales opcionales
    ) -> Dict[str, Any]: # Regresa diccionario con: Estimación completa, método usado, si fue éxitosa e info adicional (mask usada, loss hist, ...)
        batch_size = x_available.shape[0] # Obtiene cuántas muestras hay en el lote
        n_sensors = self.model.n_sensors # Y el número total de sensores
        
        # Determinar qué sensores estimar
        if sensors_to_estimate is None: # Si el usuario no indicó qué sensores estimar...
            sensors_to_estimate = [] # Crea una lista vacía
            for i in range(batch_size): # Recorre todas las muestras
                unavailable = torch.where(mask_available[i] < 0.5)[0].tolist() # Busca sensores no disponibles
                sensors_to_estimate.extend(unavailable) # Agrega esos sensores a la lista
            sensors_to_estimate = list(set(sensors_to_estimate)) # Elimina posibles sensores repetidos
        
        # Verificar si los sensores a estimar están en x_out, se necesitaría saber x_out_indices del modelo
        # Por ahora, asumir que todos los sensores pueden estar en x_out
        sensors_in_x_out = list(range(n_sensors)) # Crea una lista con todos los sensores del sistema
        
        result = { # Crea el diccionario donde se guardará el resultado final
            'x_estimated': None, # Se guardará la estimación obtenida
            'method_used': None, # El método usado
            'success': False, # Inicialmente se asume que no hubo éxito
            'metadata': {} # Espacio para info adicional
        }
        
        if method == "auto": # Si se eligió selección automática de método...
            x_est, m_used, success = self.estimate_with_mask(
                x_available, mask_available, state_idx, sensors_to_estimate, **kwargs # Intenta primero estimar usando máscaras factibles
            )
            
            if success: # Si la estimación mediante máscara funcionó...
                result['x_estimated'] = x_est # Guarda la estimación
                result['method_used'] = 'mask' # Registra que se usó el método de máscara
                result['success'] = True # Marca el proceso como exitoso
                result['metadata']['mask_used'] = m_used # Guarda la máscara utilizada
                return result # Regresa inmediatamente el resultado
            
            # Si no hay máscara...
            if all(s in sensors_in_x_out for s in sensors_to_estimate): # Verifica si todos los sensores están en x_out
                # Caso (e): Optimizar z
                x_est, z_opt, loss_history = self.estimate_by_optimizing_z(
                    x_available, mask_available, state_idx, sensors_to_estimate, **kwargs # Realiza estimación optimizando z
                )
                result['x_estimated'] = x_est # Guarda la estimación
                result['method_used'] = 'optimize_z' # Registra el método usado
                result['success'] = True # Marca éxito
                result['metadata']['z_opt'] = z_opt # Guarda la variable latente optimizada
                result['metadata']['loss_history'] = loss_history # Guarda el historial de pérdidas
            else: # Si hay sensores que requieren otro tipo de estimación...
                # Caso (f): Optimizar x_in
                x_est, x_in_opt, loss_history = self.estimate_by_optimizing_x_in(
                    x_available, mask_available, state_idx, sensors_to_estimate, **kwargs # Optimiza x_in
                )
                result['x_estimated'] = x_est # Guarda la estimación
                result['method_used'] = 'optimize_x_in' # Registra el método usado
                result['success'] = True # Marca éxito
                result['metadata']['x_in_opt'] = x_in_opt # Guarda la entrada optimizada
                result['metadata']['loss_history'] = loss_history # Guarda el historial de pérdidas
        
        elif method == "mask": # Si el usuario obliga a usar máscaras...
            x_est, m_used, success = self.estimate_with_mask( # Ejecuta estimación por máscara
                x_available, mask_available, state_idx, sensors_to_estimate, **kwargs
            )
            result['x_estimated'] = x_est # Guarda la estimación
            result['method_used'] = 'mask' # Registra el método usado
            result['success'] = success # Guarda si la estimación tuvo éxito
            if success: # Si el proceso fue exitoso
                result['metadata']['mask_used'] = m_used # Guarda la máscara usada
        
        elif method == "optimize_z": # Si se solicita explícitamente optimizar z
            x_est, z_opt, loss_history = self.estimate_by_optimizing_z( # Ejecuta optimización de z
                x_available, mask_available, state_idx, sensors_to_estimate, **kwargs
            )
            result['x_estimated'] = x_est # Guarda la estimación
            result['method_used'] = 'optimize_z' # Registra el método usado
            result['success'] = True # Marca éxito
            result['metadata']['z_opt'] = z_opt # Guarda la z optimizada
            result['metadata']['loss_history'] = loss_history # Guarda el historial de pérdidas
        
        elif method == "optimize_x_in": # Si se solicita explícitamente optimizar x_in
            x_est, x_in_opt, loss_history = self.estimate_by_optimizing_x_in( # Ejecuta optimización de x_in
                x_available, mask_available, state_idx, sensors_to_estimate, **kwargs
            )
            result['x_estimated'] = x_est # Guarda la estimación
            result['method_used'] = 'optimize_x_in' # Registra el método usado
            result['success'] = True # Marca éxito
            result['metadata']['x_in_opt'] = x_in_opt # Guarda la entrada optimizada
            result['metadata']['loss_history'] = loss_history # Guarda el historial de pérdidas
        
        # Guardar en historial
        self.estimation_history.append({ # Agrega info al historial de estimaciones
            'timestamp': len(self.estimation_history), # Usa el tamaño actual del historial como identificador temporal
            'method': result['method_used'], # Guarda qué método fue usado
            'success': result['success'], # Guarda si la estimación tuvo éxito
            'sensors_estimated': sensors_to_estimate # Guarda qué sensores fueron estimados
        })
        
        return result # Regresa el resultado completo
    
    def estimate_single_sensor( # Estima sólo un sensor específico
        self, # Referencia al objeto actual
        x_available: torch.Tensor, # Valores de sensores disponibles
        mask_available: torch.Tensor, # Máscara que indica qué sensores están disponibles
        state_idx: int, # Índice del estado actual del sistema
        sensor_idx: int, # Índice del sensor a estimar
        **kwargs # Parámetros (para estimate_state) adicionales opcionales
    ) -> Tuple[torch.Tensor, float]: # Regresa valor estimado para ese sensor y nivel de confianza de la estimación (inv de la pérdida)
        result = self.estimate_state( # Llama al estimador general de estados
            x_available, mask_available, state_idx, # Pasa los datos disponibles, la máscara de disponibilidad y el estado actual
            sensors_to_estimate=[sensor_idx], # Indica que sólo se estime este sensor
            **kwargs # Reenvía parámetros adicionales
        )
        
        if result['success'] and result['x_estimated'] is not None: # Si la estimación fue exitosa y existe resultado...
            estimated_value = result['x_estimated'][0, sensor_idx] # Obtiene el valor estimado del sensor solicitado
            
            # Calcular confianza basada en la pérdida final
            loss_history = result['metadata'].get('loss_history', [[]]) # Obtiene el historial de errores de optimización
            if loss_history and loss_history[0]: # Si existe historial válido
                final_loss = loss_history[0][-1] if isinstance(loss_history[0], list) else loss_history[-1] # Obtiene último error registrado
                confidence = 1.0 / (1.0 + final_loss) # Convierte el error en una medida de confianza, entre 0 y 1
            else: # Si no existe historial de pérdidas...
                confidence = 0.5 # Asigna confianza intermedia por defecto
            
            return estimated_value, confidence # Regresa valor estimado y confianza
        
        return torch.tensor(0.0), 0.0 # Si falló la estimación devuelve cero y confianza nula


class MultiModalStateEstimator: # Estimador de estado multimodal que combina múltiples fuentes de información.
    """
    Esto es útil cuando se tienen múltiples sensores redundantes o de diferentes modalidades.
    El artículo menciona esto en el contexto de "Multisensory Correlation".
    """
    
    def __init__(self, estimators: Dict[str, StateEstimator]): # Constructor. estimators: Diccionario {modalidad: StateEstimator}
        self.estimators = estimators # Guarda el diccionario de estimadores disponibles
    
    def estimate_fusion( # Fusiona estimaciones de múltiples modalidades
        self, # Referencia al objeto actual
        measurements: Dict[str, Tuple[torch.Tensor, torch.Tensor]], # Mediciones agrupadas por modalidad
        state_idx: int, # índice del estado actual del sistema
        fusion_method: str = "weighted_average" # Método de fusión a utilizar: "weighted_average", "max_confidence", "kalman"
    ) -> Dict[str, Any]: # Regresa estimación fusionada
        results = {} # Diccionario para almacenar resultados individuales
        confidences = {} # Diccionario para almacenar niveles de confianza
        
        # Obtener estimaciones de cada modalidad
        for modality, (x_avail, mask_avail) in measurements.items(): # Recorre todas las modalidades disponibles
            estimator = self.estimators.get(modality) # Busca el estimador asociado a esa modalidad
            if estimator is None: # Si no existe estimador para esa modalidad...
                continue # La ignora y pasa a la siguiente
            
            result = estimator.estimate_state(x_avail, mask_avail, state_idx) # Realiza estimación usando esa modalidad
            results[modality] = result # Lo guarda
            
            # Calcular confianza
            if result['success']: # Si la estimación tuvo éxito...
                loss_history = result['metadata'].get('loss_history', [[]]) # Obtiene historial de errores
                if loss_history and loss_history[0]: # Si existe historial válido...
                    final_loss = loss_history[0][-1] if isinstance(loss_history[0], list) else loss_history[-1] # Obtiene último error
                    confidences[modality] = 1.0 / (1.0 + final_loss) # Calcula confianza a partir del error
                else: # Si no existe historial...
                    confidences[modality] = 0.5 # Asigna confianza media
            else: # Si la estimación falló...
                confidences[modality] = 0.0 # Asigna confianza nula
        
        # Fusionar según método
        if fusion_method == "weighted_average": # Si se eligió promedio ponderado...
            total_weight = sum(confidences.values()) # Suma todas las confianzas
            if total_weight > 0: # Si hay al menos alguna confianza positiva...
                fused_estimation = torch.zeros_like(results[list(results.keys())[0]]['x_estimated']) # Crea tensor vacío para la estimación final
                for modality, result in results.items(): # Recorre cada resultado individual
                    if result['x_estimated'] is not None: # Si existe estimación válida...
                        weight = confidences[modality] / total_weight # Calcula peso normalizado
                        fused_estimation += weight * result['x_estimated'] # Acumula estimación ponderada
            else: # Si todas las confianzas son cero...
                fused_estimation = results[list(results.keys())[0]]['x_estimated'] # Usa la primera estimación disponible
        
        elif fusion_method == "max_confidence": # Si se eligió la modalidad más confiable...
            best_modality = max(confidences, key=confidences.get) # Busca la modalidad con mayor confianza
            fused_estimation = results[best_modality]['x_estimated'] # Usa sólo esa estimación
        
        else: # Si se especificó un método desconocido...
            fused_estimation = results[list(results.keys())[0]]['x_estimated'] # Usa la primera estimación como respaldo y seguridad
        
        return { # Regresa un diccionario con...
            'fused_estimation': fused_estimation, # Estimación final fusionada
            'individual_results': results, # Resultados individuales
            'confidences': confidences, # Confianzas calculadas
            'fusion_method': fusion_method # Método de fusión usado
        }

# ============================================
# FUNCIONES DE UTILIDAD
# ============================================

def create_test_measurements( # Genera datos de prueba simulando sensores disponibles y no disponibles
    n_sensors: int = 6, # No total de sensores
    n_available: int = 3, # Cantidad de sensores que estarán disponibles
    device: str = "cpu" # Dispositivo donde se crearán los tensores
) -> Tuple[torch.Tensor, torch.Tensor]: # Regresa datos disponibles y su máscara de disponibilidad
    x_full = torch.randn(1, n_sensors, device=device) # Genera una muestra aleatoria para todos los sensores
    
    available_indices = np.random.choice(n_sensors, n_available, replace=False) # Selecciona aleatoriamente qué sensores estarán disponibles
    mask_available = torch.zeros(1, n_sensors, device=device) # Crea una máscara inicial con todos los sensores ocultos
    mask_available[0, available_indices] = 1.0 # Marca como disponibles los sensores seleccionados
    
    x_available = x_full * mask_available # Conserva sólo los sensores disponibles y pone cero en los demás
    
    return x_available, mask_available # Regresa los datos visibles y la máscara

def calculate_estimation_error( # Calcula el error de una estimación solo para los sensores que se estimaron
    x_true: torch.Tensor, # Valores reales de los sensores
    x_estimated: torch.Tensor, # Valores estimados por el modelo
    mask_estimated: torch.Tensor # Máscara que indica qué sensores se evaluarán (1 = estimado)
) -> float: # Regresa el error medio absoluto
    if mask_estimated.sum() == 0: # Si no hay sensores marcados para evaluar...
        return 0.0 # Regresa error cero para evitar divisiones inválidas
    
    error = torch.abs((x_true - x_estimated) * mask_estimated) # Calcula error absoluto
    return error.sum().item() / mask_estimated.sum().item() # Regresa el error promedio por sensor evaluado

# ============================================
# EJEMPLO DE USO COMPLETO
# ============================================

def run_state_estimation_example(): # Ejecuta un ejemplo completo de estimación de estado.
    print("=" * 70)
    print("GeMuCo - Fase 6: State Estimation")
    print("=" * 70)
    
    # 1. Configuración
    n_sensors = 6 # No. total de sensores (3 ángulos + 3 tooltip)
    dim_z = 16 # Tamaño del espacio latente z
    dim_p = 2 # Tamaño del vector de sesgo paramétrico
    n_states = 9 # No de estados posibles del entorno (3 longitudes × 3 ángulos)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguración:")
    print(f"  - n_sensors: {n_sensors}") # Muestra cantidad de sensores
    print(f"  - dim_z: {dim_z}") # Tamaño de z
    print(f"  - dim_p: {dim_p}") # Tamaño de p
    print(f"  - n_states: {n_states}") # Cantidad de estados
    print(f"  - device: {device}") # Dispositivo
    
    # 2. Crear modelo (simulado, con pesos aleatorios para demostración)
    # NOTA: En un caso real, cargaríamos un modelo entrenado
    model = GeMuCoNetwork( # Crea la red neuronal GeMuCo
        n_sensors=n_sensors, # No de sensores
        dim_z=dim_z, # Dim del espacio latente
        dim_p=dim_p, # Dim del sesgo paramétrico
        hidden_sizes=[128, 64, 64, 128], # Tamaños de las capas ocultas
        use_batchnorm=True # Activa normalización por lotes
    ).to(device) # Mueve el modelo al dispositivo
    
    mask_manager = MaskManager(n_sensors) # Crea el gestor de máscaras con máscaras factibles
    
    mask_only_joints = torch.cat([torch.ones(3), torch.zeros(3)]) # Construye máscara donde sólo existen los sensores articulares (ang como entrada)
    mask_manager.add_mask(mask_only_joints) # Agrega la máscara al gestor
    
    mask_manager.add_mask(torch.ones(n_sensors)) # Agrega una máscara donde todos los sensores están disponibles
    
    pb_manager = ParametricBiasManager(dim_p=dim_p, n_states=n_states) # Crea gestor de sesgos paramétricos
    
    latent_optimizer = LatentOptimizer( # Crea optimizador para la variable latente z
        model=model, # Modelo a usar
        learning_rate=0.01, # Tasa de aprendizaje
        n_iterations=30, # No de iteraciones   
        verbose=False
    )
    
    xin_optimizer = XInOptimizer( # Crea optimizador para x_in
        model=model, # Modelo a usar
        mask_manager=mask_manager, # Gestor de máscaras
        learning_rate=0.01, # Tasa de aprendizaje
        n_iterations=30, # No de iteraciones   
        verbose=False
    )
    
    # 6. Crear state estimator
    estimator = StateEstimator( # Crea el estimador principal
        model=model, # Red
        mask_manager=mask_manager, # Gestor de máscaras
        pb_manager=pb_manager, # Gestor de sesgos paramétricos
        latent_optimizer=latent_optimizer, # Optimizador de z
        xin_optimizer=xin_optimizer, # Optimizador de x_in
        device=device # Dispositivo
    )
    
    # 7. Ej 1: Estimación con máscara factible
    print("\n" + "-" * 50)
    print("Ejemplo 1: Estimación con máscara factible (caso a/b)")
    print("-" * 50)
    
    # Simulamos que solo tenemos los ángulos (3 primeros sensores)
    x_available = torch.randn(1, n_sensors, device=device) # Genera una muestra aleatoria
    x_available[0, 3:] = 0 # Oculta los sensores de posición tool-tip no disponibles
    mask_available = torch.zeros(1, n_sensors, device=device) # Crea máscara vacía
    mask_available[0, 0:3] = 1.0 # Marca disponibles los tres primeros sensores (solo ángulos disponibles)
    
    print(f"  Sensores disponibles: índices 0,1,2 (ángulos)")
    print(f"  Valores disponibles: {x_available[0, 0:3].detach().cpu().numpy()}")
    
    result = estimator.estimate_state( # Ejecuta la estimación
        x_available=x_available, # Datos disponibles
        mask_available=mask_available, # Máscara de disponibilidad
        state_idx=0, # Estado usado
        method="mask" # Fuerza el uso del método basado en máscaras
    )
    
    if result['success']: # Si la estimación se realizó correctamente...
        print(f"\n  Resultado:")
        print(f"    Método usado: {result['method_used']}") # Muestra qué método usó el sistema
        print(f"    Tooltip estimado: {result['x_estimated'][0, 3:6].detach().cpu().numpy()}") # Muestra la posición estimada del tool-tip
    
    # 8. Ejemplo 2: Estimación optimizando z
    print("\n" + "-" * 50)
    print("Ejemplo 2: Estimación optimizando z (caso e)")
    print("-" * 50)
    
    # Simulamos mediciones parciales donde no hay máscara factible
    x_available = torch.randn(1, n_sensors, device=device) # Genera una nueva muestra aleatoria
    mask_available = torch.zeros(1, n_sensors, device=device) # Crea una máscara inicialmente vacía
    # Disponemos de sensores no estándar (combinación extraña)
    mask_available[0, [0, 2, 4]] = 1.0 # Marca como disponibles los sensores 0, 2 y 4
    
    print(f"  Sensores disponibles: índices 0, 2, 4")
    print(f"  Valores disponibles: {x_available[0, [0,2,4]].detach().cpu().numpy()}")
    
    result = estimator.estimate_state( # Ejecuta la estimación
        x_available=x_available, # Datos disponibles
        mask_available=mask_available, # Máscara de sensores disponibles
        state_idx=0, # Estado actual
        method="optimize_z", # Fuerza el uso de optimización sobre z
        n_iterations=20 # Realiza 20 iteraciones de optimización
    )
    
    if result['success']: # Si la estimación fue exitosa...
        print(f"\n  Resultado:")
        print(f"    Método usado: {result['method_used']}") # Muestra el método usado
        print(f"    Estimación completa: {result['x_estimated'][0].detach().cpu().numpy()}") # Muestra todos los sensores estimados
        final_loss = result['metadata']['loss_history'][0][-1] if result['metadata']['loss_history'] else 0 # Calcula la pérdida final obtenida
        print(f"    Pérdida final: {final_loss:.6f}")
    
    # 9. Ej 3: Estimación con auto-detcción de método:
    print("\n" + "-" * 50)
    print("Ejemplo 3: Estimación con auto-detección (method='auto')")
    print("-" * 50)
    
    result = estimator.estimate_state( # Ejecuta estimación automática
        x_available=x_available, # Datos disponibles
        mask_available=mask_available, # Máscara de disponibilidad
        state_idx=0, # Estado actual
        method="auto", # Permite que el sistema elija el método
        n_iterations=20 # No de iteraciones permitido
    )
    
    print(f"\n  Resultado:")
    print(f"    Método seleccionado: {result['method_used']}")
    print(f"    Éxito: {result['success']}")
    
    # 10. Mostrar historial
    print("\n" + "-" * 50)
    print("Historial de estimaciones")
    print("-" * 50)
    for entry in estimator.estimation_history[-5:]: # Recorre las últimas 5 estimaciones realizadas
        print(f"  {entry}") # Muestra la información registrada de cada estimación
    
    print("\n" + "=" * 70)
    print("Resumen de la Fase 6")
    print("=" * 70)
    print("""
    Implementado:
    ✓ Caso (a/b): Estimación directa con máscara factible
    ✓ Caso (e): Estimación optimizando z (valores en x_out)
    ✓ Caso (f): Estimación optimizando x_in (valores no en x_out)
    ✓ Auto-detección del mejor método según el artículo
    ✓ Estimación de sensores individuales
    ✓ Estimación multimodal con fusión de sensores
    ✓ Cálculo de errores de estimación
    ✓ Historial de estimaciones

    Correspondencia con el artículo:
    - Sección II-H: Casos (a), (b), (e), (f) de la Figura 3
    - Ecuación (5): Función de pérdida para state estimation
    - "If x_out contains the value to be estimated..." lógica de decisión
    
    Esto permite que el robot:
    - Infiera sensores fallidos o no disponibles
    - Combine múltiples modalidades sensoriales
    - Estime estados incluso con información parcial
    """)

    return estimator


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    estimator = run_state_estimation_example()