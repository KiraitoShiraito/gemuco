"""07
GeMuCo: Generalized Multisensory Correlational Model         Fase 7: Controlador de robot

Basado en la sección II-I del artículo:
"Depending on the structure of the network, either (a), (b), (e), or (f) in Fig. 3 is computed, as in Section II-H. The calculation depends on
whether the control input is contained in x_in or x_out, and whether the target state can be input directly or the target state needs to be 
expressed in the form of a loss function."

Casos de control según el artículo:
- Actuador en x_out + referencia directa → usar caso a) o b)
- Actuador en x_out + referencia en forma de pérdida → usar caso e) optimizando z
- Actuador no en x_out → usar caso f) optimizando x_in
"""

import torch  # Tensores, redes y cálculo numérico
import torch.nn as nn  # Componentes predefinidos para construir redes
from typing import Optional, List, Tuple, Dict, Any, Callable, Union  # Tipos de datos para documentar parámetros y retornos
import numpy as np  # Op mat y arreglos numéricos
import time  # Trabajar con tiempo y mediciones temporales
from collections import defaultdict  # Diccionario que crea valores por defecto automáticamente

# Importar de fases anteriores
try:
    from model import GeMuCoNetwork, MaskManager, ParametricBiasManager  # Importa red GeMuCo y gestores auxiliares
    from optimizer import LatentOptimizer, XInOptimizer  # Importa los optimizadores desarrollados en la fase anterior
except ImportError:  # Si los archivos no se encuentran o fallan al importar...
    print("Importando módulos locales...")


class GeMuCoController: # Clase encargada de las tareas de control para GeMuCo
    """
    Implementa los casos de control descritos en el artículo (sección II-I):
    - Control con actuador en x_out y referencia directa
    - Control con actuador en x_out y referencia como pérdida
    - Control con actuador no en x_out
    
    El artículo también menciona que la función de pérdida puede tomar varias formas:
    - || A x1_pred - x1_ref ||_2
    - || x1_pred - x1_ref ||_2 + || x2_pred ||_2 (Ej: minimizar energía)
    """
    
    def __init__( # Constructor
        self, # Referencia al objeto actual
        model: GeMuCoNetwork, # Modelo red GeMuCo entrenado
        mask_manager: MaskManager, # Gestor de máscaras válidas M
        pb_manager: ParametricBiasManager, # Gestor de sesgos paramétricos
        latent_optimizer: LatentOptimizer, # Optimizador del espacio latente z ( caso e) )
        xin_optimizer: XInOptimizer, # Optimizador de variables de entrada x_in ( caso f) )
        device: str = "cuda" if torch.cuda.is_available() else "cpu" # Dispositivo de cálculo (GPU o CPU)
    ):
        self.model = model # Guarda el modelo GeMuCo dentro del controlador
        self.mask_manager = mask_manager # El gestor de máscaras
        self.pb_manager = pb_manager # El gestor de sesgos paramétricos
        self.latent_optimizer = latent_optimizer # El optimizador del espacio latente
        self.xin_optimizer = xin_optimizer # El optimizador de entradas
        self.device = device # El dispositivo de ejecución
        
        self.control_history = [] # Donde se guardará el historial de acciones de control
    
    def control_with_direct_input( # Caso a) / b): Calcular acciones de control directamente desde entradas conocidas (actuador esta en x_out)
        self, # Referencia al objeto actual
        target_values: torch.Tensor, # Valores deseados para los sensores objetivo [1, n_sensors]
        mask_target: torch.Tensor, # Máscara que indica qué sensores son importantes para el objetivo [1, n_sensors]
        state_idx: int, # Índice del estado o contexto actual
        actuator_indices: List[int] # Índices de los actuadores que pueden controlarse
    ) -> Tuple[torch.Tensor, bool]: # Regresa comandos de control para los actuadores [1, len(actuador_indices)] y si hay solución
        """
        Según el artículo (sección II-I):
        "If the control input is contained in x_out and all target states can be input directly from x_in, then either (a) or (b) is performed.
        That is, x_out = h(x_in, m)."
        """
        batch_size = target_values.shape[0] # Obtiene cuántas muestras se están procesando simultáneamente
        n_sensors = self.model.n_sensors # Y el número total de sensores del modelo
        
        # Obtener PB para este estado
        p_k = self.pb_manager.get_pb(state_idx) # Recupera el vector paramétrico asociado al estado actual
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1) # Replica ese vector para todas las muestras del lote
        
        # Para control directo, se necesita una máscara que permita especificar los valores objetivo como entrada
        # Esto es, resolver x_out = h(x_in, m, p) para x_in
        
        # Encontrar una máscara que tenga los actuadores como entrada y que pueda predecir los sensores objetivo
        best_mask = None # Donde se guardará la mejor máscara encontrada
        best_score = -1 # Puntaje inicial bajo para poder comparar
        
        for mask in self.mask_manager.get_all_masks(): # Recorre todas las máscaras válidas registradas
            mask_indices = set(torch.where(mask > 0.5)[0].tolist()) # Obtiene los índices activos de la máscara actual
            
            if set(actuator_indices).issubset(mask_indices): # Verifica que todos los actuadores requeridos estén en la mascara
                # También debería poder predecir los sensores objetivo:
                score = len(mask_indices) # Calcula puntuación basada en la cantidad de sensores disponibles
                if score > best_score: # Si esta máscara es mejor que la mejor encontrada hasta ahora...
                    best_score = score # Actualiza la mejor puntuación
                    best_mask = mask # Guarda esta máscara como la mejor opción
        
        if best_mask is None: # Si no hay ninguna máscara compatible...
            return torch.zeros(1, len(actuator_indices)), False # Regresa comandos nulos e indica fallo
        
        m_batch = best_mask.unsqueeze(0).expand(batch_size, -1).to(self.device) # Replica la máscara para todo el lote y la mueve al dispositivo
        
        # Ahora se necesita optimizar x_in para que x_out coincida con target
        def loss_function(x_out_pred: torch.Tensor, x_in_opt: torch.Tensor) -> torch.Tensor:  # Func de pérdida usada durante la optimización
            # Solo nos importan los sensores objetivo
            error = (x_out_pred - target_values) * mask_target # Calcula el error únicamente en los sensores importantes
            return torch.norm(error, p=2) # Regresa la magnitud total del error usando norma Euclidiana
        
        # Optimizar x_in
        x_in_opt, x_out_pred, loss_history = self.xin_optimizer.optimize_x_in_from_predictions( # Ejecuta la optimización de entradas
            loss_function=loss_function, # Función de pérdida
            m=m_batch, # Máscara seleccionada para el control
            p=p_batch, # Parámetros asociados al estado actual
            x_in_init=None, # Comienza desde una entrada inicial automática
            n_iterations=50, # Número de iteraciones de optimización
            learning_rate=0.01, # Tasa de aprendizaje
            return_history=True # Guardar historial de pérdidas
        )
        
        # Extraer comandos de control (solo los actuadores)
        control_commands = x_in_opt[0, actuator_indices] # Extrae sólo los valores correspondientes a los actuadores
        
        self._log_control( # Guarda esta acción de control en el historial
            method="direct_input", # Nombre del método usado
            target=target_values, # Objetivo que se quería alcanzar
            control=control_commands, # Comandos calculados
            loss=loss_history[-1] if loss_history else 0, # Último valor de pérdida obtenido
            success=True # Indica que la operación fue exitosa
        )
        
        return control_commands.unsqueeze(0), True # Regresa los comandos calculados
    
    def control_by_optimizing_z( # Caso e): Controla el sistema buscando el mejor vector latente z (el actuador está en x_out)
        self, # Referencia al objeto actual
        target_reference: Union[torch.Tensor, Callable], # Referencia deseada (tensor o función que calcula pérdida)
        state_idx: int, # Índice del estado o contexto actual
        actuator_indices: List[int], # Índices de sensores o actuadores cuyos valores se usarán como comandos de control
        loss_function_custom: Optional[Callable] = None, # Función de pérdida personalizada opcional
        additional_losses: Optional[List[Tuple[Callable, float]]] = None, # Lista de pérdidas extra con sus pesos
        n_iterations: int = 50, # No de iteraciones de optimización
        learning_rate: float = 0.01, # Tasa de aprendizaje
        z_init: Optional[torch.Tensor] = None, # Valor inicial opcional de z
        return_full_result: bool = False, # Indica si se regresa toda la info del proceso
        verbose: bool = False # Indica si se mostrará progreso
    ) -> Union[Tuple[torch.Tensor, bool], Dict[str, Any]]: # Regresa comandos de control [1, len(actuator_indices)] y éxito, o un dicc completo
        """
        Según el artículo (sección II-I):
        "If the control input is contained in x_out and the target state must be expressed in the form of a loss function, (e) is executed.
        This corresponds to the case where the loss function is executed in the form of h_loss(x_out_pred, x_out_ref)."
        
        El artículo menciona que h_loss puede tomar varias formas, como:
        - || A x1_pred - x1_ref ||_2
        - || x1_pred - x1_ref ||_2 + || x2_pred ||_2
        """
        batch_size = 1 # Se trabaja con una sola muestra
        
        # Obtener PB
        p_k = self.pb_manager.get_pb(state_idx) # Obtiene el sesgo paramétrico asociado al estado actual
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1) # Se adapta al formato de lote
        
        # Definir función de pérdida
        if loss_function_custom is not None: # Si se proporcionó una función de pérdida personalizada...
            loss_fn = loss_function_custom # Se usa directamente
        elif isinstance(target_reference, torch.Tensor): # Si el objetivo es un tensor con valores deseados...
            # Crear pérdida estándar MSE
            def loss_fn(x_out_pred: torch.Tensor) -> torch.Tensor: # Define una función de pérdida
                return torch.norm(x_out_pred - target_reference, p=2) # Calcula la distancia euclidiana al objetivo
        elif callable(target_reference): # Si el objetivo es una función...
            loss_fn = target_reference # Se usa como función de pérdida
        else: # Si el formato es inválido...
            raise ValueError("target_reference debe ser tensor o función callable")
        
        # Añadir pérdidas adicionales si se especifican
        if additional_losses: # Si existen pérdidas adicionales...
            original_loss_fn = loss_fn # Guarda la función original
            def loss_fn_with_additional(x_out_pred: torch.Tensor) -> torch.Tensor: # Crea una nueva función combinada
                loss = original_loss_fn(x_out_pred) # Calcula la pérdida principal
                for add_loss, weight in additional_losses: # Recorre todas las pérdidas adicionales
                    loss = loss + weight * add_loss(x_out_pred) # Suma cada pérdida ponderada
                return loss # Regresa la pérdida total
            loss_fn = loss_fn_with_additional # Reemplaza la función original por la versión extendida
        
        # Optimizar z
        if verbose:
            print(f"Optimizando z para control (caso e)...")
        
        z_opt, x_out_pred, loss_history = self.latent_optimizer.optimize_z_from_x_out_loss( # Busca el mejor z para minimizar la pérdida
            loss_function=loss_fn, # Función de pérdida a minimizar
            z_init=z_init, # Valor inicial de z
            n_iterations=n_iterations, # Número de iteraciones
            learning_rate=learning_rate, # Tasa de aprendizaje
            return_history=True # Guarda el historial de pérdidas
        )
        
        control_commands = x_out_pred[0, actuator_indices] # Extrae los valores de salida que se usarán como comandos de control de x_out_pred
        
        # Registrar
        self._log_control( # Guarda la acción en el historial
            method="optimize_z", # Método usado
            target=target_reference if isinstance(target_reference, torch.Tensor) else None, # Objetivo usado si es tensor
            control=control_commands, # Comandos generados
            loss=loss_history[-1] if loss_history else 0, # Última pérdida obtenida
            success=True, # Marca la operación como exitosa
            z_opt=z_opt # Guarda también el z encontrado
        )
        
        if return_full_result: # Si se pidió toda la información
            return { # Regresa un diccionario con:
                'control_commands': control_commands.unsqueeze(0), # Comandos de control en formato lote
                'success': True,
                'x_out_pred': x_out_pred, # Salida completa predicha
                'z_opt': z_opt, # Vector latente optimizado
                'loss_history': loss_history, # Historial de pérdidas
                'method': 'optimize_z' # Método usado
            }
        
        return control_commands.unsqueeze(0), True # Regresa sólo comandos
    
    def control_by_optimizing_x_in( # Caso f): Controla el sistema buscando directamente los mejores valores de entrada x_in (actuador no está en x_out)
        self, # Referencia al objeto actual
        target_reference: Union[torch.Tensor, Callable], # Objetivo deseado o función que define el objetivo
        state_idx: int, # Índice del estado o contexto actual
        actuator_indices: List[int], # Índices de los actuadores que se desean controlar (en x_in que se optimizaron)
        m_control: Optional[torch.Tensor] = None, # Máscara que especifica entradas de comandos de control (si None, usa todos los actuadores)
        loss_function_custom: Optional[Callable] = None, # Función de pérdida personalizada opcional
        x_in_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, # Restricciones o penalizaciones sobre x_in (límites, suavidad, ...)
        n_iterations: int = 50, # No de iteraciones de optimización
        learning_rate: float = 0.01, # Tasa de aprendizaje
        x_in_init: Optional[torch.Tensor] = None, # Valor inicial opcional para x_in
        return_full_result: bool = False, # Indica si se regresa dicc con info completa
        verbose: bool = False # Progreso
    ) -> Union[Tuple[torch.Tensor, bool], Dict[str, Any]]: # Regresa comandos de control y si hay solución [1, len(actuator_indices)], o un diccionario completo
        """
        Según el artículo (sección II-I):
        "If the control input is not included in x_out, (f) is executed. This means that the variable to be optimized is changed to x_in_opt."
        "Since the loss function can also include the loss with respect to x_in_opt, the loss function is h_loss(x_out_pred, x_out_ref, x_in_opt)."
        """
        batch_size = 1 # Se trabaja con una sola muestra
        
        # Obtener PB
        p_k = self.pb_manager.get_pb(state_idx) # Obtiene el sesgo paramétrico correspondiente al estado actual
        p_batch = p_k.unsqueeze(0).expand(batch_size, -1) # Lo adapta al formato de lote
        
        # Crea máscara para control si no se provee
        if m_control is None: # Si no se proporcionó una máscara de control...
            m_control = torch.zeros(1, self.model.n_sensors, device=self.device) # Crea una máscara inicialmente vacía
            m_control[0, actuator_indices] = 1.0 # Activa únicamente los actuadores indicados
        
        # Definir función de pérdida
        if loss_function_custom is not None: # Si hay una función de pérdida personalizada...
            def loss_fn(x_out_pred: torch.Tensor, x_in_opt: torch.Tensor) -> torch.Tensor: # Define una función intermedia
                return loss_function_custom(x_out_pred, x_in_opt) # Ejecuta la función personalizada
        elif isinstance(target_reference, torch.Tensor): # Si el objetivo es un tensor con valores deseados...
            def loss_fn(x_out_pred: torch.Tensor, x_in_opt: torch.Tensor) -> torch.Tensor: # Define la función de pérdida
                loss = torch.norm(x_out_pred - target_reference, p=2) # Calcula la distancia euclidiana respecto al objetivo
                if x_in_constraints is not None: # Si existen restricciones para x_in...
                    loss = loss + x_in_constraints(x_in_opt) # Añade la penalización correspondiente
                return loss # Regresa la pérdida total
        elif callable(target_reference): # Si el objetivo es una función...
            def loss_fn(x_out_pred: torch.Tensor, x_in_opt: torch.Tensor) -> torch.Tensor: # Define la función de pérdida
                loss = target_reference(x_out_pred) # Evalúa la función objetivo usando la salida predicha
                if x_in_constraints is not None: # Si hay restricciones...
                    loss = loss + x_in_constraints(x_in_opt) # Añade la penalización al valor de pérdida
                return loss # Regresa la pérdida total
        else: # Si el objetivo no tiene un formato válido...
            raise ValueError("target_reference debe ser tensor o función callable")
        
        # Optimizar x_in
        if verbose:
            print(f"Optimizando x_in para control (caso f)...")
        
        x_in_opt, x_out_pred, loss_history = self.xin_optimizer.optimize_x_in_from_predictions( # Busca el mejor x_in para minimizar la pérdida
            loss_function=loss_fn, # Función de pérdida usada
            m=m_control, # Máscara de control
            p=p_batch, # Sesgo paramétrico del estado actual
            x_in_init=x_in_init, # Valor inicial de x_in
            n_iterations=n_iterations, # No de iteraciones
            learning_rate=learning_rate, # Tasa de aprendizaje
            return_history=True # Guarda historial de pérdidas
        )
        
        control_commands = x_in_opt[0, actuator_indices] # Extrae comandos de contro (sólo los actuadores) de interés
        
        # Registrar
        self._log_control( # Guarda la acción en el historial
            method="optimize_x_in", # Método usado
            target=target_reference if isinstance(target_reference, torch.Tensor) else None, # Objetivo usado si es tensor
            control=control_commands, # Comandos generados
            loss=loss_history[-1] if loss_history else 0, # Última pérdida obtenida
            success=True, # Marca la operación como exitosa
            x_in_opt=x_in_opt # Guarda también el x_in optimizado
        )
        
        if return_full_result: # Si se solicitó toda la info...
            return { # Regresa un diccionario con:
                'control_commands': control_commands.unsqueeze(0), # Comandos de control en formato lote
                'success': True, # Indica que la optimización fue exitosa
                'x_out_pred': x_out_pred, # Salida predicha obtenida
                'x_in_opt': x_in_opt, # Valores de entrada optimizados
                'loss_history': loss_history, # Historial completo de pérdidas
                'method': 'optimize_x_in' # Método usado
            }
        
        return control_commands.unsqueeze(0), True # Regresa sólo los comandos y el indicador de éxito
    
    def compute_control( # Decide automáticamente qué caso de comandos de control usar según la estructura de red
        self, # Referencia al objeto actual
        target: Union[torch.Tensor, Dict[str, Any]], # Objetivo o configuración especial de control. Puede ser tensor o dict
        state_idx: int, # Índice del estado actual del sistema
        actuator_indices: List[int], # Índices de los actuadores que se van a controlar
        sensor_indices: List[int], # Índices de los sensores a controlar, relacionados con el objetivo
        method: str = "auto", # Método de control a usar ("auto", "direct_input", "optimize_z", "optimize_x_in")
        **kwargs # Parámetros adicionales opcionales
    ) -> Dict[str, Any]: # Regresa dicc con: Comandos de control, si hay solución, método usado e info adicional
        result = { # Donde se guardará el resultado
            'control_commands': None, # Aquí se almacenarán los comandos calculados
            'success': False, # Inicialmente se asume que no hubo éxito
            'method_used': None, # Método realmente usado
            'metadata': {} # Info adicional del proceso
        }
        
        # Determinar si los actuadores están en x_out o x_in
        # Por ahora, asumimos que los actuadores están en x_in (caso más común)
        # En una implementación completa, esto se determinaría de la estructura
        
        actuators_in_x_out = False # Indica si los actuadores forman parte de la salida x_out (asumir en x_in)
        
        if method == "auto": # Si se pidió selección automática del método...
            # Intentar primero control directo si es posible
            if not actuators_in_x_out: # Si los actuadores NO están representados en x_out...
                # Actuadores en x_in: usar caso (f)
                if isinstance(target, torch.Tensor): # Si el objetivo es un tensor de valores deseados...
                    # Crear máscara para los sensores objetivo
                    mask_target = torch.zeros(1, self.model.n_sensors, device=self.device) # Crea una máscara inicialmente vacía
                    mask_target[0, sensor_indices] = 1.0 # Activa únicamente los sensores relevantes
                    
                    target_tensor = torch.zeros(1, self.model.n_sensors, device=self.device) # Crea un tensor completo de referencia
                    target_tensor[0, sensor_indices] = target # Coloca los valores objetivo en los sensores correspondientes
                    
                    control_result = self.control_by_optimizing_x_in( # Controla optimizando directamente x_in
                        target_reference=target_tensor, # Referencia deseada completa
                        state_idx=state_idx, # Estado actual
                        actuator_indices=actuator_indices, # Actuadores a controlar
                        **kwargs # Parámetros extra
                    )
                elif isinstance(target, dict) and 'loss_function' in target: # Si se proporcionó un diccionario con función de pérdida...
                    control_result = self.control_by_optimizing_x_in( # Usa optimización de x_in
                        target_reference=target['loss_function'], # Función de pérdida personalizada
                        state_idx=state_idx, # Estado actual
                        actuator_indices=actuator_indices, # Actuadores a controlar
                        **kwargs # Parámetros extra
                    )
                else: # Para cualquier otro tipo de objetivo...
                    control_result = self.control_by_optimizing_x_in( # También usa optimización de x_in
                        target_reference=target, # Objetivo proporcionado
                        state_idx=state_idx, # Estado actual
                        actuator_indices=actuator_indices, # Actuadores a controlar
                        **kwargs # Parámetros extra
                    )
                
                if isinstance(control_result, tuple): # Si el método regresó una tupla simple...
                    commands, success = control_result # Separa comandos y estado de éxito
                    result['control_commands'] = commands # Guarda los comandos calculados
                    result['success'] = success # Guarda si tuvo éxito
                    result['method_used'] = 'optimize_x_in' # Registra el método usado
                else: # Si devolvió un diccionario completo...
                    result['control_commands'] = control_result['control_commands'] # Guarda los comandos calculados
                    result['success'] = control_result['success'] # Guarda si tuvo éxito
                    result['method_used'] = control_result['method'] # Guarda el método usado
                    result['metadata'] = control_result # Guarda toda la info adicional
            
            else: # Si los actuadores sí pertenecieran a x_out, usar caso e)
                if isinstance(target, torch.Tensor): # Si el objetivo es un tensor...
                    target_tensor = torch.zeros(1, self.model.n_sensors, device=self.device) # Crea tensor completo de referencia
                    target_tensor[0, sensor_indices] = target # Coloca los valores deseados
                    control_result = self.control_by_optimizing_z( # Controla optimizando el espacio latente z
                        target_reference=target_tensor, # Objetivo completo
                        state_idx=state_idx, # Estado actual
                        actuator_indices=actuator_indices, # Actuadores a controlar
                        **kwargs # Parámetros extra
                    )
                else: # Si el objetivo no es un tensor...
                    control_result = self.control_by_optimizing_z( # También usa optimización de z
                        target_reference=target, # Objetivo proporcionado
                        state_idx=state_idx, # Estado actual
                        actuator_indices=actuator_indices, # Actuadores a controlar
                        **kwargs # Parámetros extra
                    )
                
                if isinstance(control_result, tuple): # Si regresa una tupla...
                    commands, success = control_result # Extrae comandos y éxito
                    result['control_commands'] = commands # Guarda comandos
                    result['success'] = success # Guarda éxito
                    result['method_used'] = 'optimize_z' # Registra método usado
                else: # Si regresa un diccionario...
                    result['control_commands'] = control_result['control_commands'] # Guarda comandos
                    result['success'] = control_result['success'] # Guarda éxito
                    result['method_used'] = control_result['method'] # Guarda método usado
                    result['metadata'] = control_result # Guarda info adicional
        
        elif method == "direct_input": # Si se pidió control directo...
            if isinstance(target, torch.Tensor): # Solo funciona si el objetivo es un tensor
                mask_target = torch.zeros(1, self.model.n_sensors, device=self.device) # Crea máscara vacía
                mask_target[0, sensor_indices] = 1.0 # Activa sensores objetivo
                commands, success = self.control_with_direct_input( # Ejecuta control directo
                    target, mask_target, state_idx, actuator_indices # Val objetivo, máscara de sensores relevante, est actual, actuadores a controlar
                )
                result['control_commands'] = commands # Guarda comandos calculados
                result['success'] = success # Guarda resultado de éxito
                result['method_used'] = 'direct_input' # Registra método usado
        
        elif method == "optimize_z": # Si el usuario usa optimización sobre z...
            if isinstance(target, torch.Tensor): # Si el objetivo es un tensor con valores deseados...
                target_tensor = torch.zeros(1, self.model.n_sensors, device=self.device) # Crea un tensor completo lleno de ceros
                target_tensor[0, sensor_indices] = target # Coloca los valores objetivo en los sensores indicados
                control_result = self.control_by_optimizing_z( # Ejecuta la optimización sobre el espacio latente z
                    target_reference=target_tensor, # Referencia completa a alcanzar
                    state_idx=state_idx, # Estado actual del sistema
                    actuator_indices=actuator_indices, # Actuadores que se desean controlar
                    return_full_result=True, # Solicita toda la info del proceso
                    **kwargs # Parámetros adicionales
                )
            else: # Si el objetivo no es un tensor...
                control_result = self.control_by_optimizing_z( # Ejecuta igual la optimización de z
                    target_reference=target, # Objetivo o función proporcionada
                    state_idx=state_idx, # Estado actual
                    actuator_indices=actuator_indices, # Actuadores que se desean controlar
                    return_full_result=True, # Solicita resultados completos
                    **kwargs # Parámetros adicionales
                )
            result['control_commands'] = control_result['control_commands'] # Guarda los comandos de control calculados
            result['success'] = control_result['success'] # Si la operación tuvo éxito
            result['method_used'] = control_result['method'] # El método realmente usado
            result['metadata'] = control_result # Y toda la información adicional generada
        
        elif method == "optimize_x_in": # Si el usuario usa optimizar x_in...
            if isinstance(target, torch.Tensor): # Si el objetivo es un tensor...
                control_result = self.control_by_optimizing_x_in( # Ejecuta la optimización de x_in
                    target_reference=target, # Objetivo deseado
                    state_idx=state_idx, # Estado actual
                    actuator_indices=actuator_indices, # Actuadores que se controlarán
                    return_full_result=True, # Solicita todos los resultados
                    **kwargs # Parámetros adicionales
                )
            else: # Si el objetivo es otro tipo válido (Ej: Una función)...
                control_result = self.control_by_optimizing_x_in( # Ejecuta la optimización de x_in
                    target_reference=target, # Objetivo recibido
                    state_idx=state_idx, # Estado actual
                    actuator_indices=actuator_indices, # Actuadores a controlar
                    return_full_result=True, # Solicita resultados completos
                    **kwargs # Parámetros adicionales
                )
            result['control_commands'] = control_result['control_commands'] # Guarda los comandos calculados
            result['success'] = control_result['success'] # Si el proceso fue exitoso
            result['method_used'] = control_result['method'] # El nombre del método usado
            result['metadata'] = control_result # Info extra sobre la optimización

        return result # Regresa el resultado final del proceso de control
    
    def _log_control(self, method: str, target, control, loss: float, success: bool, **kwargs): # Registra una acción de control en el historial
        self.control_history.append({ # Agrega un nuevo registro al historial de control
            'timestamp': len(self.control_history), # Usa el tamaño actual del historial como identificador temporal
            'method': method, # Guarda el método de control usado
            'target': target.detach().cpu().numpy() if isinstance(target, torch.Tensor) else target, # Convierte el objetivo a NumPy si es un tensor
            'control': control.detach().cpu().numpy() if isinstance(control, torch.Tensor) else control, # Convierte los comandos de control a NumPy si son tensores
            'loss': loss, # Guarda la pérdida final obtenida
            'success': success, # Guarda si la acción fue exitosa
            **kwargs # Guarda cualquier dato adicional recibido
        })
    
    def get_control_history(self) -> List[Dict]: # Regresa el historial completo de acciones de control
        return self.control_history # Regresa la lista con todos los registros almacenados

# ============================================
# CONTROLADORES ESPECIALIZADOS
# ============================================

class ToolTipController(GeMuCoController): # Controlador especializado para mover el tool-tip (extremo final del robot)
    """
    Basado en el experimento PR2 del artículo (sección III-A).
    """
    def control_tool_tip( # Controla la posición del tool-tip
        self, # Referencia al objeto actual
        target_position: torch.Tensor, # Posición objetivo del tool-tip ( [1, 3] posición deseada (x, y, z) )
        state_idx: int, # Índice del estado actual de agarre del sistema
        joint_angles_init: Optional[torch.Tensor] = None, # Ángulos articulares iniciales opcionales
        n_iterations: int = 50 # No de iteraciones de optimización
    ) -> Dict[str, Any]: # Regresa info del proceso de los comandos de control (ángulos de articulaciones)
        # Crear tensor de referencia completo
        reference = torch.zeros(1, self.model.n_sensors, device=self.device) # Crea un vector de referencia lleno de ceros
        reference[0, 3:6] = target_position # Coloca la posición deseada en las coordenadas del tool-tip en sensores 3,4,5
        
        actuator_indices = [0, 1, 2] # Define qué actuadores controlan el movimiento (son los ángulos de articulaciones, sensores 0,1,2)
        
        # Pérdida adicional: minimizar cambio brusco (como en ecuación 6 del artículo)
        if joint_angles_init is not None: # Si se proporcionaron ángulos iniciales...
            def smoothness_loss(x_in_opt: torch.Tensor) -> torch.Tensor: # Define una pérdida para movimientos suaves
                return 0.3 * torch.norm(x_in_opt[0, actuator_indices] - joint_angles_init, p=2) # Calcula una penalización proporcional a la diferencia
            
            result = self.compute_control( # Ejecuta el control
                target=reference, # Objetivo deseado
                state_idx=state_idx, # Estado actual
                actuator_indices=actuator_indices, # Actuadores involucrados
                sensor_indices=[3, 4, 5], # Sensores correspondientes al tool-tip
                method="optimize_x_in", # Usa optimización directa de x_in
                x_in_constraints=smoothness_loss, # Añade restricción de suavidad
                n_iterations=n_iterations # No de iteraciones
            )
        else: # Si no se proporcionaron ángulos iniciales...
            result = self.compute_control( # Ejecuta el control sin restricción de suavidad
                target=reference, # Objetivo deseado
                state_idx=state_idx, # Estado actual
                actuator_indices=actuator_indices, # Actuadores involucrados
                sensor_indices=[3, 4, 5], # Sensores del tool-tip
                method="optimize_x_in", # Usa optimización de x_in
                n_iterations=n_iterations # No de iteraciones
            )

        return result # Regresa el resultado obtenido


class WholeBodyController(GeMuCoController): # Controlador para controlar varias partes del sistema al mismo tiempo
    def control_with_balance( # Para controlar el sistema manteniendo una tarea principal y equilibrio (la punta del plumero con equilibrio)
        self, # Referencia al propio objeto
        target_tooltip: torch.Tensor, # Posición objetivo deseada para el tool-tip
        target_cog: torch.Tensor, # Posición objetivo deseada para el centro de gravedad [1, 2] deseado
        state_idx: int, # Índice del estado paramétrico que se usará
        weight_tooltip: float = 1.0, # Importancia asignada al objetivo del tool-tip
        weight_cog: float = 0.01, # Importancia asignada al equilibrio (centro de gravedad)
        n_iterations: int = 50 # No de iteraciones de optimización
    ) -> Dict[str, Any]: # Regresa un diccionario con el resultado del control
        """
        Basado en la ecuación (10) del artículo: L = || x_tool_pred - x_tool_ref ||_2 + 0.01 || x_cog_pred - x_cog_ref ||_2
        """
        # Definir función de pérdida multi-objetivo
        def loss_function(x_out_pred: torch.Tensor) -> torch.Tensor: # Función de pérdida multiobjetivo personalizada para optimización
            tooltip_error = torch.norm(x_out_pred[0, 3:6] - target_tooltip, p=2) # Calcula el error entre el tool-tip obtenido y el deseado
            cog_error = torch.norm(x_out_pred[0, 1:3] - target_cog, p=2) # Calcula el error entre el centro de gravedad obtenido y el deseado
            return weight_tooltip * tooltip_error + weight_cog * cog_error # Regresa Combinados ambos errores usando sus pesos
        
        actuator_indices = [0, 1, 2, 3] # Actuadores (ang de articulaciones) que podrán modificarse durante la optimización
        
        result = self.compute_control( # Ejecuta el cálculo del control
            target=loss_function, # Usa la función de pérdida personalizada como objetivo
            state_idx=state_idx, # Estado paramétrico actual
            actuator_indices=actuator_indices, # Actuadores que se optimizarán
            sensor_indices=[3, 4, 5, 1, 2], # Sensores relacionados con tool-tip y centro de gravedad
            method="optimize_z", # Usa optimización del espacio latente z
            n_iterations=n_iterations # No de iteraciones de optimización
        )
        
        return result # Regresa resultado del control

# ============================================
# FUNCIONES DE UTILIDAD PARA PÉRDIDAS COMUNES
# ============================================

def create_end_effector_loss( # Crea una función de pérdida para acercar el efector final a una posición deseada
    end_effector_indices: List[int], # Índices de las variables que representan el efector final
    target_position: torch.Tensor, # Posición objetivo deseada
    weight: float = 1.0 # Peso o importancia de esta pérdida
) -> Callable[[torch.Tensor], torch.Tensor]: # Regresa una función de pérdida
    def loss_fn(x_out_pred: torch.Tensor) -> torch.Tensor: # Función que calculará la pérdida
        error = x_out_pred[0, end_effector_indices] - target_position # Calcula la diferencia entre posición obtenida y deseada
        return weight * torch.norm(error, p=2) # Regresa la distancia euclidiana multiplicada por su peso
    return loss_fn # Regresa la función creada

def create_joint_limit_loss( # Crea una función que penaliza articulaciones fuera de sus límites permitidos.
    joint_indices: List[int], # Índices de las articulaciones a supervisar
    min_angles: torch.Tensor, # Ángulos mínimos permitidos
    max_angles: torch.Tensor, # Ángulos máximos permitidos
    weight: float = 0.1 # Importancia de la penalización
) -> Callable[[torch.Tensor], torch.Tensor]: # Regresa una función de pérdida
    def loss_fn(x_in_opt: torch.Tensor) -> torch.Tensor: # Función que calculará la penalización
        joint_angles = x_in_opt[0, joint_indices] # Obtiene los ángulos actuales de las articulaciones
        # Penalizar violaciones de límites
        below_min = torch.relu(min_angles - joint_angles) # Calcula cuánto se encuentran por debajo del mínimo permitido
        above_max = torch.relu(joint_angles - max_angles) # Y por encima del máximo permitido
        return weight * (torch.sum(below_min) + torch.sum(above_max)) # Calcula la penalización total
    return loss_fn # Regresa la funcióm creada

def create_energy_minimization_loss( # Crea una función para minimizar el esfuerzo o energía (movimiento mínimp) del sistema
    actuator_indices: List[int], # Actuadores que se desean supervisar
    reference_angles: torch.Tensor, # Ángulos de referencia considerados eficientes
    weight: float = 0.01 # Importancia de esta penalización
) -> Callable[[torch.Tensor], torch.Tensor]: # Regresa una función de pérdida
    def loss_fn(x_in_opt: torch.Tensor) -> torch.Tensor: # Función que calculará la penalización
        deviation = x_in_opt[0, actuator_indices] - reference_angles # Diferencia entre ángulos actuales y ángulos de referencia
        return weight * torch.norm(deviation, p=2) # Penaliza desviaciones grandes respecto a la referencia
    return loss_fn # Regresa la función creada

# ============================================
# EJEMPLO DE USO COMPLETO
# ============================================

def run_control_example(): # Ejecuta una demo completa de control con Gemuco
    print("=" * 70)
    print("GeMuCo - Fase 7: Control")
    print("=" * 70)
    
    # 1. Configuración
    n_sensors = 6 # No total de sensores del sistema, 3 ángulos + 3 tooltip
    dim_z = 16 # Tamaño del espacio latente z
    dim_p = 2 # No de parámetros del vector paramétrico p
    n_states = 9 # No de estados paramétricos disponibles
    device = "cuda" if torch.cuda.is_available() else "cpu" # Usa GPU si está disponible, en otro caso CPU
    
    print(f"\nConfiguración:")
    print(f"  - n_sensors: {n_sensors}") # Muestra cantidad de sensores
    print(f"  - dim_z: {dim_z}") # Dimensión de z
    print(f"  - dim_p: {dim_p}") # Dimensión de p
    print(f"  - device: {device}") # Dispositivo
    
    # 2. Crear modelo (simulado)
    model = GeMuCoNetwork( # Crea la red principal GeMuCo
        n_sensors=n_sensors, # No de sensores
        dim_z=dim_z, # Tamaño del espacio latente
        dim_p=dim_p, # Tamaño del vector paramétrico
        hidden_sizes=[128, 64, 64, 128], # Capas ocultas de la red
        use_batchnorm=True # Activa Batch Normalization
    ).to(device) # Mueve la red al dispositivo
    
    # 3. Crear mask manager
    mask_manager = MaskManager(n_sensors) # Crea el gestor de máscaras
    mask_manager.add_mask(torch.cat([torch.ones(3), torch.zeros(3)])) # Agrega una máscara válida al sistema (solo ángulos)
    mask_manager.add_mask(torch.ones(n_sensors)) # Agrega otra máscara válida (todos)
    
    pb_manager = ParametricBiasManager(dim_p=dim_p, n_states=n_states) # Crea el gestor de sesgos paramétricos
    
    # 5. Crear optimizadores
    latent_optimizer = LatentOptimizer( # Crea el optimizador del espacio latente z
        model=model, # Modelo sobre el que trabajará
        learning_rate=0.01, # Tasa de aprendizaje
        n_iterations=30, # No de iteraciones
        verbose=False
    )
    
    xin_optimizer = XInOptimizer( # Crea el optimizador de entradas x_in
        model=model, # Modelo a usar
        mask_manager=mask_manager, # Gestor de máscaras
        learning_rate=0.01, # Tasa de aprendizaje
        n_iterations=30, # No de iteraciones
        verbose=False
    )
    
    # 6. Crear controlador
    controller = GeMuCoController( # Crea el controlador principal
        model=model, # Modelo neuronal
        mask_manager=mask_manager, # Gestor de máscaras
        pb_manager=pb_manager, # Gestor de estados paramétricos
        latent_optimizer=latent_optimizer, # Optimizador de z
        xin_optimizer=xin_optimizer, # Optimizador de x_in
        device=device # Dispositivo
    )
    
    # 7. Ejemplo 1: Control optimizando x_in (caso f)
    print("\n" + "-" * 50)
    print("Ejemplo 1: Control optimizando x_in (caso f)")
    print("-" * 50)
    
    target_tooltip = torch.tensor([0.5, 0.5, 0.5], device=device) # Define posición objetivo [0.5, 0.5, 0.5] del tool-tip
    
    reference = torch.zeros(1, n_sensors, device=device) # Crea vector de referencia completo
    reference[0, 3:6] = target_tooltip # Coloca la posición objetivo en los sensores del tool-tip
    
    actuator_indices = [0, 1, 2] # Define qué actuadores (áng de articulaciones) pueden controlarse
    
    result = controller.compute_control( # Ejecuta el cálculo del control
        target=reference, # Referencia deseada
        state_idx=0, # Estado paramétrico usado
        actuator_indices=actuator_indices, # Actuadores controlables
        sensor_indices=[3, 4, 5], # Sensores asociados al tool-tip
        method="optimize_x_in", # Método basado en optimizar x_in
        n_iterations=30, # No de iteraciones
        verbose=True
    )
    
    if result['success']: # Si la optimización fue exitosa...
        print(f"\n  Resultado:")
        print(f"    Método usado: {result['method_used']}") # Método empleado
        print(f"    Comandos de control (ángulos): {result['control_commands'][0].detach().cpu().numpy()}") # Muestra los comandos calculados
        if 'loss' in result['metadata']: # Verifica si existe info de pérdida...
            print(f"    Pérdida final: {result['metadata']['loss']:.6f}") # Muestra el error final alcanzado
    
    # 8. Ejemplo 2: Control con pérdida personalizada
    print("\n" + "-" * 50)
    print("Ejemplo 2: Control con pérdida personalizada (como ecuación 6 del artículo)")
    print("-" * 50)
    
    # Similar a la ecuación (6) del artículo: L = || x_tool_pred - x_tool_ref ||_2 + 0.3 || θ_opt - θ_orig ||_2
    
    joint_angles_init = torch.tensor([0.1, 0.2, 0.15], device=device) # Define unos ángulos articulares iniciales de referencia
    
    def custom_loss(x_out_pred: torch.Tensor, x_in_opt: torch.Tensor) -> torch.Tensor: # Define una función de pérdida personalizada
        tooltip_error = torch.norm(x_out_pred[0, 3:6] - target_tooltip, p=2) # Calcula el error de posición del tool-tip
        smoothness = 0.3 * torch.norm(x_in_opt[0, 0:3] - joint_angles_init, p=2) # Penaliza movimientos demasiado distintos de la postura inicial
        return tooltip_error + smoothness # Regresa la suma de ambos errores
    
    # Crear máscara para control
    m_control = torch.zeros(1, n_sensors, device=device) # Crea una máscara inicialmente vacía
    m_control[0, actuator_indices] = 1.0 # Activa únicamente los actuadores que podrán controlarse
    
    control_result = controller.control_by_optimizing_x_in( # Ejecuta el control optimizando x_in
        target_reference=custom_loss, # Usa la función de pérdida personalizada
        state_idx=0, # Usa el estado paramétrico 0
        actuator_indices=actuator_indices, # Actuadores que se optimizarán
        m_control=m_control, # Máscara de control
        n_iterations=30, # No de iteraciones de optimización
        return_full_result=True, # Regresa toda la info generada
        verbose=True
    )
    
    print(f"\n  Resultado:")
    print(f"    Comandos de control: {control_result['control_commands'][0].detach().cpu().numpy()}") # Muestra comandos calculados
    print(f"    Pérdida final: {control_result['loss_history'][-1]:.6f}") # Muestra el último valor de la función de pérdida
    
    # 9. Mostrar historial
    print("\n" + "-" * 50)
    print("Historial de control")
    print("-" * 50)
    for entry in controller.control_history[-3:]: # Recorre las últimas 3 acciones registradas
        print(f"  Método: {entry['method']}, Loss: {entry['loss']:.6f}") # Muestra método usado y pérdida obtenida
    
    print("\n" + "=" * 70)
    print("Resumen de la Fase 7")
    print("=" * 70)
    print("""
    Implementado:
    ✓ Control con entrada directa (caso a/b)
    ✓ Control optimizando z (caso e) - actuadores en x_out
    ✓ Control optimizando x_in (caso f) - actuadores no en x_out
    ✓ Auto-detección del método de control
    ✓ Pérdidas personalizadas (incluyendo multi-objetivo)
    ✓ Controladores especializados (ToolTip, WholeBody)
    ✓ Restricciones sobre comandos (límites, suavidad, energía)

    Correspondencia con el artículo:
    - Sección II-I: Casos de control según Figura 3
    - Ecuación (6): L = || x_tool_pred - x_tool_ref ||_2 + 0.3 || θ_opt - θ_orig ||_2
    - Ecuación (7): Control con restricciones de torque
    - Ecuación (10): Control multi-objetivo (tooltip + centro de gravedad)
    
    Esto permite que el robot:
    - Calcule movimientos para alcanzar posiciones deseadas
    - Mantenga equilibrio mientras manipula herramientas
    - Minimice energía o movimiento brusco
    - Combine múltiples objetivos de control
    """)
    
    return controller


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    controller = run_control_example()