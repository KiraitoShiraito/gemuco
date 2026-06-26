"""05
GeMuCo: Generalized Multisensory Correlational Model          Fase 5: Optimizador - Optimización iterativa de z (estado latente)

Basado en la sección II-G del artículo:
    "We describe the optimization computation that is frequently used in the state estimation, control, and simulation."

Algoritmo (sección II-G):
    1. Assign the initial value z_init to the variable z_opt to be optimized
    2. Infer the predicted value of x_out as x_out_pred = h_dec(z_opt)
    3. Calculate the loss L using the loss function h_loss
    4. Calculate ∂L/∂z_opt using the backward propagation
    5. Update z_opt by gradient descent method
    6. Repeat processes 2)-5) to optimize z_opt

El artículo también menciona una técnica para mejorar la convergencia:
    "For example, we determine the maximum value γ_max of γ divide [0, γ_max] equally into N_batch values, and update z_opt with each γ.
    Then, we select z_opt with the smallest L and repeat steps 4) and 5) with various γ."
"""

import torch  # Trabajar con tensores y redes neuronales
import torch.nn as nn  # Capas y componentes de redes neuronales
import torch.optim as optim  # Algoritmos para entrenar redes neuronales
from typing import Optional, Callable, Tuple, List, Dict, Any
import numpy as np  # Cálculos matemáticos y arreglos numéricos
from collections import defaultdict  # Diccionario para crear valores por defecto automáticamente
import time  # Medir tiempos de ejecución

# Importar de fases anteriores
try:
    from model import GeMuCoNetwork, MaskManager
except ImportError:
    print("Importando módulos locales...")


class LatentOptimizer: # Busca el mejor vector de estado latente z mediante optimización iterativa
    """
    Este es el núcleo de muchas operaciones en GeMuCo:
    - State estimation: Cuando no hay máscara factible
    - Control: Cuando el actuador está en x_out
    - Simulation: Para predecir el estado actual desde comandos
    
    El artículo (sección II-G) describe un algoritmo de optimización por gradiente descendente con backpropagation a través de la red.
    """
    def __init__( # Constructor
        self, # Referencia al propio objeto
        model: GeMuCoNetwork, # Red GeMuCo (con encoder y decoder) ya entrenada
        learning_rate: float = 0.01, # Tamaño de tasa usado durante la optimización de z
        n_iterations: int = 50, # No. de iteraciones de optimización
        n_batch_gamma: int = 5, # Cantidad de pruebas n_batch para búsqueda de tasa de aprendizaje. No. de tasas de aprendizaje a probar
        gamma_max: float = 0.1, # Valor máximo permitido para tasa de aprendizaje adaptativo
        verbose: bool = False, # Muestra info detallada durante el proceso
        device: str = "cpu"
    ):
        self.model = model # Guarda el modelo GeMuCo
        self.base_lr = learning_rate # Guarda la tasa de aprendizaje por defecto
        self.n_iterations = n_iterations # Guarda el no. de iteraciones por defecto
        self.n_batch_gamma = n_batch_gamma  # Guarda cuántos valores gamma probar
        self.gamma_max = gamma_max  # Guarda el valor máximo de gamma
        self.verbose = verbose  # Guarda si se mostrarán mensajes de depuración
        self.device = device
        
        self.optimization_history = []  # Donde se almacenará historiales de optimización
    
    def optimize_z_from_x_out_loss(  # Optimiza z para minimizar una función de pérdida definida sobre x_out
        self,
        loss_function: Callable[[torch.Tensor], torch.Tensor], # Func que toma x_out_pred y que calcula qué tan buena es una solución
        z_init: Optional[torch.Tensor] = None,  # Valor inicial opcional para z (Si es None se usa ceros)
        n_iterations: Optional[int] = None,  # No. de iteraciones opcional (si None, usa self.n_iterations)
        learning_rate: Optional[float] = None,  # Tasa de aprendizaje opcional (si None, usa self.base_lr)
        return_history: bool = False  # Indica si se devuelve también el historial de pérdidas
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:  # Regresa z optimizado, salida z_out generada e historial de pérdidas
        """
        Este es el caso principal (e) en la Figura 3 del artículo: "the loss function is executed in the form of h_loss(x_out_pred, x_out_ref)"
        """
        if n_iterations is None:  # Si no se especificó no de iteraciones...
            n_iterations = self.n_iterations  # Usa el valor por defecto
        if learning_rate is None:  # Si no se especificó tasa de aprendizaje...
            learning_rate = self.base_lr  # Usa el valor por defecto
        
        # Inicializar z_opt
        if z_init is None:  # Si no se proporcionó un z inicial...
            z_opt = torch.zeros(1, self.model.dim_z, requires_grad=True, device=self.device)  # Crea un vector z lleno de 0s que podrá modificarse
        else: # Sino...
            z_opt = z_init.clone().detach().requires_grad_(True).to(self.device)  # Copia el z inicial y habilita cálculo de gradientes
        
        optimizer = optim.Adam([z_opt], lr=learning_rate) # Crea optimizador Adam que modificará z
        
        loss_history = [] # Donde se guardarán las pérdidas
        
        self.model.eval()  # Pone la red en evaluación (no entrenar, solo optimizar z)
        
        for iteration in range(n_iterations): # Repite el proceso de optimización varias veces
            optimizer.zero_grad() # Borra gradientes anteriores
            
            x_out_pred = self.model.decode(z_opt)  # Convierte z en una salida x_out usando el decoder
            
            # Calcular pérdida
            loss = loss_function(x_out_pred)  # Calcula qué tan buena es la salida
            loss_history.append(loss.item())  # Guarda el valor numérico de la pérdida
            
            loss.backward() # Calcula gradientes para saber cómo modificar z
            
            optimizer.step() # Actualiza z usando los gradientes
            
            if self.verbose and iteration % 10 == 0: # Si está activado modo detallado y toca mostrar progreso...
                print(f"  Iter {iteration}: loss = {loss.item():.6f}, z_norm = {z_opt.norm().item():.4f}")  # Muestra info del proceso
        
        # Decodificar el z final
        x_out_opt = self.model.decode(z_opt).detach()  # Genera la salida final usando el mejor z encontrado
        z_opt_final = z_opt.detach()  # Desconecta z del sistema de gradientes
        
        if return_history:  # Si se pidió regresar historial...
            return z_opt_final, x_out_opt, loss_history  # Regresa z, salida e historial
        return z_opt_final, x_out_opt  # Regresa únicamente z y salida
    
    def optimize_z_from_measurements( # Optimiza z para que x_out_pred coincida con las mediciones disponibles
        self,
        x_out_measured: torch.Tensor,  # Valores medidos en sensores. Mediciones parciales de x_out [batch_size, n_sensors]
        mask_x_out: torch.Tensor,  # Máscara que indica qué sensores fueron medidos (1=medido, 0=no medido)
        z_init: Optional[torch.Tensor] = None,  # Valor de z inicial opcional (Si es None usa 0s)
        n_iterations: Optional[int] = None,  # No. de iteraciones opcional
        learning_rate: Optional[float] = None,  # Tasa de aprendizaje opcional
        return_history: bool = False  # Si se devuelve historial...
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:  # Regresa estado latente z optimizado, salida x_out estimada e historial de pérdidas
        """        
        Esta es la forma (e) de state estimation en el artículo (sección II-H): "h_loss(x_out_pred, x_out_data) = || m_x_out ⊙ (x_out_pred - x_out_data) ||_2"
        
        Args: x_out_measured: (los valores no medidos pueden ser cualquier valor)
        """
        def loss_function(x_out_pred: torch.Tensor) -> torch.Tensor:  # Func de pérdida interna
            # Ecuación (5) del artículo
            error = (x_out_pred - x_out_measured) * mask_x_out  # Calcula error solo en sensores medidos
            return torch.norm(error, p=2)  # Devuelve magnitud total del error (L2 norm) (MSE sería al cuadrado)
        
        return self.optimize_z_from_x_out_loss(  # Reutiliza el optimizador general
            loss_function=loss_function,  # Usa la pérdida definida arriba
            z_init=z_init,  # Pasa z inicial
            n_iterations=n_iterations,  # Pasa no de iteraciones
            learning_rate=learning_rate,  # Pasa tasa de aprendizaje
            return_history=return_history  # Pasa opción de historial
        )
    
    def optimize_z_from_reference(  # Optimiza z para acercarse a una referencia deseada en x_out
        self,
        x_out_ref: torch.Tensor,  # Salida deseada para x_out [1, n_sensors] o [batch_size, n_sensors]
        weight_matrix: Optional[torch.Tensor] = None,  # Matriz de pesos opcionales para cada componente
        additional_losses: Optional[List[Tuple[Callable, float]]] = None,  # Lista de (loss func, peso) de restricciones o pérdidas adicionales
        z_init: Optional[torch.Tensor] = None,  # z inicial opcional
        n_iterations: Optional[int] = None,  # No. de iteraciones opcional
        learning_rate: Optional[float] = None,  # Tasa de aprendizaje opcional
        return_history: bool = False  # Si se devuelve historial...
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:  # Regresa Estado latente z óptimo, salida x_out obtenida e historial de pérdidas
        """
        Esta es la forma (e) de control en el artículo (sección II-I): "the loss function is executed in the form of h_loss(x_out_pred, x_out_ref)"
        
        El artículo menciona que h_loss puede tomar varias formas, como:
            - || A x1_pred - x1_ref ||_2
            - || x1_pred - x1_ref ||_2 + || x2_pred ||_2
        """
        def loss_function(x_out_pred: torch.Tensor) -> torch.Tensor:  # Func de pérdida personalizada
            # Pérdida principal: diferencia con referencia
            if weight_matrix is not None:  # Si existen pesos...
                error = (x_out_pred - x_out_ref) * weight_matrix  # Error ponderado
            else:
                error = x_out_pred - x_out_ref  # Error normal
            
            loss = torch.norm(error, p=2)  # Magnitud total del error
            
            # Pérdidas adicionales (Ej: regularización)
            if additional_losses:  # Si existen pérdidas adicionales...
                for add_loss, weight in additional_losses:  # Recorre cada restricción
                    loss = loss + weight * add_loss(x_out_pred)  # Añade penalización extra
            
            return loss  # Regresa pérdida total
        
        return self.optimize_z_from_x_out_loss(  # Usa el optimizador general
            loss_function=loss_function,  # Func de pérdida creada
            z_init=z_init,  # z inicial
            n_iterations=n_iterations,  # No. de iteraciones
            learning_rate=learning_rate,  # Tasa de aprendizaje
            return_history=return_history  # Historial opcional
        )
    
    def optimize_z_with_lr_search(  # Optimiza z probando varias tasas de aprendizaje automáticamente
        self,
        loss_function: Callable[[torch.Tensor], torch.Tensor],  # Func de pérdida
        z_init: Optional[torch.Tensor] = None,  # z inicial opcional
        n_iterations: int = 30,  # No. de iteraciones
        n_search_steps: int = 5,  # Cantidad de tasas de aprendizaje a probar por iteración
        gamma_min: float = 0.001,  # Tasa de aprendizaje mínimo
        gamma_max: float = 0.1  # Tasa de aprendizaje máximo
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:  # Regresa estado latente z final, salida x_out final y mejor pérdida
        """
        Según el artículo (sección II-G):
            "determine the maximum value γ_max of γ divide [0, γ_max] equally into N_batch values, and update z_opt with each γ. Then, we select
            z_opt with the smallest L in steps 2) and 3), and repeat steps 4) and 5) with various γ for the z_opt."
        """
        if z_init is None:  # Si no se proporcionó estado latente z inicial...
            z_opt = torch.zeros(1, self.model.dim_z, requires_grad=True, device=self.device)  # Crea z inicial en cero
        else:
            z_opt = z_init.clone().detach().requires_grad_(True).to(self.device)  # Copia z inicial
        
        self.model.eval()  # Modo evaluación
        
        gammas = np.linspace(gamma_min, gamma_max, n_search_steps) # Genera varias tasas de aprendizaje entre mínimo y máximo
        
        for iteration in range(n_iterations):  # Bucle principal
            best_z = None  # Mejor z encontrado hasta ahora
            best_loss = float('inf')  # Inicializa mejor pérdida con infinito
            
            # Probar diferentes tasas
            for gamma in gammas:  # Prueba cada tasa
                z_test = z_opt.clone().detach().requires_grad_(True)  # Crea copia temporal de z
                optimizer = optim.Adam([z_test], lr=gamma)  # Optimizador usando ese gamma
                
                # Un paso de actualización
                optimizer.zero_grad()  # Limpia gradientes
                x_out_pred = self.model.decode(z_test)  # Genera salida
                loss = loss_function(x_out_pred)  # Calcula pérdida
                loss.backward()  # Calcula gradientes
                optimizer.step()  # Actualiza z temporal
                
                # Evaluar la nueva pérdida
                with torch.no_grad():  # Sin calcular gradientes
                    x_out_pred_new = self.model.decode(z_test)  # Nueva salida después de actualizar
                    loss_new = loss_function(x_out_pred_new).item()  # Calcula nueva pérdida
                
                if loss_new < best_loss:  # Si encontró una mejor solución...
                    best_loss = loss_new  # Guarda mejor pérdida
                    best_z = z_test.clone().detach()  # Guarda mejor estado latente z
            
            # Actualizar z_opt con el mejor resultado
            if best_z is not None:  # Si existe una mejor solución...
                z_opt = best_z.clone().detach().requires_grad_(True)  # Actualiza z principal
            
            if self.verbose and iteration % 5 == 0:  # Mostrar progreso opcional
                print(f"  Iter {iteration}: best_loss = {best_loss:.6f}")  # Imprime mejor pérdida
        
        # Decodificar el z final
        x_out_opt = self.model.decode(z_opt).detach()  # Genera salida final
        
        return z_opt.detach(), x_out_opt, best_loss # Regresa resultado final


class XInOptimizer:  # Optimiza directamente los valores de entrada x_in (entrada de la red) en lugar del vector latente z
    """
    Esto corresponde al caso (f) en el artículo (secciones II-H, II-I):
        "the variable to be optimized z_opt and its initial value z_init are changed to x_in_opt and x_in_init, respectively."
    
    Se usa cuando:
        - El valor a estimar no está en x_out (state estimation)
        - El actuador no está en x_out (control)
    """
    def __init__(  # Constructor
        self,  # Referencia al propio objeto
        model: GeMuCoNetwork,  # Red GeMuCo entrenada
        mask_manager: MaskManager,  # Gestor de máscaras válidas
        learning_rate: float = 0.01,  # Tasa de aprendizaje para la optimización
        n_iterations: int = 50,  # No. de iteraciones de optimización
        verbose: bool = False,  # Mostrar mensajes detallados
        device: str = "cpu"
    ):
        self.model = model  # Guarda el modelo GeMuCo
        self.mask_manager = mask_manager  # El gestor de máscaras
        self.base_lr = learning_rate  # La tasa de aprendizaje por defecto
        self.n_iterations = n_iterations  # El no. de iteraciones por defecto
        self.verbose = verbose
        self.device = device
    
    def optimize_x_in_from_predictions(  # Optimiza los valores de entrada x_in para min una func de pérdida sobre x_out
        self,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], # Func que toma (x_out_pred, x_out_ref) y retorna pérdida
        m: torch.Tensor,  # Máscara que indica qué sensores están disponibles
        p: torch.Tensor,  # Sesgo paramétrico asociado al estado actual
        x_in_init: Optional[torch.Tensor] = None,  # Valor inicial opcional para x_in (si None, usa 0s)
        n_iterations: Optional[int] = None,  # No. de iteraciones opcional
        learning_rate: Optional[float] = None,  # Tasa de aprendizaje opcional
        return_history: bool = False  # Indica si debe regresar el historial de pérdidas
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:  # Regresa x_in optimizado, salida x_out generada e historial de pérdidas
        if n_iterations is None:  # Si no se especificó cantidad de iteraciones...
            n_iterations = self.n_iterations  # Usa el valor por defecto
        if learning_rate is None:  # Si no se especificó tasa de aprendizaje...
            learning_rate = self.base_lr  # Usa el valor guardado
        
        # Inicializar x_in_opt
        if x_in_init is None:  # Si no se proporcionó una entrada inicial...
            x_in_opt = torch.zeros(1, self.model.n_sensors, requires_grad=True, device=self.device) # Crea un vector inicial lleno de 0s
        else:
            x_in_opt = x_in_init.clone().detach().requires_grad_(True).to(self.device) # Copia la entrada inicial y habilita gradientes
        
        # Asegurar que m y p estén en el mismo dispositivo
        m = m.to(self.device)  # Mueve la máscara al mismo dispositivo que el modelo
        p = p.to(self.device)  # Mueve el sesgo paramétrico al mismo dispositivo
        
        optimizer = optim.Adam([x_in_opt], lr=learning_rate) # Crea optimizador Adam para modificar x_in
        
        loss_history = []  # Donde se guardarán las pérdidas
        
        self.model.eval()  # Coloca el modelo en modo evaluación
        
        for iteration in range(n_iterations):  # Repite el proceso de optimización
            optimizer.zero_grad()  # Borra gradientes anteriores
            
            # Forward pass con x_in_opt
            x_out_pred, _ = self.model(x_in_opt, m, p)  # Genera una predicción usando la entrada actual
            
            # Calcular pérdida
            loss = loss_function(x_out_pred, x_in_opt)  # Calcula qué tan buena es la solución actual
            loss_history.append(loss.item())  # Guarda el valor numérico de la pérdida
            
            loss.backward() # Calcula gradientes respecto a x_in
            
            optimizer.step() # Modifica x_in para reducir el error
            
            if self.verbose and iteration % 10 == 0: # Si se desea mostrar progreso cada 10 iteraciones...
                print(f"  Iter {iteration}: loss = {loss.item():.6f}, x_in_norm = {x_in_opt.norm().item():.4f}") # Muestra estado actual de la optimización
        
        # Predicción final
        with torch.no_grad():  # Ejecuta sin almacenar gradientes
            x_out_opt, _ = self.model(x_in_opt, m, p)  # Calcula la salida final usando la mejor entrada encontrada
        
        x_in_opt_final = x_in_opt.detach()  # Desconecta x_in del sistema de gradientes
        
        if return_history:  # Si se solicitó historial...
            return x_in_opt_final, x_out_opt, loss_history  # Regresa entrada, salida e historial
        return x_in_opt_final, x_out_opt  # Regresa sólo entrada y salida
    
    def optimize_x_in_from_measurements(  # Optimiza las entradas x_in para que x_out_pred coincida con mediciones
        self,
        x_out_measured: torch.Tensor, # Mediciones parciales de x_out
        m: torch.Tensor,  # Máscara de sensores x_in disponibles
        p: torch.Tensor,  # Sesgo paramétrico del estado actual
        mask_x_out: torch.Tensor,  # Máscara que indica qué componentes x_out fueron medidos
        x_in_init: Optional[torch.Tensor] = None,  # Entrada x_in inicial opcional
        n_iterations: Optional[int] = None,  # No. de iteraciones opcional
        learning_rate: Optional[float] = None,  # Tasa de aprendizaje opcional
        return_history: bool = False  # Si se devolverá historial
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:  # Regresa entrada optimizada (estimación del estado), salida x_out e historial de pérdidas
        """
        Esta es la forma (f) de state estimation en el artículo (sección II-H):
            "the loss function is the same as Eq. 5. That is, instead of the latent representation z, we propagate the error directly to the network input x_in"
        """
        def loss_function(x_out_pred: torch.Tensor, x_in_opt: torch.Tensor) -> torch.Tensor: # Func de pérdida interna
            # Similar a ecuación (5) pero con x_out_pred
            error = (x_out_pred - x_out_measured) * mask_x_out  # Calcula error únicamente en sensores observados
            return torch.norm(error, p=2)  # Regresa magnitud total del error
        
        return self.optimize_x_in_from_predictions(  # Reutiliza el optimizador general
            loss_function=loss_function,  # Func de pérdida definida arriba
            m=m,  # Máscara de sensores
            p=p,  # Sesgo paramétrico
            x_in_init=x_in_init,  # Entrada inicial
            n_iterations=n_iterations,  # No de iteraciones
            learning_rate=learning_rate,  # Tasa de aprendizaje
            return_history=return_history  # Historial opcional
        )
    
    def optimize_x_in_for_control(  # Optimiza entradas x_in para alcanzar un objetivo x_out de control
        self,
        x_out_ref: torch.Tensor,  # Salida x_out deseada que se quiere alcanzar
        m: torch.Tensor,  # Máscara de sensores disponibles (indica qué x_in son comandos de control)
        p: torch.Tensor,  # Sesgo paramétrico del estado actual
        x_in_constraints: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, # Func de pérdida adicional para x_in (Ej: regularización)
        x_in_init: Optional[torch.Tensor] = None,  # Entrada x_in inicial opcional
        n_iterations: Optional[int] = None,  # No. de iteraciones opcional
        learning_rate: Optional[float] = None,  # Tasa de aprendizaje opcional
        return_history: bool = False  # Si se devolverá historial
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:  # Regresa entrada óptima (comandos de control), salida obtenida e historial
        """
        Esta es la forma (f) de control en el artículo (sección II-I): "the loss function can also include the loss with respect to x_in_opt"
        """
        def loss_function(x_out_pred: torch.Tensor, x_in_opt: torch.Tensor) -> torch.Tensor: # Func de pérdida personalizada
            # Pérdida principal: error en referencia
            loss = torch.norm(x_out_pred - x_out_ref, p=2) # Calcula distancia entre salida actual y objetivo
            
            # Restricciones adicionales sobre x_in (Ej: suavidad, límites)
            if x_in_constraints is not None:  # Si hay restricciones...
                loss = loss + x_in_constraints(x_in_opt)  # Añade penalización extra
            
            return loss  # Regresa pérdida total
        
        return self.optimize_x_in_from_predictions(  # Usa el optimizador general
            loss_function=loss_function,  # Func de pérdida creada
            m=m,  # Máscara de sensores
            p=p,  # Sesgo paramétrico
            x_in_init=x_in_init,  # Entrada inicial
            n_iterations=n_iterations,  # Iteraciones
            learning_rate=learning_rate,  # Tasa de aprendizaje
            return_history=return_history  # Historial opcional
        )

# ============================================
# FUNCIONES AUXILIARES PARA PÉRDIDAS COMUNES
# ============================================

def create_mse_loss(reference: torch.Tensor, mask: Optional[torch.Tensor] = None): # Crea una func de pérdida MSE con máscara, personalizada
    def loss_fn(x_out_pred: torch.Tensor) -> torch.Tensor: # Func que calculará el error
        if mask is not None: # Si hay una máscara
            error = (x_out_pred - reference) * mask # Calcula error sólo en componentes seleccionados
        else: # Si no existe máscara
            error = x_out_pred - reference # Usa todos los componentes
        return torch.mean(error ** 2) # Regresa el Error Cuadrático Medio (MSE)
    return loss_fn # Regresa la func creada


def create_weighted_loss( # Crea una func de pérdida ponderada
    reference: torch.Tensor, # Valores objetivo
    weights: torch.Tensor # Peso o importancia de cada componente
) -> Callable[[torch.Tensor], torch.Tensor]: # Regresa una func de pérdida
    """Crea una función de pérdida ponderada"""
    def loss_fn(x_out_pred: torch.Tensor) -> torch.Tensor: # Func que calculará el error
        error = (x_out_pred - reference) * weights # Error multiplicado por los pesos
        return torch.norm(error, p=2) # Regresa magnitud total del error ponderado
    return loss_fn # Regresa la función creada

def create_multi_objective_loss( # Crea una func de pérdida con varios objetivos simultáneos
    objectives: List[Tuple[torch.Tensor, torch.Tensor, float]] # Lista de objetivos, referencias y pesos
) -> Callable[[torch.Tensor], torch.Tensor]: # Regresa una func de pérdida
    def loss_fn(x_out_pred: torch.Tensor) -> torch.Tensor: # Func que calculará el error total
        total_loss = 0.0 # Inicializa pérdida acumulada
        for component, ref, weight in objectives: # Recorre cada objetivo
            error = component(x_out_pred) - ref # Calcula diferencia respecto al objetivo
            total_loss = total_loss + weight * torch.norm(error, p=2) # Suma error ponderado al total
        return total_loss # Regresa pérdida total
    return loss_fn # Regresa la función creada

# ============================================
# EJEMPLO DE USO COMPLETO
# ============================================

def run_optimization_example(): # Func principal que ejecuta un ejemplo completo de optimización iterativa de z y x_in
    print("=" * 70)
    print("GeMuCo - Fase 5: Optimización Iterativa de z")
    print("=" * 70)
    
    # 1. Configuración
    n_sensors = 6 # No. total de sensores usados (3 áng y 3 tooltip)
    dim_z = 16 # Tamaño del espacio latente z
    dim_p = 2 # Dim del sesgo paramétrico
    device = "cuda" if torch.cuda.is_available() else "cpu" # Usa GPU si existe; si no, CPU
    
    # 2. Crear modelo (simulado, sin entrenar para este ejemplo)
    model = GeMuCoNetwork( # Crea una nueva red GeMuCo
        n_sensors=n_sensors, # No. de sensores de entrada/salida
        dim_z=dim_z, # Tamaño del vector latente z
        dim_p=dim_p, # Tamaño del vector de sesgo paramétrico
        hidden_sizes=[128, 64, 64, 128], # Tamaños de las capas ocultas
        use_batchnorm=True # Activa normalización interna
    ).to(device) # Mueve el modelo al dispositivo
    
    # Para este ejemplo, se usa pesos aleatorios (simulando un modelo ya entrenado)
    print(f"\nModelo creado: {model.get_num_params():,} parámetros") # Muestra cuántos parámetros tiene la red
    
    # 3. Crear optimizadores
    latent_optimizer = LatentOptimizer( # Crea el optimizador encargado de modificar z
        model=model, # Le pasa el modelo que usará
        learning_rate=0.01, # Tasa de aprendizaje
        n_iterations=30, # No. de iteraciones por optimización
        verbose=True
    )
    
    # 4. Ejemplo 1: Optimizar z para alcanzar una referencia en x_out
    print("\n" + "-" * 50)
    print("Ejemplo 1: Optimizar z para alcanzar una referencia en tool-tip")
    print("-" * 50)
    
    # Referencia deseada: tool-tip en posición [0.5, 0.5, 0.5] (normalizada)
    x_out_ref = torch.zeros(1, n_sensors, device=device) # Crea un vector de referencia lleno de 0s
    x_out_ref[0, 3:6] = torch.tensor([0.5, 0.5, 0.5]) # Define como objetivo una posición concreta del tool-tip
      # solo tooltip
    
    def example_loss(x_out_pred): # Func que calcula qué tan lejos está la predicción del objetivo
        # Solo nos importa el error en tool-tip (sensores 3,4,5)
        error = x_out_pred[0, 3:6] - x_out_ref[0, 3:6] # Calcula el error sólo en las coordenadas del tool-tip
        return torch.norm(error, p=2) # Regresa la distancia Euclidiana al objetivo
    
    z_opt, x_out_opt, loss_history = latent_optimizer.optimize_z_from_x_out_loss( # Busca el mejor vector z
        loss_function=example_loss, # Usa la función de error definida arriba
        n_iterations=30, # Realiza 30 iteraciones
        return_history=True # Guarda historial de errores
    )
    
    print(f"\nResultado:")
    print(f"  z_opt norm: {z_opt.norm().item():.4f}") # Muestra el tamaño del vector z encontrado
    print(f"  x_out_opt (tooltip): {x_out_opt[0, 3:6].detach().cpu().numpy()}") # Muestra la posición final obtenida
    print(f"  Referencia deseada: {x_out_ref[0, 3:6].detach().cpu().numpy()}") # Muestra la posición objetivo
    print(f"  Error final: {loss_history[-1]:.6f}") # Muestra el error final conseguido
    
    # 5. Ejemplo 2: Optimizar z desde mediciones parciales
    print("\n" + "-" * 50)
    print("Ejemplo 2: Optimizar z desde mediciones parciales (state estimation)")
    print("-" * 50)
    
    # Simulamos que solo medimos los ángulos (sensores 0,1,2) y queremos estimar tooltip
    x_out_measured = torch.zeros(1, n_sensors, device=device) # Crea un vector para almacenar mediciones
    x_out_measured[0, 0:3] = torch.tensor([0.3, 0.2, 0.1]) # Supone que sólo conocemos algunos ángulos
    
    mask_x_out = torch.zeros(1, n_sensors, device=device) # Crea máscara indicando qué datos existen
    mask_x_out[0, 0:3] = 1.0 # Marca los primeros tres sensores como observados (solo ángulos medidos)
    
    z_opt, x_out_est, loss_history = latent_optimizer.optimize_z_from_measurements( # Busca el z que mejor explique esas mediciones
        x_out_measured=x_out_measured, # Mediciones observadas
        mask_x_out=mask_x_out, # Máscara de observación
        n_iterations=30, # No. de iteraciones
        return_history=True # Guarda historial
    )
    
    print(f"\nResultado:")
    print(f"  Ángulos medidos: {x_out_measured[0, 0:3].detach().cpu().numpy()}") # Muestra los ángulos conocidos
    print(f"  Tooltip estimado: {x_out_est[0, 3:6].detach().cpu().numpy()}") # La posición estimada del tool-tip
    print(f"  Error final: {loss_history[-1]:.6f}") # El error final
    
    # 6. Ejemplo 3: Optimización con búsqueda de learning rate
    print("\n" + "-" * 50)
    print("Ejemplo 3: Optimización con búsqueda adaptativa de learning rate")
    print("-" * 50)
    
    z_opt, x_out_opt, best_loss = latent_optimizer.optimize_z_with_lr_search( # Optimiza usando varias tasas automáticamente
        loss_function=example_loss, # Utiliza la misma func de error del ejemplo 1
        n_iterations=20, # Realiza 20 iteraciones
        n_search_steps=5, # Prueba 5 valores distintos de tasas
        gamma_min=0.001, # Tasa mínimo
        gamma_max=0.05 # Tasa máximo
    )
    
    print(f"\nResultado:")
    print(f"  Mejor pérdida: {best_loss:.6f}") # Muestra el menor error encontrado
    print(f"  Tooltip obtenido: {x_out_opt[0, 3:6].detach().cpu().numpy()}") # Muestra la posición final alcanzada
    
    print("\n" + "=" * 70)
    print("Resumen de la Fase 5")
    print("=" * 70)
    print("""
    Implementado:
    ✓ Optimización iterativa de z (estado latente) con backpropagation
    ✓ Soporte para pérdidas arbitrarias (control, state estimation)
    ✓ Búsqueda adaptativa de learning rates (técnica del artículo)
    ✓ Optimización de x_in (caso f: cuando el actuador no está en x_out)
    ✓ Funciones auxiliares para pérdidas comunes (MSE, ponderada, multi-objetivo)

    Correspondencia con el artículo:
    - Sección II-G: Algoritmo completo de optimización iterativa
    - Ecuación (4): Actualización por gradiente descendente
    - Técnica de múltiples learning rates para convergencia más rápida
    - Casos (e) y (f) de la Figura 3
    
    Esto sienta las bases para:
    - State estimation (Fase 6): usar optimización de z o x_in con mediciones parciales
    - Control (Fase 7): usar optimización de z o x_in con referencias deseadas
    - Simulation (Fase 8): similar a state estimation con comandos enviados
    """)

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================

if __name__ == "__main__": # Comprueba si este archivo se está ejecutando directamente
    run_optimization_example() # Ejecuta el ejemplo completo y guarda los resultados