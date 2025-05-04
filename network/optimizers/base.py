import numpy as np
from typing import Tuple, Callable, Dict, List
from ..tensor import Variable
from ..gradient_tape import GradientTape

class Optimizer:
    def __init__(self):
        self.__iterations: int = 0
        self.__variables: List[Variable] = []
        pass
        
    def apply_gradients(grads_and_vars: Tuple[Tuple[np.typing.NDArray, np.typing.NDArray]]):
        pass
    
    def compute_gradients(self, loss: Callable, var_list: List[Variable], grad_loss: np.typing.NDArray = None, name: str = None, tape: GradientTape = None):
        pass
    
    def build(self, var_list: List[Variable]):
        pass
    
    def add_slot(self, var: Variable, slot_name: str):
        pass
        
    def get_slot(self, var, slot_name):
        pass
    
    def update_step(gradient: np.typing.NDArray, variable: Variable):
        pass
    
    def get_config(self):
        pass
    
    def get_slot_names():
        pass