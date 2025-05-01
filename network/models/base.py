from typing import List, Optional, Union, Tuple, Callable

import numpy as np

class Model:
    def __init__(self, name: Optional[str] = None):
        self.__name = name or self.__class__.__name__
        self.__optimizer: Union[str, 'Optimizer'] = None
        self.__layers: List['Layer'] = []
        self.__variables: List['Variable'] = []
        self.__built: bool = False
        self.__input_shape: Tuple[int] = None
        self.__output_shape: Tuple[int] = None  
        self.__loss: Callable = None
        self.__losses: List[Callable] = []
        self.__metrics: List[Union[str, 'Metric']] = []
        self.__dtype: np.typing.DTypeLike  
        
    def compile(self, optimizer: Union[str, 'Optimizer'], loss: Callable, metrics: List[Union[str, 'Metric']] = None):
        pass
    
    def build(self, input_shape: Tuple[int], dtype: Optional[np.typing.DTypeLike] = np.float64):
        self.__input_shape = input_shape
        self.__dtype = dtype
        pass
    
    def call(self, inputs: np.typing.NDArray, training: bool, mask: np.typing.NDArray):
        raise NotImplementedError("Should be implemented by Subclass.")
    
    def add_weight(self, shape: Tuple[int], initializer: Union[str, Callable], trainable: bool, name: Optional[str]):
        pass
        
    @property
    def name(self):
        return self.__name
    
    @property
    def built(self):
        return self.__built
        
    @property
    def layers(self):
        return self.__layers
        
    @property
    def variables(self):
        return self.__variables
    
    @property
    def trainable_variables(self):
        return filter(lambda x: x.trainable, self.variables)
    
    @property
    def non_trainable_variables(self):
        return filter(lambda x: not x.trainable, self.variables)