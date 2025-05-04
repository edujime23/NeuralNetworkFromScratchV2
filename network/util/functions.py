from typing import Callable, Dict

class function(Callable):
    def __init__(self, func: Callable):
        self.__function = func
        self.__flags: Dict[str, bool] = {}
        
    def __call__(self, *args, **kwargs):
        return self.__function(*args, **kwargs)
    
    @property
    def flags(self):
        return self.__flags
    
    @flags.setter
    def flags(self, value):
        self.__flags = value
        
    @flags.deleter
    def flags(self):
        self.__flags.clear()