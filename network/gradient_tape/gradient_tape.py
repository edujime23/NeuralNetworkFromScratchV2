import numpy as np
from .func_gradients import FuncGradients
from typing import List, Self, Callable, Tuple, Dict, overload, Iterable, Optional

class GradientTape:
    _GRADIENTS_TAPES: List[Self] = []

    def __init__(self, persistent: Optional[bool] = False) -> None:
        self.operations = []
        self.watched: List[np.typing.ArrayLike] = []
        self.persistent = persistent

    def __enter__(self) -> Self:
        self.operations.clear()
        GradientTape._GRADIENTS_TAPES.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        None if self.persistent else GradientTape._GRADIENTS_TAPES.remove(self)
       
    @overload 
    def watch(self, *objs: List[np.typing.ArrayLike]) -> None: ...
    @overload
    def watch(self, obj: np.typing.ArrayLike) -> None: ...
    def watch(self, *objs: np.typing.ArrayLike):
        if isinstance(objs, np.ndarray):
            return self._watch(objs)
        elif isinstance(objs, Iterable):
            if len(objs) <= 1:
                return self._watch(objs[0])
            for o in objs:
                self.watch(o)
            
    def _watch(self, obj: np.typing.ArrayLike):
        if obj not in self.watched:
            self.watched.append(obj)

    def record(self, func: Callable, method: str, inputs: Tuple[np.typing.ArrayLike], kwargs: Dict[str, np.typing.ArrayLike], result: np.typing.ArrayLike) -> None:
        if all(input not in self.watched for input in inputs):
            return
        self.watch(result)
        self.operations.append({
            'func': func,
            'method': method,
            'inputs': inputs,
            'kwargs': kwargs,
            'result': result,
        })
        
    # Overload hell
    @overload
    def gradient(self, targets: List[np.typing.ArrayLike], sources: List[np.typing.ArrayLike]) -> List[List[np.typing.ArrayLike]]: ...
    @overload
    def gradient(self, targets: List[np.typing.ArrayLike], sources: Dict[str, np.typing.ArrayLike]) -> List[Dict[str, np.typing.ArrayLike]]: ...
    @overload
    def gradient(self, targets: List[np.typing.ArrayLike], sources: np.typing.ArrayLike) -> List[Dict[str, np.typing.ArrayLike]]: ...
    @overload
    def gradient(self, targets: Dict[str, np.typing.ArrayLike], sources: List[np.typing.ArrayLike]) -> List[List[np.typing.ArrayLike]]: ...
    @overload
    def gradient(self, targets: Dict[str, np.typing.ArrayLike], sources: Dict[str, np.typing.ArrayLike]) -> List[Dict[str, np.typing.ArrayLike]]: ...
    @overload
    def gradient(self, target: Dict[str, np.typing.ArrayLike], sources: np.typing.ArrayLike) -> np.typing.ArrayLike: ...
    @overload
    def gradient(self, target: np.typing.ArrayLike, sources: List[np.typing.ArrayLike]) -> List[np.typing.ArrayLike]: ...
    @overload
    def gradient(self, target: np.typing.ArrayLike, sources: Dict[str, np.typing.ArrayLike]) -> Dict[str, np.typing.ArrayLike]: ...
    @overload
    def gradient(self, target: np.typing.ArrayLike, sources: np.typing.ArrayLike) -> np.typing.ArrayLike: ...
    def gradient(self, target, sources):
        if isinstance(target, (np.ndarray, np.number)):
            if isinstance(sources, dict):
                return {key: self.gradient(target, val)
                        for key, val in sources.items()}
            elif isinstance(sources, (list, tuple)):
                return [self._gradient(target, src) for src in sources]
            elif isinstance(sources, np.ndarray):
                return self._gradient(target, sources)
            return self._gradient(target, sources)

        if isinstance(target, (list, tuple)):
            return [self.gradient(t, sources) for t in target]

        if isinstance(target, dict):
            return {key: self.gradient(val, sources)
                    for key, val in target.items()}

        raise TypeError(f"Unsupported target type: {type(target)}")
        

    def _gradient(self, target: np.typing.ArrayLike, source: np.typing.ArrayLike):
        if target not in self.watched:
            return None
        grads = { id(target): np.ones_like(target) }
        for op in reversed(self.operations):
            func = op['func']
            inputs = op['inputs']
            result = op['result']
            kwargs = op['kwargs']

            if id(result) not in grads:
                continue

            grad_output = grads[id(result)]
            if hasattr(FuncGradients, func.__name__):
                dfunc = getattr(FuncGradients, func.__name__)
                try:
                    grads_list = dfunc(grad_output, inputs, **kwargs)
                except TypeError: 
                    dfunc(grad_output, inputs)
                    
                for inp, g in zip(inputs, grads_list):
                    grads[id(inp)] = grads.get(id(inp), 0.0) + g
                    
        return grads.get(id(source))
    
    def reset(self, watched: bool = True) -> None:
        self.operations.clear()
        self.watched.clear() if watched else None
    
    def __dell__(self):
        GradientTape._GRADIENTS_TAPES.remove(self)

