import numpy as np
from .gradients import GRADIENTS
from typing import Any, Iterable, List, Self, Callable, Tuple, Dict, Union, overload, Optional
import warnings

class GradientTape:
    _GRADIENTS_TAPES: List[Self] = []

    def __init__(self, persistent: Optional[bool] = False) -> None:
        self.operations: List[Dict[str, object]] = []
        self.watched: List[int] = []
        self.persistent = persistent

    def __enter__(self) -> Self:
        self.operations.clear()
        GradientTape._GRADIENTS_TAPES.append(self)
        return self

    def __exit__(self, *args: Tuple[object], **kwargs: Dict[str, object]) -> None:
        None if self.persistent else GradientTape._GRADIENTS_TAPES.remove(self)
    
    
    def watch(self, *objs: Tuple[np.typing.NDArray[Any]]) -> None:
        """
        Watches the provided objects for gradient computation.
        Parameters:
            *objs (Tuple[np.typing.NDArray[Any]]): One or more numpy arrays or numbers to be watched.
        Notes:
            - If multiple objects are provided, each is checked to ensure it is a numpy array or number.
            - A warning is issued for any object that is not a numpy array or number, and such objects are skipped.
        """
        
        if len(objs) == 1:
            return self._watch(objs[0])
        for o in objs:
            if not issubclass(type(o), (np.ndarray, np.number)):
                warnings.warn(f"Cannot watch object of type {type(o)}. Only numpy arrays and numbers are supported.")
                continue
            self._watch(o)
            
    def _watch(self, obj: np.typing.NDArray[Any]):
        if id(obj) not in self.watched:
            self.watched.append(id(obj))

    def record(self, func: Callable, method: str, inputs: Tuple[np.typing.NDArray[Any]], kwargs: Dict[str, np.typing.NDArray[Any]], result: np.typing.NDArray[Any]) -> None:
        if all(id(input) not in self.watched for input in inputs):
            return
        self.watch(result)
        self.operations.append({
            'func': func,
            'method': method,
            'inputs': inputs,
            'kwargs': kwargs,
            'result': result,
        })
        
    def gradient(
        self, 
        target: Union[np.typing.NDArray[Any], List[np.typing.NDArray[Any]], Tuple[np.typing.NDArray[Any]], Dict[str, np.typing.NDArray[Any]]], 
        sources: Union[np.typing.NDArray[Any], List[np.typing.NDArray[Any]], Dict[str, np.typing.NDArray[Any]]],
        
    ) -> Union[
        Iterable[Tuple[np.typing.ArrayLike]],
        Tuple[np.typing.ArrayLike],   
    ]:
        if issubclass(type(target), (np.ndarray, np.number)):
            if isinstance(sources, dict):
                return {key: self.gradient(target, val)
                        for key, val in sources.items()}
            elif isinstance(sources, (list, tuple)):
                return [self.gradient(target, src) for src in sources]
            return self._gradient(target, sources)

        elif isinstance(target, (list, tuple)):
            return [self.gradient(t, sources) for t in target]

        elif isinstance(target, dict):
            return {key: self.gradient(val, sources)
                    for key, val in target.items()}

        raise TypeError(f"Unsupported target type: {type(target)}")
    

    def _gradient(self, target: np.typing.NDArray[Any], source: np.typing.NDArray[Any]) -> Union[np.typing.NDArray[Any]]:
        if id(target) not in self.watched:
            return None

        grads: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
            id(target): (np.ones_like(target, dtype=target.dtype), np.zeros_like(target, dtype=target.dtype))
        }

        for op in reversed(self.operations):
            func = op['func']
            inputs = op['inputs']
            result = op['result']
            kwargs = op['kwargs']

            if id(result) not in grads:
                continue

            grad_res = grads[id(result)]
            dfunc = GRADIENTS.get(func.__name__)
            if not dfunc:
                warnings.warn(f"Gradient function for {func.__name__} not found. Skipping.")
                continue

            try:
                raw = dfunc(grad_res, inputs, **kwargs)
            except TypeError:
                raw = dfunc(grad_res, inputs)

            for inp, g in zip(inputs, raw):
                g_h, g_ah = g if isinstance(g, tuple) else (g, np.zeros_like(g))

                prev_h, prev_ah = grads.get(id(inp), (np.zeros_like(g_h), np.zeros_like(g_ah)))
                grads[id(inp)] = (prev_h + g_h, prev_ah + g_ah)

        holo, anti = grads.get(id(source), (0.0, 0.0))

        return (holo, anti)
    
    def reset(self, watched: bool = True) -> None:
        self.operations.clear()
        self.watched.clear() if watched else None
    
    def __del__(self):
        if self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)

