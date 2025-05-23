import numpy as np
from .gradients import GRADIENTS, numerical_derivative
from typing import Any, Iterable, List, Self, Callable, Tuple, Dict, Union, Optional
import warnings


class GradientTape:
    _GRADIENTS_TAPES: List[Self] = []

    def __init__(self, persistent: Optional[bool] = False) -> None:
        self.operations: List[Dict[str, object]] = []
        self.watched: set[int] = set()  # Use set for faster membership
        self.persistent = persistent

    def __enter__(self) -> Self:
        self.operations.clear()
        GradientTape._GRADIENTS_TAPES.append(self)
        return self

    def __exit__(self, *args: Tuple[object], **kwargs: Dict[str, object]) -> None:
        if not self.persistent:
            GradientTape._GRADIENTS_TAPES.remove(self)

    def watch(self, *objs: Tuple[np.typing.NDArray[Any]]) -> None:
        for o in objs:
            if not isinstance(o, (np.ndarray, np.generic)):
                warnings.warn(f"Cannot watch object of type {type(o)}. Only numpy arrays and numbers are supported.")
                continue
            self._watch(o)

    def _watch(self, obj: np.typing.NDArray[Any]) -> None:
        self.watched.add(id(obj))

    def record(self, func: Callable, method: str, inputs: Tuple[np.typing.NDArray[Any]], kwargs: Dict[str, np.typing.NDArray[Any]], result: np.typing.NDArray[Any]) -> None:
        # If none of the inputs are watched, skip recording
        if all(id(i) not in self.watched for i in inputs):
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
        if isinstance(target, (np.ndarray, np.generic)):
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

    def _gradient(self, target: np.typing.NDArray[Any], source: np.typing.NDArray[Any]) -> Union[np.typing.NDArray[Any], None]:
        if id(target) not in self.watched:
            return None

        target_dtype = np.result_type(target, np.complex64)

        # Object id to index mapping for grads arrays
        obj_ids = [id(target)]
        grads_h = [np.ones_like(target, dtype=target_dtype)]
        grads_ah = [np.zeros_like(target, dtype=target_dtype)]
        id_to_idx = {id(target): 0}

        # Traverse operations in reverse for backprop
        for op in reversed(self.operations):
            func = op['func']
            inputs = op['inputs']
            result = op['result']
            kwargs = op['kwargs']

            if id(result) not in id_to_idx:
                continue  # No gradient to propagate from this result

            idx_res = id_to_idx[id(result)]
            grad_res = (grads_h[idx_res], grads_ah[idx_res])

            # Get gradient function from GRADIENTS dictionary
            dfunc = GRADIENTS.get(func.__name__)
            if not dfunc:
                if aliases := getattr(func, '__aliases__', None):
                    for alias in aliases:
                        dfunc = GRADIENTS.get(alias)
                        if dfunc:
                            break

            if dfunc:
                try:
                    raw = dfunc(grad_res, inputs, **kwargs)
                except TypeError:
                    raw = dfunc(grad_res, inputs)
            else:
                warnings.warn(f"No matches for gradient function '{func.__name__}' found. Using numerical approximation.")
                raw = numerical_derivative(func, inputs, kwargs)

            for inp, g in zip(inputs, raw):
                # g can be tuple (holo, anti) or just holo
                g_h, g_ah = g if isinstance(g, tuple) else (g, np.zeros_like(g))

                if id(inp) not in id_to_idx:
                    idx = len(obj_ids)
                    id_to_idx[id(inp)] = idx
                    obj_ids.append(id(inp))
                    grads_h.append(np.zeros_like(g_h))
                    grads_ah.append(np.zeros_like(g_ah))
                else:
                    idx = id_to_idx[id(inp)]

                grads_h[idx] += g_h
                grads_ah[idx] += g_ah

        # Return gradient for source
        if id(source) not in id_to_idx:
            return None

        idx_source = id_to_idx[id(source)]
        holo = grads_h[idx_source]
        anti = grads_ah[idx_source]

        if np.isrealobj(source):
            return (np.real(holo + np.conj(anti)),)
        return (holo, anti)

    def reset(self, watched: bool = True) -> None:
        self.operations.clear()
        if watched:
            self.watched.clear()

    def delete(self) -> None:
        self.__del__()

    def __del__(self):
        if self in GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES.remove(self)
