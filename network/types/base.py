# network/types/base.py

import numpy as np
from typing import Tuple, Union
from ..gradientTape import Tape

class BaseType(np.ndarray):
    def __new__(cls, value: Union[np.typing.NDArray, np.number], shape: Tuple[int], dtype: np.typing.DTypeLike, name: str):
        obj = super().__new__(cls, shape, dtype)
        obj.__name = name
        obj[:] = value[:] if value else np.zeros(shape=shape)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 1) unwrap inputs to raw numpy arrays
        raw_inputs = [
            x.view(np.ndarray) if isinstance(x, BaseType) else x
            for x in inputs
        ]
        # 2) perform the actual numpy operation
        raw_out = getattr(ufunc, method)(*raw_inputs, **kwargs)

        # 3) wrap back into BaseType if array
        if isinstance(raw_out, np.ndarray):
            out = raw_out.view(BaseType)
        else:
            # scalar result, nothing to tape
            return raw_out

        # 4) record the op against the wrapped output
        if Tape._CURRENT_TAPE is not None:
            Tape._CURRENT_TAPE.record(ufunc, method, inputs, kwargs, out)

        return out

    @staticmethod
    def _initialize(arr: np.typing.NDArray, shape: Tuple[int], initializer: str):
        match initializer:
            case 'zeros':
                arr[:] = 0
            case 'ones':
                arr[:] = 1
            case 'random_normal':
                arr[:] = np.random.normal(0, 1, shape)
            case 'random_uniform':
                arr[:] = np.random.uniform(0, 1, shape)
            case 'xavier_uniform':
                limit = np.sqrt(6 / np.sum(shape))
                arr[:] = np.random.uniform(-limit, limit, shape)
            case 'xavier_normal':
                stddev = np.sqrt(2 / np.sum(shape))
                arr[:] = np.random.normal(0, stddev, shape)
            case 'he_normal':
                stddev = np.sqrt(2 / shape[0])
                arr[:] = np.random.normal(0, stddev, shape)
            case 'he_uniform':
                limit = np.sqrt(6 / shape[0])
                arr[:] = np.random.uniform(-limit, limit, shape)
            case 'glorot_normal':
                stddev = np.sqrt(2 / np.sum(shape))
                arr[:] = np.random.normal(0, stddev, shape)
            case 'glorot_uniform':
                limit = np.sqrt(6 / np.sum(shape))
                arr[:] = np.random.uniform(-limit, limit, shape)
            case 'lecun_normal':
                stddev = np.sqrt(1 / shape[0])
                arr[:] = np.random.normal(0, stddev, shape)
            case 'lecun_uniform':
                limit = np.sqrt(3 / shape[0])
                arr[:] = np.random.uniform(-limit, limit, shape)

    @property
    def name(self):
        return self.__name