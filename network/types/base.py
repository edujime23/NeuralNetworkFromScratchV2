import numpy as np
from typing import Tuple, Optional

class BaseType(np.ndarray):
    def __new__(cls, shape: Tuple[int], dtype: np.typing.DTypeLike, name: str):
        obj = super().__new__(cls, shape, dtype)
        obj.__name = name
        obj.fill(0)
        return obj
    
    # def __array_wrap__(self, obj, context=None):
    #     func, args, _ = context
    #     op_type = func.__name__

    #     recorder: Tape = Tape._active_tape

    #     new_tracked_array = BaseType(obj)

    #     # Record the operation if a tape is active
    #     if recorder:
    #         # Filter args to include only tracked or representative values
    #         inputs_for_record = tuple(a if isinstance(a, BaseType) else np.asarray(a) for a in args)
    #         recorder._record_operation(op_type, inputs_for_record, new_tracked_array)

    #     return new_tracked_array
    
    @staticmethod
    def _initialize(arr: np.typing.NDArray, shape: Tuple[int], initializer: str):
        match initializer:
            case 'zeros':
                print("zeros")
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