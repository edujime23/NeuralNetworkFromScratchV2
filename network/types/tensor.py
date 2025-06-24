import contextlib
from opcode import hasarg
import os
import sys
import warnings
from typing import Self

import numpy as np

from ..queues import tapes

# Keep a reference to the original warnings.showwarning
_ORIGINAL_SHOWWARNING = warnings.showwarning


class Tensor:
    __slots__ = ("__data", "_name", "_dtype", "_shape")

    def __init__(
        self,
        value: np.ndarray | float | int | complex | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | str | None = None,
        name: str | None = None,
    ):
        if value is None:
            arr = np.zeros(()) if shape is None else np.zeros(shape, dtype=dtype)
        else:
            arr = np.asarray(value, dtype=dtype)
            if shape is not None and arr.shape != shape:
                try:
                    arr = arr.reshape(shape)
                except ValueError as e:
                    raise ValueError(f"Shape mismatch: got {arr.shape}, expected {shape}") from e

        self.__data = arr.copy()
        self.__data.setflags(write=False)

        self._name = name
        self._dtype = self.__data.dtype
        self._shape = self.__data.shape

    @property
    def name(self) -> str | None:
        """Return the user-provided name (or None)."""
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape as a tuple of ints."""
        return self._shape

    def get_shape(self) -> tuple[int, ...]:
        """Alias for .shape (tf.Tensor-like)."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Return the NumPy dtype."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions (rank)."""
        return len(self._shape)

    def rank(self) -> int:
        """Alias for ndim (tf.Tensor-like)."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.__data.size

    @property
    def device(self) -> str:
        """Always 'CPU' (NumPy-based)."""
        return "CPU"

    @property
    def data(self) -> np.ndarray:
        """Internal read-only NumPy array."""
        return self.__data

    @property
    def T(self) -> Self:
        """Shorthand for transpose()."""
        return self.transpose()

    @property
    def real(self):
        return self.data.real

    @property
    def imag(self):
        return self.data.imag

    @property
    def graph(self):
        """tf.Tensor exposes .graph; here, always None."""
        return None

    @property
    def numpy(self) -> np.ndarray:
        """Return a writable copy of the underlying array (like tf.Tensor.numpy())."""
        return self.data.copy()

    def eval(self) -> np.ndarray:
        """Alias for .numpy() (tf 2.x eager mode compatibility)."""
        return self.numpy()

    def dot(self, b: Self, out=None) -> Self:
        return Tensor(self.data.dot(b=b, out=out))

    @staticmethod
    def _find_user_frame():
        """Find the first frame that's in user code (not in numpy, site-packages, etc.)"""
        frame = sys._getframe()

        # Get the directory containing tensor.py to exclude all internal types
        types_dir = os.path.dirname(os.path.abspath(__file__))

        skip_patterns = (
            "numpy",
            "site-packages",
            "warnings.py",
            "<frozen ",
            "__pycache__",
        )

        while frame:
            filename = frame.f_code.co_filename
            abs_filename = os.path.abspath(filename)

            # Skip all files in the types directory (tensor.py, variable.py, etc.)
            if os.path.dirname(abs_filename) == types_dir:
                frame = frame.f_back
                continue

            if any(pattern in filename for pattern in skip_patterns):
                frame = frame.f_back
                continue

            # Found user code frame
            return filename, frame.f_lineno

        # Fallback - return None to use original warning location
        return None, None

    @staticmethod
    def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
        """Custom warning handler that shows warnings at the user code location"""
        user_filename, user_lineno = Tensor._find_user_frame()

        if user_filename and user_lineno:
            _ORIGINAL_SHOWWARNING(message, category, user_filename, user_lineno, file, line)
        else:
            # Fallback to original behavior
            _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file, line)

    @contextlib.contextmanager
    def _warning_context(self):
        """Context manager to temporarily replace warning handler"""
        original = warnings.showwarning
        warnings.showwarning = self._custom_showwarning
        try:
            yield
        finally:
            warnings.showwarning = original

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Support np.asarray(tensor) or np.array(tensor). If dtype is given,
        cast accordingly.
        """
        with self._warning_context():
            return self.data if dtype is None else self.data.astype(dtype)

    def item(self) -> int | float | complex:
        """
        If this Tensor holds exactly one element, return it as a Python scalar.
        Otherwise, raise an error.
        """
        if self.size != 1:
            raise ValueError("Can only call .item() on a scalar Tensor.")
        return self.data.item()

    def __bool__(self):
        """
        The truth value of a Tensor is only valid if it has exactly one element.
        """
        if self.size != 1:
            raise ValueError(
                "The truth value of a Tensor with more than one element is ambiguous."
            )
        return bool(self.data)

    def __int__(self):
        if self.size != 1:
            raise ValueError("Can only convert a scalar Tensor to int.")
        return int(self.data)

    def __float__(self):
        if self.size != 1:
            raise ValueError("Can only convert a scalar Tensor to float.")
        return np.float64(self.data)

    def __complex__(self):
        if self.size != 1:
            raise ValueError("Can only convert a scalar Tensor to complex.")
        return np.complex128(self.data)

    def __repr__(self) -> str:
        return f"Tensor{repr(self.data).removeprefix("array")}"

    def __format__(self, format_spec: str) -> str:
        if self.ndim == 0:
            return format(self.data.item(), format_spec)

        return self.__repr__()

    def __array__(self, dtype: np.dtype = None):
        return self.__data.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        tensor_inputs = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                tensor_inputs.append(inp)
            elif hasattr(inp, 'value'):
                tensor_inputs.append(inp.value)
            else:
                tensor_inputs.append(Tensor(inp))

        tensor_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                tensor_kwargs[k] = v
            elif hasattr(v, 'value'):
                tensor_kwargs[k] = v.value
            else:
                tensor_kwargs[k] = Tensor(v)

        # Extract raw NumPy arrays for the actual ufunc call:
        np_inputs = tuple(
            inp.data
            for inp in tensor_inputs
        )
        np_kwargs = {
            k: v.data
            for k, v in tensor_kwargs.items()
        }

        # print(inputs, ufunc.__name__)


        out = Tensor(self.__data.__array_ufunc__(ufunc, method, *np_inputs, **np_kwargs))

        if tapes:
            tapes[-1]._record_operation(
                ufunc.__name__,
                tensor_inputs,
                kwargs,
                out,
            )

        return out

    def __array_function__(self, func, types, inputs, kwargs):
        tensor_inputs = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                tensor_inputs.append(inp)
            elif hasattr(inp, 'value'):
                tensor_inputs.append(inp.value)
            elif isinstance(inp, np.ndarray):
                tensor_inputs.append(Tensor(inp))
            else:
                tensor_inputs.append(inp)

        tensor_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                tensor_kwargs[k] = v
            elif hasattr(v, 'value'):
                tensor_kwargs[k] = v.value
            elif isinstance(v, np.ndarray):
                tensor_kwargs[k] = Tensor(v)
            else:
                tensor_kwargs[k] = v

        # Extract raw NumPy arrays for the actual ufunc call:
        np_inputs = tuple(
            inp.data if hasattr(inp, 'data') else inp
            for inp in tensor_inputs
        )
        np_kwargs = {
            k: v.data if hasattr(v, 'data') else v
            for k, v in tensor_kwargs.items()
        }

        out = Tensor(func(*np_inputs, **np_kwargs))

        if tapes:
            tapes[-1]._record_operation(
                func.__name__,
                tensor_inputs,
                kwargs,
                out,
            )

        return out

    # Arithmetic operators returning Tensor
    def __add__(self, other) -> Self:
        return np.add(self, other)

    def __sub__(self, other) -> Self:
        return np.subtract(self, other)

    def __mul__(self, other) -> Self:
        return np.multiply(self, other)

    def __truediv__(self, other) -> Self:
        return np.divide(self, other)

    def __floordiv__(self, other) -> Self:
        return np.floor_divide(self, other)

    def __mod__(self, other) -> Self:
        return np.mod(self, other)

    def __pow__(self, other) -> Self:
        return np.power(self, other)

    def __matmul__(self, other) -> Self:
        return np.matmul(self, other)

    def __radd__(self, other) -> Self:
        return np.add(other, self)

    def __rsub__(self, other) -> Self:
        return np.subtract(other, self)

    def __rmul__(self, other) -> Self:
        return np.multiply(other, self)

    def __rtruediv__(self, other) -> Self:
        return np.divide(other, self)

    def __rfloordiv__(self, other) -> Self:
        return np.floor_divide(other, self)

    def __rmod__(self, other) -> Self:
        return np.mod(other, self)

    def __rpow__(self, other) -> Self:
        return np.power(other, self)

    def __rmatmul__(self, other) -> Self:
        return np.matmul(other, self)

    def __neg__(self) -> Self:
        return np.negative(self)

    def __pos__(self) -> Self:
        return self

    # Comparison ops
    def __lt__(self, other) -> Self:
        return np.less(self, other)

    def __le__(self, other) -> Self:
        return np.less_equal(self, other)

    def __eq__(self, other) -> Self:
        return np.equal(self, other)

    def __ne__(self, other) -> Self:
        return np.not_equal(self, other)

    def __gt__(self, other) -> Self:
        return np.greater(self, other)

    def __ge__(self, other) -> Self:
        return np.greater_equal(self, other)

    # Indexing / slicing
    def __getitem__(self, idx) -> Self:
        sliced = self.data[idx]
        out = Tensor(sliced, dtype=sliced.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1]._record_operation(
                    "__getitem__", (self,), {}, out
                )
        return out

    def __setitem__(self, idx, value):
        raise NotImplementedError(
            "In-place assignment on a Tensor is not supported for autodiff."
        )

    # Shape manipulation
    def reshape(self, *shape: int) -> Self:
        with self._warning_context():
            arr = self.data.reshape(*shape)
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1]._record_operation(
                    "reshape", (self,), {"newshape": shape}, out
                )
        return out

    def transpose(self, *axes: int) -> Self:
        with self._warning_context():
            arr = self.data.transpose(*axes)
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1]._record_operation(
                    "transpose",
                    (self,),
                    {"axes": axes or None},
                    out,
                )
        return out

    def flatten(self) -> Self:
        """
        Return a copy of the tensor collapsed into one dimension.
        """
        with self._warning_context():
            arr = self.data.flatten()
        out = Tensor(arr, dtype=arr.dtype, name=self.name)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1]._record_operation(
                    "flatten", (self,), {}, out
                )
        return out

    def squeeze(self):
        with self._warning_context():
            arr = self.data.squeeze()
        out = Tensor(arr, dtype=arr.dtype, name=self.name, shape=arr.shape)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1]._record_operation(
                    "squeeze", (self,), {}, out
                    )
        return out

    # Reductions
    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        with self._warning_context():
            arr = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1]._record_operation(
                    "sum",
                    (self,),
                    {"axis": axis, "keepdims": keepdims},
                    out,
                )
        return out

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        with self._warning_context():
            arr = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1]._record_operation(
                    "mean",
                    (self,),
                    {"axis": axis, "keepdims": keepdims},
                    out,
                )
        return out

    def astype(self, dtype: np.dtype | str) -> Self:
        with self._warning_context():
            arr = self.data.astype(dtype)
        return Tensor(arr, dtype=arr.dtype, name=self.name)

    # Logical checks
    def all(self) -> bool:
        return np.all(self.data)

    def any(self) -> bool:
        return np.any(self.data)

    # Norm (vector/matrix/tensor)
    def norm(
        self,
        ord: None | int | float | str = None,
        axis: None | int | tuple[int, ...] = None,
        keepdims: bool = False,
    ) -> Self:
        with self._warning_context():
            arr = np.linalg.norm(self.data, ord=ord, axis=axis, keepdims=keepdims)
        return Tensor(arr, dtype=arr.dtype, name=None)

    # Additional tf.Tensor-like methods
    def clip_by_value(self, clip_value_min: float, clip_value_max: float) -> Self:
        """
        Clip (limit) the values in the tensor.
        """
        with self._warning_context():
            arr = np.clip(self.data, clip_value_min, clip_value_max)
        return Tensor(arr, dtype=arr.dtype, name=self.name)

    def clip_by_norm(
        self, clip_norm: float, axes: None | int | tuple[int, ...] = None
    ) -> Self:
        """
        Clip tensor values by the given L2-norm.
        """
        with self._warning_context():
            if axes is None:
                total_norm = np.linalg.norm(self.data)
                if total_norm > clip_norm:
                    arr = self.data * (clip_norm / total_norm)
                else:
                    arr = self.data
            else:
                # Compute norms along specified axes and broadcast
                norms = np.linalg.norm(self.data, axis=axes, keepdims=True)
                factor = clip_norm / np.maximum(norms, clip_norm)
                arr = self.data * factor
        return Tensor(arr, dtype=arr.dtype, name=self.name)

    def __len__(self) -> int:
        """
        Return the length along the first axis, or 1 if scalar.
        """
        return 1 if self._shape == () else self._shape[0]

    def __iter__(self):
        """
        Iterate over the first axis, yielding Tensor slices.
        """
        if self._shape == ():
            yield self
        else:
            for idx in range(self._shape[0]):
                yield Tensor(self.data[idx], dtype=self._dtype, name=None)