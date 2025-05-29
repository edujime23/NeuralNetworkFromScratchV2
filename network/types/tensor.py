import numpy as np
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Self, Tuple, Union

# Keep a reference to the original warnings.showwarning
_ORIGINAL_SHOWWARNING = warnings.showwarning


class Tensor:
    """
    A “tf.Tensor‐like” wrapper around a read‐only NumPy ndarray.  Externally,
    it provides:
      - attributes:   .shape, .dtype, .ndim, .size, .name, .device, .T
      - conversion:   .numpy()
      - indexing:    t[i], t[:,2], etc.
      - Python scalar:  t.item()
      - Truth‐value:   bool(t) only if t.size == 1
      - Arithmetic & ufuncs:  x + y, x * 3, x**2, np.sqrt(x), np.mean(x), etc.
      - Reductions & reshaping:  t.sum(axis=...), t.mean(...), t.reshape(...), etc.
      - Casting:  t.astype(np.float32)
      - If you call any NumPy ufunc or function on a Tensor, it returns a new Tensor
        and calls GradientTape.record(...) so that autodiff “sees” every primitive.
    """

    __slots__ = ("data", "_name", "_dtype", "_shape")

    def __init__(
        self,
        value: Optional[Union[np.ndarray, float, int, complex]] = None,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[np.dtype] = None,
        name: Optional[str] = None
    ):

        arr = np.asarray(value, dtype=dtype)
        if shape is not None and arr.shape != shape:
            raise ValueError(f"Shape mismatch: got {arr.shape}, expected {shape}")


        self.data = arr.copy()
        self.data.setflags(write=False)


        self._name = name
        self._dtype = self.data.dtype
        self._shape = self.data.shape


    @property
    def name(self) -> Optional[str]:
        """Return the user‐provided name (or None)."""
        return self._name

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def device(self) -> str:
        """Always “CPU” (NumPy‐based)."""
        return "CPU"

    @property
    def T(self) -> "Tensor":
        """Shorthand for transpose()."""
        return self.transpose()


    def numpy(self) -> np.ndarray:
        """Return a writable copy of the underlying array (like tf.Tensor.numpy())."""
        return self.data.copy()

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Support np.asarray(tensor) or np.array(tensor).  If dtype is given,
        cast accordingly.
        """
        return self.data if dtype is None else self.data.astype(dtype)

    def item(self) -> Union[int, float, complex]:
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
            raise ValueError("The truth value of a Tensor with more than one element is ambiguous.")
        return bool(self.data)

    def __int__(self):
        if self.size != 1:
            raise ValueError("Can only convert a scalar Tensor to int.")
        return int(self.data)

    def __float__(self):
        if self.size != 1:
            raise ValueError("Can only convert a scalar Tensor to float.")
        return float(self.data)

    def __complex__(self):
        if self.size != 1:
            raise ValueError("Can only convert a scalar Tensor to complex.")
        return complex(self.data)

    def __repr__(self) -> str:
        return (
            f"Tensor(name={self._name}, shape={self._shape}, "
            f"dtype={self._dtype}, data=\n{self.data})"
        )


    @staticmethod
    def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
        frame = sys._getframe()
        skip_keywords = ("numpy", "site-packages", "warnings.py", "<frozen ", ".venv", __file__)
        while frame:
            fname = frame.f_code.co_filename
            if np.all(kw not in fname for kw in skip_keywords):
                break
            frame = frame.f_back
        target_file = frame.f_code.co_filename if frame else filename
        target_lineno = frame.f_lineno if frame else lineno
        _ORIGINAL_SHOWWARNING(message, category, target_file, target_lineno, file, line)


    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        np_inputs = tuple(inp.data if isinstance(inp, Tensor) else inp for inp in inputs)
        np_kwargs = {k: (v.data if isinstance(v, Tensor) else v) for k, v in kwargs.items()}


        warnings.showwarning = self._custom_showwarning
        try:
            result_array = getattr(ufunc, method)(*np_inputs, **np_kwargs)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING


        if isinstance(result_array, np.ndarray):
            out = Tensor(result_array, dtype=result_array.dtype, name=None)
        else:

            out = Tensor(np.asarray(result_array), name=None)

        from ..tape import GradientTape
        if GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES[-1].record(ufunc, method, inputs, kwargs, out)

        return out

    def __array_function__(self, func, types, inputs, kwargs):

        np_inputs = tuple(inp.data if isinstance(inp, Tensor) else inp for inp in inputs)
        np_kwargs = {k: (v.data if isinstance(v, Tensor) else v) for k, v in kwargs.items()}


        warnings.showwarning = self._custom_showwarning
        try:
            result_array = func(*np_inputs, **np_kwargs)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING


        if isinstance(result_array, np.ndarray):
            out = Tensor(result_array, dtype=result_array.dtype, name=None)
        else:
            out = Tensor(np.asarray(result_array), name=None)

        from ..tape import GradientTape
        if GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES[-1].record(func, "__call__", inputs, kwargs, out)

        return out


    def __add__(self, other) -> "Tensor":
        return np.add(self, other)

    def __radd__(self, other) -> "Tensor":
        return np.add(other, self)

    def __sub__(self, other) -> "Tensor":
        return np.subtract(self, other)

    def __rsub__(self, other) -> "Tensor":
        return np.subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return np.multiply(self, other)

    def __rmul__(self, other) -> "Tensor":
        return np.multiply(other, self)

    def __truediv__(self, other) -> "Tensor":
        return np.divide(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        return np.divide(other, self)

    def __floordiv__(self, other) -> "Tensor":
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other) -> "Tensor":
        return np.floor_divide(other, self)

    def __mod__(self, other) -> "Tensor":
        return np.mod(self, other)

    def __rmod__(self, other) -> "Tensor":
        return np.mod(other, self)

    def __pow__(self, other) -> "Tensor":
        return np.power(self, other)

    def __matmul__(self, other) -> "Tensor":
        return np.matmul(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        return np.matmul(other, self)

    def __neg__(self) -> "Tensor":
        return np.negative(self)

    def __pos__(self) -> "Tensor":
        return self

    def __lt__(self, other) -> "Tensor":
        return np.less(self, other)

    def __le__(self, other) -> "Tensor":
        return np.less_equal(self, other)

    def __eq__(self, other) -> "Tensor":
        return np.equal(self, other)

    def __ne__(self, other) -> "Tensor":
        return np.not_equal(self, other)

    def __gt__(self, other) -> "Tensor":
        return np.greater(self, other)

    def __ge__(self, other) -> "Tensor":
        return np.greater_equal(self, other)


    def __getitem__(self, idx) -> "Tensor":
        sliced = self.data[idx]
        out = Tensor(sliced, dtype=sliced.dtype, name=None)
        from ..tape import GradientTape
        if GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES[-1].record(
                np.ndarray.__getitem__, "__call__", (self,), {}, out
            )
        return out

    def __setitem__(self, idx, value):
        raise NotImplementedError("In-place assignment on a Tensor is not supported for autodiff.")


    def reshape(self, *shape) -> "Tensor":
        """
        Return a new Tensor with the specified shape.
        Internally this is exactly `self.data.reshape(shape)` → Tensor, and is recorded.
        """
        arr = self.data.reshape(*shape)
        out = Tensor(arr, dtype=arr.dtype, name=None)
        from ..tape import GradientTape
        if GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES[-1].record(
                np.ndarray.reshape, "__call__", (self,), {"newshape": shape}, out
            )
        return out

    def transpose(self, *axes) -> "Tensor":
        """
        Return a transposed view of the tensor (axes reversed if none specified).
        """
        arr = self.data.transpose(*axes)  # built-in np.ndarray.transpose
        out = Tensor(arr, dtype=arr.dtype, name=None)
        from ..tape import GradientTape
        if GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES[-1].record(
                np.ndarray.transpose, "__call__", (self,), {"axes": axes or None}, out
            )
        return out

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """
        Compute the sum over the specified axes, returning a new Tensor.
        """
        warnings.showwarning = self._custom_showwarning
        try:
            arr = self.data.sum(axis=axis, keepdims=keepdims)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING

        out = Tensor(arr, dtype=arr.dtype, name=None)
        from ..tape import GradientTape
        if GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES[-1].record(
                np.ndarray.sum, "__call__", (self,), {"axis": axis, "keepdims": keepdims}, out
            )
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """
        Compute the arithmetic mean over the specified axes, returning a new Tensor.
        """
        warnings.showwarning = self._custom_showwarning
        try:
            arr = self.data.mean(axis=axis, keepdims=keepdims)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING

        out = Tensor(arr, dtype=arr.dtype, name=None)
        from ..tape import GradientTape
        if GradientTape._GRADIENTS_TAPES:
            GradientTape._GRADIENTS_TAPES[-1].record(
                np.ndarray.mean, "__call__", (self,), {"axis": axis, "keepdims": keepdims}, out
            )
        return out
    
    def view(self, dtype: Optional[np.dtype] = None) -> Self:
        return self.data.view(dtype=dtype)

    def astype(self, dtype: Union[np.dtype, str]) -> "Tensor":
        """
        Cast to a given type, e.g. x.astype(np.float32) or x.astype('int64').
        """
        arr = self.data.astype(dtype)
        return Tensor(arr, dtype=arr.dtype, name=self.name)