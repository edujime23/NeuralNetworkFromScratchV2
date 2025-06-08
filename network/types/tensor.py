import contextlib
import sys
import warnings
from typing import Self

import numpy as np
from ..queues import tapes

# Keep a reference to the original warnings.showwarning
_ORIGINAL_SHOWWARNING = warnings.showwarning


class Tensor:
    """
    A “tf.Tensor-like” wrapper around a read-only NumPy ndarray. Externally, it provides:
      - attributes:   .shape, .dtype, .ndim, .size, .name, .device, .T, .graph (None)
      - conversion:   .numpy(), .eval()
      - indexing:    t[i], t[:, 2], etc.
      - Python scalar:  t.item()
      - Truth-value:   bool(t) only if t.size == 1
      - Arithmetic & ufuncs:  x + y, x * 3, x**2, np.sqrt(x), np.mean(x), etc.
      - Reductions & reshaping:  t.sum(axis=...), t.mean(...), t.reshape(...), etc.
      - Casting:  t.astype(np.float32)
      - Additional “tf-like” methods:  clip_by_value, clip_by_norm, get_shape, rank, eval
      - If you call any NumPy ufunc or function on a Tensor, it returns a new Tensor
        and (in a real TF scenario) would call GradientTape.record(...) so that
        autodiff “sees” every primitive. Here, we simulate the recording hooks.
    """

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

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Support np.asarray(tensor) or np.array(tensor). If dtype is given,
        cast accordingly.
        """
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

    @staticmethod
    def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
        frame = sys._getframe()
        skip_keywords = (
            "numpy",
            "site-packages",
            "warnings.py",
            "<frozen ",
            ".venv",
            __file__,
        )
        while frame:
            fname = frame.f_code.co_filename
            if all(kw not in fname for kw in skip_keywords):
                break
            frame = frame.f_back
        target_file = frame.f_code.co_filename if frame else filename
        target_lineno = frame.f_lineno if frame else lineno
        _ORIGINAL_SHOWWARNING(message, category, target_file, target_lineno, file, line)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Build a parallel tuple of just the Tensor‐objects:
        tensor_inputs = tuple(
            inp if isinstance(inp, Tensor) else Tensor(inp)
            for inp in inputs
        )

        # Extract raw NumPy arrays for the actual ufunc call:
        np_inputs = tuple(
            inp.data if isinstance(inp, Tensor) else inp
            for inp in inputs
        )
        np_kwargs = {
            k: (v.data if isinstance(v, Tensor) else v)
            for k, v in kwargs.items()
        }

        warnings.showwarning = self._custom_showwarning
        try:
            result_array = getattr(ufunc, method)(*np_inputs, **np_kwargs)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING

        # Wrap the NumPy result into a fresh Tensor:
        if isinstance(result_array, np.ndarray):
            out = Tensor(result_array, dtype=result_array.dtype, name=None)
        else:
            out = Tensor(np.asarray(result_array), name=None)

        if tapes:
            tapes[-1].record(
                ufunc,
                method,
                tensor_inputs,
                kwargs,
                out,
            )

        return out

    def __array_function__(self, func, types, inputs, kwargs):
        # Normalize the inputs into Tensors:
        tensor_inputs = tuple(
            inp if isinstance(inp, Tensor) else Tensor(inp)
            for inp in inputs
        )
        np_inputs = tuple(
            inp.data if isinstance(inp, Tensor) else inp
            for inp in inputs
        )
        np_kwargs = {
            k: (v.data if isinstance(v, Tensor) else v)
            for k, v in kwargs.items()
        }

        warnings.showwarning = self._custom_showwarning
        try:
            result_array = func(*np_inputs, **np_kwargs)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING

        if isinstance(result_array, np.ndarray):
            out = Tensor(result_array, dtype=result_array.dtype, name=None)
        else:
            out = Tensor(np.asarray(result_array), name=None)

        if tapes:
            tapes[-1].record(
                func,
                "__call__",
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
                tapes[-1].record(
                    np.ndarray.__getitem__, "__call__", (self,), {}, out
                )
        return out

    def __setitem__(self, idx, value):
        raise NotImplementedError(
            "In-place assignment on a Tensor is not supported for autodiff."
        )

    # Shape manipulation
    def reshape(self, *shape: int) -> Self:
        arr = self.data.reshape(*shape)
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1].record(
                    np.ndarray.reshape, "__call__", (self,), {"newshape": shape}, out
                )
        return out

    def transpose(self, *axes: int) -> Self:
        arr = self.data.transpose(*axes)
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1].record(
                    np.ndarray.transpose,
                    "__call__",
                    (self,),
                    {"axes": axes or None},
                    out,
                )
        return out

    def flatten(self) -> Self:
        """
        Return a copy of the tensor collapsed into one dimension.
        """
        # Use NumPy’s flatten to get a 1-D array
        arr = self.data.flatten()
        out = Tensor(arr, dtype=arr.dtype, name=self.name)
        # Record for autodiff if a GradientTape is active
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1].record(
                    np.ndarray.flatten, "__call__", (self,), {}, out
                )
        return out

    # Reductions
    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        warnings.showwarning = self._custom_showwarning
        try:
            arr = self.data.sum(axis=axis, keepdims=keepdims)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1].record(
                    np.ndarray.sum,
                    "__call__",
                    (self,),
                    {"axis": axis, "keepdims": keepdims},
                    out,
                )
        return out

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Self:
        warnings.showwarning = self._custom_showwarning
        try:
            arr = self.data.mean(axis=axis, keepdims=keepdims)
        finally:
            warnings.showwarning = _ORIGINAL_SHOWWARNING
        out = Tensor(arr, dtype=arr.dtype, name=None)
        with contextlib.suppress(ImportError):
            if tapes:
                tapes[-1].record(
                    np.ndarray.mean,
                    "__call__",
                    (self,),
                    {"axis": axis, "keepdims": keepdims},
                    out,
                )
        return out

    def astype(self, dtype: np.dtype | str) -> Self:
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
        arr = np.linalg.norm(self.data, ord=ord, axis=axis, keepdims=keepdims)
        # Wrap scalar/array result into Tensor
        return Tensor(arr, dtype=arr.dtype, name=None)

    # Additional tf.Tensor-like methods
    def clip_by_value(self, clip_value_min: float, clip_value_max: float) -> Self:
        """
        Clip (limit) the values in the tensor.
        """
        arr = np.clip(self.data, clip_value_min, clip_value_max)
        return Tensor(arr, dtype=arr.dtype, name=self.name)

    def clip_by_norm(
        self, clip_norm: float, axes: None | int | tuple[int, ...] = None
    ) -> Self:
        """
        Clip tensor values by the given L2-norm.
        """
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
