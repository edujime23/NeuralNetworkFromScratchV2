import numpy as np
from typing import Callable, Tuple, Dict, Any, List, Union, Optional

def numerical_derivative(
    func: Callable[..., np.ndarray],
    inputs: Tuple[np.ndarray, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    batch_func: Optional[Callable[[List[Tuple[int, np.ndarray]]], List[np.ndarray]]] = None,
    *,
    verbose: bool = False,
    return_real_if_input_real: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Wirtinger derivatives of `func` w.r.t. each array in `inputs`.

    Supports:
    - Complex-step differentiation for purely real-valued inputs (machine-precision).
    - Sixth-order finite difference for complex-valued inputs.

    Parameters:
        func: Callable that returns np.ndarray
        inputs: Tuple of np.ndarray to differentiate with respect to
        kwargs: Additional keyword arguments for func (default empty dict)
        batch_func: Optional batching version of func for evaluating many inputs at once (currently unused)
        verbose: If True, print debug info
        return_real_if_input_real: If True and input is real, returns gradients as real arrays.

    Returns:
        List of (∂f/∂z, ∂f/∂z̄) pairs, one for each input array.
    """
    if kwargs is None:
        kwargs = {}

    assert isinstance(inputs, tuple), "inputs must be a tuple of np.ndarray"
    for i, x in enumerate(inputs):
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Input {i} is not a np.ndarray")
        if not np.issubdtype(x.dtype, np.floating) and not np.issubdtype(x.dtype, np.complexfloating):
            raise TypeError(f"Input {i} dtype {x.dtype} not float or complex float")

    f0 = func(*inputs, **kwargs)
    f0 = np.asarray(f0)
    f0_shape = f0.shape

    # Cache keyed by (idx, hash of bytes) for efficiency
    cache: Dict[Tuple[int, int], np.ndarray] = {}
    grads: List[Tuple[np.ndarray, np.ndarray]] = []
    base_inputs = list(inputs)

    def replace_input(arr_list: List[np.ndarray], idx: int, new_arr: np.ndarray) -> None:
        arr_list[idx] = new_arr

    def arr_hash(arr: np.ndarray) -> int:
        # Use Python's built-in hash of bytes, which is faster and less memory heavy than storing bytes
        # Note: hash is not guaranteed unique, but collisions are very unlikely for this use case.
        return hash(arr.tobytes())

    def call_func_with_cache(idx: int, x_new: np.ndarray) -> np.ndarray:
        key = (idx, arr_hash(x_new))
        if key not in cache:
            if verbose:
                print(f"Calling func for input {idx} with perturbed array (hash {key[1]})")
            replace_input(base_inputs, idx, x_new)
            fx = func(*base_inputs, **kwargs)
            fx = np.asarray(fx)
            if fx.shape != f0_shape:
                raise ValueError(
                    f"Shape mismatch when calling func for input {idx}: got {fx.shape}, expected {f0_shape}"
                )
            cache[key] = fx
            replace_input(base_inputs, idx, inputs[idx])
        return cache[key]

    def tiny_eps_for(arr: np.ndarray) -> float:
        eps_machine = np.finfo(arr.dtype).eps
        norm_val = np.linalg.norm(arr)
        norm_val = norm_val if norm_val > 1e-12 else 1.0  # Avoid too small norm
        eps = np.cbrt(eps_machine) * norm_val
        if verbose:
            print(f"tiny_eps_for: dtype={arr.dtype}, eps={eps}, norm={norm_val}")
        return eps

    for idx, x in enumerate(inputs):
        grad_z = np.zeros_like(x, dtype=np.complex128)
        grad_conj_z = np.zeros_like(x, dtype=np.complex128)
        is_complex = np.iscomplexobj(x)

        x_flat = x.ravel()
        gz_flat = grad_z.ravel()
        gzb_flat = grad_conj_z.ravel()

        for i in range(x_flat.size):
            orig_val = x_flat[i]

            if not is_complex:
                eps_c = tiny_eps_for(x.real)
                x_pert = x_flat.copy().astype(np.complex128)
                x_pert[i] = orig_val + 1j * eps_c
                f_pert = call_func_with_cache(idx, x_pert.reshape(x.shape))
                gz_flat[i] = np.sum(f_pert).imag / eps_c
                gzb_flat[i] = 0.0 + 0.0j
            else:
                a, b = orig_val.real, orig_val.imag
                eps_r = tiny_eps_for(np.array([a], dtype=np.float64))
                eps_i = tiny_eps_for(np.array([b], dtype=np.float64))

                def pert(v: float, kind: str = 'real') -> np.ndarray:
                    tmp = x_flat.copy().astype(np.complex128)
                    tmp[i] = (a + v) + 1j * b if kind == 'real' else a + 1j * (b + v)
                    return tmp.reshape(x.shape)

                shifts = [-3, -2, -1, 1, 2, 3]
                coeffs = np.array([-1, 9, -45, 45, -9, 1]) / 60.0

                # Real part finite difference
                f_r = [call_func_with_cache(idx, pert(k * eps_r, 'real')) for k in shifts]
                dfx = sum(c * f for c, f in zip(coeffs, f_r)) / eps_r
                dfx_sum = np.sum(dfx)

                # Imaginary part finite difference
                f_i = [call_func_with_cache(idx, pert(k * eps_i, 'imag')) for k in shifts]
                dfy = sum(c * f for c, f in zip(coeffs, f_i)) / eps_i
                dfy_sum = np.sum(dfy)

                gz_flat[i] = 0.5 * (dfx_sum - 1j * dfy_sum)
                gzb_flat[i] = 0.5 * (dfx_sum + 1j * dfy_sum)

        # Optionally convert gradients to real if input was real
        if return_real_if_input_real and not is_complex:
            grad_z = grad_z.real.astype(x.dtype)
            grad_conj_z = grad_conj_z.real.astype(x.dtype)

        grads.append((grad_z, grad_conj_z))

    return grads