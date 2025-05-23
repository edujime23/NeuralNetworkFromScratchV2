import numpy as np
from typing import Callable, Tuple, Dict, Any, List
from .util import ensure_shape

def numerical_derivative(
    func: Callable,
    inputs: Tuple[np.ndarray, ...],
    kwargs: Dict[str, Any],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Wirtinger derivatives of func w.r.t. each array in inputs.
    - Uses complex‐step for purely real inputs (machine‐precision).
    - Uses sixth‐order finite differences for genuinely complex inputs.
    """
    grads: List[Tuple[np.ndarray, np.ndarray]] = []
    f0 = func(*inputs, **kwargs)
    if not isinstance(f0, np.ndarray):
        f0 = np.array(f0)
    f0_shape = f0.shape

    def replace_input(arr_list: List[np.ndarray], idx: int, new_arr: np.ndarray) -> None:
        arr_list[idx] = new_arr

    def tiny_eps_for(arr: np.ndarray) -> float:
        eps_machine = np.finfo(arr.dtype).eps
        return np.sqrt(eps_machine) * max(np.max(np.abs(arr)), 1.0)

    base_inputs: List[np.ndarray] = list(inputs)

    for idx, x in enumerate(inputs):
        grad_z = np.zeros_like(x, dtype=np.result_type(x, f0, np.complex128))
        grad_conj_z = np.zeros_like(x, dtype=np.result_type(x, f0, np.complex128))
        is_complex = np.iscomplexobj(x)
        x_flat = x.ravel()
        gz_flat = grad_z.ravel()
        gzb_flat = grad_conj_z.ravel()

        for i in range(x_flat.size):
            orig_val = x_flat[i]

            if not is_complex:
                # complex‐step for real x[i]
                eps_c = tiny_eps_for(x.real)
                x_pert_flat = x_flat.copy().astype(np.complex128)
                x_pert_flat[i] = orig_val + 1j * eps_c
                x_pert = x_pert_flat.reshape(x.shape)

                replace_input(base_inputs, idx, x_pert)
                f_pert = func(*base_inputs, **kwargs)
                if not isinstance(f_pert, np.ndarray):
                    f_pert = np.array(f_pert)
                replace_input(base_inputs, idx, x)

                f_pert = ensure_shape(f_pert, f0_shape)
                gz_flat[i] = np.sum(f_pert).imag / eps_c
                gzb_flat[i] = 0.0 + 0.0j

            else:
                # sixth‐order FD for complex x[i]
                a = orig_val.real
                b = orig_val.imag
                eps_r = tiny_eps_for(np.array([a], dtype=x.real.dtype))
                eps_i = tiny_eps_for(np.array([b], dtype=x.real.dtype))

                def pert_real(k: int) -> np.ndarray:
                    tmp = x_flat.copy().astype(np.complex128)
                    tmp[i] = (a + k * eps_r) + 1j * b
                    return tmp.reshape(x.shape)

                fr_m3 = pert_real(-3)
                fr_m2 = pert_real(-2)
                fr_m1 = pert_real(-1)
                fr_p1 = pert_real( 1)
                fr_p2 = pert_real( 2)
                fr_p3 = pert_real( 3)

                replace_input(base_inputs, idx, fr_m3)
                f_r_m3 = func(*base_inputs, **kwargs); f_r_m3 = np.array(f_r_m3)
                replace_input(base_inputs, idx, fr_m2)
                f_r_m2 = func(*base_inputs, **kwargs); f_r_m2 = np.array(f_r_m2)
                replace_input(base_inputs, idx, fr_m1)
                f_r_m1 = func(*base_inputs, **kwargs); f_r_m1 = np.array(f_r_m1)
                replace_input(base_inputs, idx, fr_p1)
                f_r_p1 = func(*base_inputs, **kwargs); f_r_p1 = np.array(f_r_p1)
                replace_input(base_inputs, idx, fr_p2)
                f_r_p2 = func(*base_inputs, **kwargs); f_r_p2 = np.array(f_r_p2)
                replace_input(base_inputs, idx, fr_p3)
                f_r_p3 = func(*base_inputs, **kwargs); f_r_p3 = np.array(f_r_p3)
                replace_input(base_inputs, idx, x)

                f_r_m3 = ensure_shape(f_r_m3, f0_shape)
                f_r_m2 = ensure_shape(f_r_m2, f0_shape)
                f_r_m1 = ensure_shape(f_r_m1, f0_shape)
                f_r_p1 = ensure_shape(f_r_p1, f0_shape)
                f_r_p2 = ensure_shape(f_r_p2, f0_shape)
                f_r_p3 = ensure_shape(f_r_p3, f0_shape)

                dfx = (
                    f_r_p3
                    - 9.0 * f_r_p2
                    + 45.0 * f_r_p1
                    - 45.0 * f_r_m1
                    +  9.0 * f_r_m2
                    -      f_r_m3
                ) / (60.0 * eps_r)
                dfx_sum = np.sum(dfx)

                def pert_imag(k: int) -> np.ndarray:
                    tmp = x_flat.copy().astype(np.complex128)
                    tmp[i] = a + 1j * (b + k * eps_i)
                    return tmp.reshape(x.shape)

                fi_m3 = pert_imag(-3)
                fi_m2 = pert_imag(-2)
                fi_m1 = pert_imag(-1)
                fi_p1 = pert_imag( 1)
                fi_p2 = pert_imag( 2)
                fi_p3 = pert_imag( 3)

                replace_input(base_inputs, idx, fi_m3)
                f_i_m3 = func(*base_inputs, **kwargs); f_i_m3 = np.array(f_i_m3)
                replace_input(base_inputs, idx, fi_m2)
                f_i_m2 = func(*base_inputs, **kwargs); f_i_m2 = np.array(f_i_m2)
                replace_input(base_inputs, idx, fi_m1)
                f_i_m1 = func(*base_inputs, **kwargs); f_i_m1 = np.array(f_i_m1)
                replace_input(base_inputs, idx, fi_p1)
                f_i_p1 = func(*base_inputs, **kwargs); f_i_p1 = np.array(f_i_p1)
                replace_input(base_inputs, idx, fi_p2)
                f_i_p2 = func(*base_inputs, **kwargs); f_i_p2 = np.array(f_i_p2)
                replace_input(base_inputs, idx, fi_p3)
                f_i_p3 = func(*base_inputs, **kwargs); f_i_p3 = np.array(f_i_p3)
                replace_input(base_inputs, idx, x)

                f_i_m3 = ensure_shape(f_i_m3, f0_shape)
                f_i_m2 = ensure_shape(f_i_m2, f0_shape)
                f_i_m1 = ensure_shape(f_i_m1, f0_shape)
                f_i_p1 = ensure_shape(f_i_p1, f0_shape)
                f_i_p2 = ensure_shape(f_i_p2, f0_shape)
                f_i_p3 = ensure_shape(f_i_p3, f0_shape)

                dfy = (
                    f_i_p3
                    - 9.0  * f_i_p2
                    + 45.0 * f_i_p1
                    - 45.0 * f_i_m1
                    +  9.0 * f_i_m2
                    -      f_i_m3
                ) / (60.0 * eps_i)
                dfy_sum = np.sum(dfy)

                dz  = 0.5 * (dfx_sum - 1j * dfy_sum)
                dzb = 0.5 * (dfx_sum + 1j * dfy_sum)
                gz_flat[i]  = dz
                gzb_flat[i] = dzb

        grads.append((gz_flat.reshape(x.shape), gzb_flat.reshape(x.shape)))

    return grads
