# linear_algebra.py
import numpy as np
from network.tape.gradient.core.registry import registry
from network.tape.gradient.types import Gradient
from network.types.tensor import Tensor


@registry.register("matmul")
def _matmul_grad(upstream: Gradient, result: Tensor, A: Tensor, B: Tensor) -> list[Gradient]:
    """
    f(A, B) = A @ B is holomorphic in both A and B.

    For matrix Z = A @ B:
    dZ/dA = B.T, dZ/dA_conj = 0 (holomorphic)
    dZ/dB = A.T, dZ/dB_conj = 0 (holomorphic)

    Chain rule for holomorphic functions:
    dL/dA = upstream.h @ B.T
    dL/dA_conj = upstream.ah @ conj(B).T
    """
    if np.iscomplexobj(A):
        # Complex A: holomorphic function
        grad_A = Gradient(
            h=upstream.h @ B.T,
            ah=upstream.ah @ np.conj(B).T
        )
    else:
        # Real A: both Wirtinger derivatives contribute
        standard_grad = upstream.h @ B.T
        wirtinger_deriv = standard_grad * 0.5
        grad_A = Gradient(
            h=wirtinger_deriv + (upstream.ah @ B.T) * 0.5,
            ah=wirtinger_deriv + (upstream.ah @ B.T) * 0.5
        )

    if np.iscomplexobj(B):
        # Complex B: holomorphic function
        grad_B = Gradient(
            h=A.T @ upstream.h,
            ah=np.conj(A).T @ upstream.ah
        )
    else:
        # Real B: both Wirtinger derivatives contribute
        standard_grad = A.T @ upstream.h
        wirtinger_deriv = standard_grad * 0.5
        grad_B = Gradient(
            h=wirtinger_deriv + (A.T @ upstream.ah) * 0.5,
            ah=wirtinger_deriv + (A.T @ upstream.ah) * 0.5
        )

    return [grad_A, grad_B]


@registry.register("tensordot")
def _tensordot_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor, axes=2) -> list[Gradient]:
    """
    f(a, b) = tensordot(a, b, axes) is holomorphic in both a and b.
    """
    # Handle different forms of axes specification
    if isinstance(axes, int):
        a_axes = tuple(range(a.ndim - axes, a.ndim))
        b_axes = tuple(range(axes))
    else:
        a_axes, b_axes = axes
        if isinstance(a_axes, int):
            a_axes = (a_axes,)
        if isinstance(b_axes, int):
            b_axes = (b_axes,)

    # Remaining axes that weren't contracted
    b_remaining = tuple(i for i in range(b.ndim) if i not in b_axes)

    # Gradient dimensions for proper contractions
    grad_a_dims = list(range(len(b_remaining), upstream.h.ndim)) + list(range(len(b_axes)))

    if np.iscomplexobj(a):
        # Complex a: holomorphic function
        grad_a = Gradient(
            h=np.tensordot(upstream.h, b, axes=(grad_a_dims, b_remaining + b_axes)),
            ah=np.tensordot(upstream.ah, np.conj(b), axes=(grad_a_dims, b_remaining + b_axes))
        )
    else:
        # Real a: both Wirtinger derivatives contribute
        standard_grad = np.tensordot(upstream.h, b, axes=(grad_a_dims, b_remaining + b_axes))
        wirtinger_deriv = standard_grad * 0.5
        ah_contrib = np.tensordot(upstream.ah, b, axes=(grad_a_dims, b_remaining + b_axes)) * 0.5
        grad_a = Gradient(
            h=wirtinger_deriv + ah_contrib,
            ah=wirtinger_deriv + ah_contrib
        )

    # Similar for b
    a_remaining = tuple(i for i in range(a.ndim) if i not in a_axes)
    grad_b_dims = list(range(len(a_remaining))) + list(range(len(a_remaining), upstream.h.ndim))

    if np.iscomplexobj(b):
        # Complex b: holomorphic function
        grad_b = Gradient(
            h=np.tensordot(a, upstream.h, axes=(a_remaining + a_axes, grad_b_dims)),
            ah=np.tensordot(np.conj(a), upstream.ah, axes=(a_remaining + a_axes, grad_b_dims))
        )
    else:
        # Real b: both Wirtinger derivatives contribute
        standard_grad = np.tensordot(a, upstream.h, axes=(a_remaining + a_axes, grad_b_dims))
        wirtinger_deriv = standard_grad * 0.5
        ah_contrib = np.tensordot(a, upstream.ah, axes=(a_remaining + a_axes, grad_b_dims)) * 0.5
        grad_b = Gradient(
            h=wirtinger_deriv + ah_contrib,
            ah=wirtinger_deriv + ah_contrib
        )

    # None for axes parameter
    return [grad_a, grad_b, None]


@registry.register("dot")
def _dot_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = dot(a, b) is holomorphic in both a and b.
    """
    # For 1D tensors (inner product)
    if a.ndim == 1 and b.ndim == 1:
        if np.iscomplexobj(a):
            grad_a = Gradient(
                h=upstream.h * b,
                ah=upstream.ah * np.conj(b)
            )
        else:
            wirtinger_deriv = (upstream.h * b) * 0.5
            ah_contrib = (upstream.ah * b) * 0.5
            grad_a = Gradient(
                h=wirtinger_deriv + ah_contrib,
                ah=wirtinger_deriv + ah_contrib
            )

        if np.iscomplexobj(b):
            grad_b = Gradient(
                h=upstream.h * a,
                ah=upstream.ah * np.conj(a)
            )
        else:
            wirtinger_deriv = (upstream.h * a) * 0.5
            ah_contrib = (upstream.ah * a) * 0.5
            grad_b = Gradient(
                h=wirtinger_deriv + ah_contrib,
                ah=wirtinger_deriv + ah_contrib
            )

    # For matrix multiplication (2D tensors)
    elif a.ndim == 2 and b.ndim == 2:
        return _matmul_grad(upstream, result, a, b)

    # For ND tensor and 1D tensor
    elif a.ndim >= 2 and b.ndim == 1:
        reshaped_upstream_h = np.reshape(upstream.h, upstream.h.shape + (1,))
        reshaped_upstream_ah = np.reshape(upstream.ah, upstream.ah.shape + (1,))

        if np.iscomplexobj(a):
            grad_a = Gradient(
                h=reshaped_upstream_h * b,
                ah=reshaped_upstream_ah * np.conj(b)
            )
        else:
            wirtinger_deriv = (reshaped_upstream_h * b) * 0.5
            ah_contrib = (reshaped_upstream_ah * b) * 0.5
            grad_a = Gradient(
                h=wirtinger_deriv + ah_contrib,
                ah=wirtinger_deriv + ah_contrib
            )

        axes_to_sum = tuple(range(a.ndim - 1))
        if np.iscomplexobj(b):
            grad_b = Gradient(
                h=np.tensordot(upstream.h, a, axes=(axes_to_sum, axes_to_sum)),
                ah=np.tensordot(upstream.ah, np.conj(a), axes=(axes_to_sum, axes_to_sum))
            )
        else:
            standard_grad = np.tensordot(upstream.h, a, axes=(axes_to_sum, axes_to_sum))
            wirtinger_deriv = standard_grad * 0.5
            ah_contrib = np.tensordot(upstream.ah, a, axes=(axes_to_sum, axes_to_sum)) * 0.5
            grad_b = Gradient(
                h=wirtinger_deriv + ah_contrib,
                ah=wirtinger_deriv + ah_contrib
            )

    else:
        # For other combinations, delegate to tensordot
        return _tensordot_grad(upstream, result, a, b, axes=([a.ndim - 1], [0]))

    return [grad_a, grad_b]


@registry.register("vdot")
def _vdot_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = vdot(a, b) = sum(conj(a) * b) is NOT holomorphic.

    vdot(a, b) = sum(a_conj * b)
    dvdot/da = 0, dvdot/da_conj = b (since vdot depends on a_conj, not a)
    dvdot/db = a_conj, dvdot/db_conj = 0
    """
    # For a: vdot depends on conj(a), so gradient goes to anti-holomorphic part
    if np.iscomplexobj(a):
        grad_a = Gradient(
            h=upstream.h * np.zeros_like(b),  # dvdot/da = 0
            ah=upstream.ah * b                # dvdot/da_conj = b
        )
    else:
        # Real case: vdot(a,b) = sum(a*b), so d/da = b
        wirtinger_deriv = b * 0.5
        grad_a = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    # For b: vdot depends holomorphically on b
    if np.iscomplexobj(b):
        grad_b = Gradient(
            h=upstream.h * np.conj(a),       # dvdot/db = a_conj
            ah=upstream.ah * a               # dvdot/db_conj (conjugate rule)
        )
    else:
        # Real case: same as above
        wirtinger_deriv = a * 0.5
        grad_b = Gradient(
            h=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv,
            ah=upstream.h * wirtinger_deriv + upstream.ah * wirtinger_deriv
        )

    return [grad_a, grad_b]


@registry.register("inner")
def _inner_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = inner(a, b) behaves like vdot for complex tensors.
    """
    return _vdot_grad(upstream, result, a, b)


@registry.register("outer")
def _outer_grad(upstream: Gradient, result: Tensor, a: Tensor, b: Tensor) -> list[Gradient]:
    """
    f(a, b) = outer(a, b) = a[:, newaxis] * b[newaxis, :]
    This is holomorphic in both a and b.
    """
    if np.iscomplexobj(a):
        grad_a = Gradient(
            h=np.tensordot(upstream.h, b, axes=([1], [0])),
            ah=np.tensordot(upstream.ah, np.conj(b), axes=([1], [0]))
        )
    else:
        standard_grad = np.tensordot(upstream.h, b, axes=([1], [0]))
        wirtinger_deriv = standard_grad * 0.5
        ah_contrib = np.tensordot(upstream.ah, b, axes=([1], [0])) * 0.5
        grad_a = Gradient(
            h=wirtinger_deriv + ah_contrib,
            ah=wirtinger_deriv + ah_contrib
        )

    if np.iscomplexobj(b):
        grad_b = Gradient(
            h=np.tensordot(a, upstream.h, axes=([0], [0])),
            ah=np.tensordot(np.conj(a), upstream.ah, axes=([0], [0]))
        )
    else:
        standard_grad = np.tensordot(a, upstream.h, axes=([0], [0]))
        wirtinger_deriv = standard_grad * 0.5
        ah_contrib = np.tensordot(a, upstream.ah, axes=([0], [0])) * 0.5
        grad_b = Gradient(
            h=wirtinger_deriv + ah_contrib,
            ah=wirtinger_deriv + ah_contrib
        )

    return [grad_a, grad_b]