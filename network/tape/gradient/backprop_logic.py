# autodiff/tape/backprop_logic.py

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from ...types import Tensor  # Assuming Tensor is defined
from .funcs import (  # Assuming these are defined in a sibling module
    GRADIENTS,
    numerical_derivative,
)
from .types import Gradient, OpNode

# Assuming GradientTapeCore is available for its utility methods
# In the actual implementation, this might be a method of GradientTapeCore
# or take a GradientTapeCore instance as an argument.
# For modularity, we define these as functions that take necessary state.

_visit_stamp_counter_bp: int = (
    1  # Separate counter for backprop traversal if needed, or reuse from tape_core
)


def _compute_raw_gradients(
    node: OpNode, grad_res: Gradient, forced_dtype: np.dtype | None = None
) -> list[tuple[Tensor, Tensor]]:
    """
    Computes the "raw" (holomorphic and anti-holomorphic) gradients for a
    single `OpNode` (primitive operation).
    """
    name = node.func.__name__
    raw_grads: Any

    if grad_func := GRADIENTS.get(name):
        try:
            raw_grads = grad_func(
                (grad_res.holomorphic, grad_res.antiholomorphic),
                node.inputs,
                **node.kwargs,
            )
        except TypeError:
            raw_grads = grad_func(
                (grad_res.holomorphic, grad_res.antiholomorphic), node.inputs
            )
    else:
        warnings.warn(
            f"No analytical gradient for '{name}', using numerical approximation. "
            "Consider defining a gradient function (e.g., using `@GradientTape.def_grad`) "
            "for better performance and accuracy.",
            stacklevel=1,
        )
        approximations = numerical_derivative(
            node.func,
            node.inputs,
            node.kwargs,
            high_precision=True,
            max_order=8,
            verbose=False,
        )
        raw_grads = []
        for J_h, J_ah in approximations:
            J_h_t = Tensor(J_h)
            J_ah_t = Tensor(J_ah)
            gh = grad_res.holomorphic.dot(J_h_t) + grad_res.antiholomorphic.dot(
                J_ah_t.conj()
            )
            gah = grad_res.holomorphic.dot(J_ah_t) + grad_res.antiholomorphic.dot(
                J_h_t.conj()
            )
            raw_grads.append((gh, gah))

    return raw_grads


def _process_raw_gradients_format(
    raw_grads: Any, num_inputs: int
) -> tuple[tuple[Tensor | None, Tensor | None], ...]:
    """
    Normalizes the format of raw gradients returned by `_compute_raw_gradients`.
    """
    processed_grads: list[tuple[Tensor | None, Tensor | None]] = []

    if (
        num_inputs == 1
        and isinstance(raw_grads, tuple)
        and len(raw_grads) == 2
        and isinstance(raw_grads[0], (Tensor, np.ndarray))
    ):
        raw_grads = (raw_grads,)
    elif isinstance(raw_grads, list):
        raw_grads = tuple(raw_grads)
    elif not isinstance(raw_grads, tuple):
        warnings.warn(
            f"Unexpected gradient format {type(raw_grads)}. "
            "Expecting a tuple or list of gradient pairs. Skipping these inputs.",
            stacklevel=2,
        )
        raw_grads = tuple(((None, None),) * num_inputs)

    for grad_pair in raw_grads:
        if isinstance(grad_pair, tuple) and len(grad_pair) == 2:
            gh, gah = grad_pair
            if isinstance(gh, np.ndarray):
                gh = Tensor(gh)
            if isinstance(gah, np.ndarray):
                gah = Tensor(gah)
            processed_grads.append((gh, gah))
        elif isinstance(grad_pair, Tensor):
            processed_grads.append((grad_pair, Tensor(np.zeros_like(grad_pair.data))))
        else:
            processed_grads.append((None, None))

    while len(processed_grads) < num_inputs:
        processed_grads.append((None, None))

    return tuple(processed_grads)


def _gradient_recursive(
    tape_instance: Any,  # This will be the GradientTape instance
    node: OpNode,
    grad_res: Gradient,
) -> None:
    """
    Recursively computes and propagates gradients backward.
    This function needs access to tape_instance's `_unbroadcast_gradient`,
    `_initialize_input_gradient`, `_accumulate_and_apply_hook`, and `forced_dtype`.
    """
    raw_grads_for_inputs = _compute_raw_gradients(
        node, grad_res, tape_instance.forced_dtype
    )
    processed_grads = _process_raw_gradients_format(
        raw_grads_for_inputs, len(node.inputs)
    )

    for i, inp in enumerate(node.inputs):
        if i >= len(processed_grads):
            continue

        gh, gah = processed_grads[i]

        if gh is None and gah is None:
            continue

        if not isinstance(inp, Tensor):
            inp = Tensor(inp)

        if gh is not None:
            gh = tape_instance._unbroadcast_gradient(gh, inp.shape)
        if gah is not None:
            gah = tape_instance._unbroadcast_gradient(gah, inp.shape)

        tape_instance._initialize_input_gradient(inp)

        if gh is not None and gah is not None:
            tape_instance._accumulate_and_apply_hook(inp, gh, gah)


def _backpropagate(tape_instance: Any, target_arr: Tensor) -> None:
    """
    Performs the reverse-mode automatic differentiation process.
    This function needs access to tape_instance's `result_to_node`,
    `_nodes_in_order`, `grads`.
    """
    root_node = tape_instance.result_to_node.get(id(target_arr))
    if root_node is None:
        return

    global _visit_stamp_counter_bp  # Use global counter for traversal
    stamp = _visit_stamp_counter_bp
    _visit_stamp_counter_bp += 1

    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.last_visited == stamp:
            continue
        node.last_visited = stamp
        stack.extend(parent for parent in node.parents if parent.last_visited != stamp)

    for node in reversed(tape_instance._nodes_in_order):
        if node.last_visited != stamp:
            continue

        res_id = id(node.result)
        if res_id not in tape_instance.grads:
            continue

        grad_res = tape_instance.grads[res_id]
        _gradient_recursive(tape_instance, node, grad_res)


def _get_final_gradient_for_source(
    tape_instance: Any,  # This will be the GradientTape instance
    source_arr: Tensor,
    unconnected_gradients: Literal["zero", "none"],
) -> tuple[Tensor, Tensor] | None:
    """
    Extracts the final accumulated (holomorphic, anti-holomorphic) gradient.
    Needs access to tape_instance's `grads` and `_get_gradient_dtype`.
    """
    grad_pair = tape_instance.grads.get(id(source_arr))
    if grad_pair is None:
        if unconnected_gradients == "zero":
            dtype_src = tape_instance._get_gradient_dtype(
                source_arr.dtype, tape_instance.forced_dtype
            )
            zero_h = Tensor(np.zeros(source_arr.shape, dtype=dtype_src))
            zero_ah = Tensor(np.zeros(source_arr.shape, dtype=dtype_src))
            return (zero_h, zero_ah)
        return None
    return (grad_pair.holomorphic, grad_pair.antiholomorphic)
