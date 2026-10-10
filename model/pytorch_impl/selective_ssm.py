"""
PyTorch implementation of the selective state-space model.

This implementation mirrors the NumPy reference exactly but uses PyTorch tensors
and operations, enabling GPU acceleration and gradient computation.

All functions should produce identical outputs to the NumPy reference when given
the same inputs (verified by tests).
"""

import torch
import numpy as np
from typing import Tuple


def discretize_zoh(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discretize continuous-time SSM parameters using Zero-Order Hold (ZOH).

    PyTorch version of the NumPy reference implementation.

    Args:
        A: Continuous-time state transition matrix, shape [state_dim, state_dim]
        B: Continuous-time input matrix, shape [state_dim, input_dim]
        delta: Step size, shape [state_dim] (one per state dimension)

    Returns:
        A_bar: Discretized state transition matrix, shape [state_dim, state_dim]
        B_bar: Discretized input matrix, shape [state_dim, input_dim]
    """
    # delta is [state_dim], expand to [state_dim, 1] for broadcasting
    delta_expanded = delta.unsqueeze(1)

    # First-order approximation (valid for small delta)
    A_bar = torch.eye(A.shape[0], dtype=A.dtype, device=A.device) + A * delta_expanded
    B_bar = B * delta_expanded

    return A_bar, B_bar


def selective_ssm_recurrent(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    delta: torch.Tensor,
    x0: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selective SSM computed recurrently (sequential, step-by-step).

    PyTorch version of the NumPy reference implementation.

    Args:
        u: Input sequence, shape [seq_len, input_dim]
        A: State transition matrix, shape [state_dim, state_dim]
        B: Input matrix, shape [state_dim, input_dim]
        C: Output matrix, shape [output_dim, state_dim]
        D: Feedthrough matrix, shape [output_dim, input_dim]
        delta: Step sizes, shape [seq_len, state_dim] (input-dependent)
        x0: Initial hidden state, shape [state_dim]. Defaults to zeros.

    Returns:
        y: Output sequence, shape [seq_len, output_dim]
        x_final: Final hidden state, shape [state_dim]
    """
    seq_len, input_dim = u.shape
    state_dim = A.shape[0]
    output_dim = C.shape[0]

    if x0 is None:
        x0 = torch.zeros(state_dim, dtype=u.dtype, device=u.device)

    y = torch.zeros((seq_len, output_dim), dtype=u.dtype, device=u.device)
    x = x0.clone()

    for k in range(seq_len):
        # Output (before state update)
        y[k] = C @ x + D @ u[k]

        # Discretize for this time step
        A_bar, B_bar = discretize_zoh(A, B, delta[k])

        # State update
        x = A_bar @ x + B_bar @ u[k]

    return y, x


def selective_ssm_scan(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    delta: torch.Tensor,
    x0: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selective SSM computed via associative scan (parallel during training).

    PyTorch version of the NumPy reference implementation.

    Args:
        u: Input sequence, shape [seq_len, input_dim]
        A: State transition matrix, shape [state_dim, state_dim]
        B: Input matrix, shape [state_dim, input_dim]
        C: Output matrix, shape [output_dim, state_dim]
        D: Feedthrough matrix, shape [output_dim, input_dim]
        delta: Step sizes, shape [seq_len, state_dim] (input-dependent)
        x0: Initial hidden state, shape [state_dim]. Defaults to zeros.

    Returns:
        y: Output sequence, shape [seq_len, output_dim]
        x_final: Final hidden state, shape [state_dim]
    """
    seq_len, input_dim = u.shape
    state_dim = A.shape[0]
    output_dim = C.shape[0]

    if x0 is None:
        x0 = torch.zeros(state_dim, dtype=u.dtype, device=u.device)

    # Discretize all time steps
    A_bar_all = torch.zeros((seq_len, state_dim, state_dim), dtype=u.dtype, device=u.device)
    B_bar_all = torch.zeros((seq_len, state_dim, input_dim), dtype=u.dtype, device=u.device)

    for k in range(seq_len):
        A_bar_all[k], B_bar_all[k] = discretize_zoh(A, B, delta[k])

    # Compute the "input terms" for the scan: B_bar_k @ u[k]
    Bu = torch.zeros((seq_len, state_dim), dtype=u.dtype, device=u.device)
    for k in range(seq_len):
        Bu[k] = B_bar_all[k] @ u[k]

    # Sequential scan (for clarity; parallel scan used in practice)
    # We compute the state trajectory: x[0], x[1], ..., x[seq_len-1]
    # where x[0] = x0 (initial state)
    # and x[k+1] = A_bar[k] @ x[k] + B_bar[k] @ u[k]
    # Then y[k] = C @ x[k] + D @ u[k]

    x_all = torch.zeros((seq_len, state_dim), dtype=u.dtype, device=u.device)
    x_all[0] = x0.clone()

    for k in range(seq_len - 1):
        x_all[k+1] = A_bar_all[k] @ x_all[k] + Bu[k]

    # Compute outputs
    y = torch.zeros((seq_len, output_dim), dtype=u.dtype, device=u.device)
    for k in range(seq_len):
        y[k] = C @ x_all[k] + D @ u[k]

    # Compute final state (after processing the last input)
    x_final = A_bar_all[-1] @ x_all[-1] + Bu[-1]

    return y, x_final


def compute_delta(
    u: torch.Tensor,
    W_delta: torch.Tensor,
    b_delta: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute input-dependent step sizes (the selective mechanism).

    PyTorch version of the NumPy reference implementation.

    Args:
        u: Input sequence, shape [seq_len, input_dim]
        W_delta: Projection weights, shape [state_dim, input_dim]
        b_delta: Bias, shape [state_dim] or None

    Returns:
        delta: Step sizes, shape [seq_len, state_dim], always non-negative
    """
    # Linear projection
    delta_linear = u @ W_delta.T  # [seq_len, state_dim]

    # Add bias if provided
    if b_delta is not None:
        delta_linear = delta_linear + b_delta.unsqueeze(0)

    # Softplus ensures non-negativity
    delta = torch.nn.functional.softplus(delta_linear)

    return delta
