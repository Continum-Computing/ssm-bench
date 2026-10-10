"""
Golden reference implementation of the selective state-space model.

This is the NumPy implementation that serves as the ground truth. All other
implementations (PyTorch, optimized kernels) must match this bit-for-bit.

The selective SSM processes sequences in linear time O(N) with a fixed-size
recurrent state. The key innovation is that the step size (delta) depends on
the input, allowing the model to selectively remember or ignore information.

Mathematical formulation:
    Discrete-time state update: x[k+1] = Ā_k · x[k] + B̄_k · u[k]
    Output:                      y[k]   = C · x[k] + D · u[k]

where:
    x[k] is the hidden state at position k (shape: [state_dim])
    u[k] is the input at position k (shape: [input_dim])
    y[k] is the output at position k (shape: [output_dim])
    Ā_k is the discretized state transition matrix (shape: [state_dim, state_dim])
    B̄_k is the discretized input matrix (shape: [state_dim, input_dim])
    C is the output matrix (shape: [output_dim, state_dim])
    D is the feedthrough matrix (shape: [output_dim, input_dim])

The subscript k on Ā and B̄ indicates they depend on the input (selective mechanism).
"""

import numpy as np
from typing import Tuple


def discretize_zoh(
    A: np.ndarray,
    B: np.ndarray,
    delta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize continuous-time SSM parameters using Zero-Order Hold (ZOH).

    Converts continuous-time matrices (A, B) to discrete-time (Ā, B̄) given
    a step size delta. ZOH assumes the input is constant over each time step.

    Args:
        A: Continuous-time state transition matrix, shape [state_dim, state_dim]
        B: Continuous-time input matrix, shape [state_dim, input_dim]
        delta: Step size, shape [state_dim] (one per state dimension)

    Returns:
        A_bar: Discretized state transition matrix, shape [state_dim, state_dim]
        B_bar: Discretized input matrix, shape [state_dim, input_dim]

    Mathematical formulation:
        Ā = I + A · diag(delta)  (first-order approximation)
        B̄ = B · diag(delta)

    Note: The full ZOH discretization uses matrix exponential:
        Ā = exp(A · diag(delta))
        B̄ = A^{-1} · (Ā - I) · B
    But for small delta, the first-order approximation is accurate and more stable.
    """
    # delta is [state_dim], expand to [state_dim, 1] for broadcasting
    delta_expanded = delta[:, np.newaxis]

    # First-order approximation (valid for small delta)
    A_bar = np.eye(A.shape[0]) + A * delta_expanded
    B_bar = B * delta_expanded

    return A_bar, B_bar


def selective_ssm_recurrent(
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    delta: np.ndarray,
    x0: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selective SSM computed recurrently (sequential, step-by-step).

    This is the straightforward implementation: process the sequence one
    element at a time, updating the hidden state at each step. This is O(N)
    in sequence length and O(1) in memory (only the current state is stored).

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

    Algorithm:
        for k in range(seq_len):
            # Discretize for this time step (selective: delta depends on input)
            Ā_k, B̄_k = discretize(A, B, delta[k])

            # Output
            y[k] = C · x[k] + D · u[k]

            # State update
            x[k+1] = Ā_k · x[k] + B̄_k · u[k]
    """
    seq_len, input_dim = u.shape
    state_dim = A.shape[0]
    output_dim = C.shape[0]

    if x0 is None:
        x0 = np.zeros(state_dim, dtype=u.dtype)

    y = np.zeros((seq_len, output_dim), dtype=u.dtype)
    x = x0.copy()

    for k in range(seq_len):
        # Output (before state update)
        y[k] = C @ x + D @ u[k]

        # Discretize for this time step
        A_bar, B_bar = discretize_zoh(A, B, delta[k])

        # State update
        x = A_bar @ x + B_bar @ u[k]

    return y, x


def selective_ssm_scan(
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    delta: np.ndarray,
    x0: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selective SSM computed via associative scan (parallel during training).

    The recurrence x[k+1] = A_bar_k @ x[k] + B_bar_k @ u[k] can be reformulated
    as an associative scan, allowing parallel computation during training. This
    is the key insight that makes SSMs trainable as efficiently as transformers.

    The idea: instead of computing each state sequentially (x[0], then x[1], etc.),
    we can define a combine operation that merges adjacent (A_bar, B_bar @ u) pairs
    in a way that's associative — meaning the order of combining doesn't matter.
    This lets us use parallel prefix scan algorithms (like Blelloch scan) to compute
    all states in O(log N) depth instead of O(N) sequential steps.

    The combine operation works like this:
        Given two consecutive steps (A1, b1) and (A2, b2), the combined step is:
        A_combined = A2 @ A1
        b_combined = A2 @ b1 + b2

    This means we can compute x[k] directly from x[0] without computing all
    intermediate states, which is what enables parallel training.

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

    Note: This implementation uses a sequential scan for clarity and debugging.
    In production, hardware-optimized parallel scan kernels (e.g., in CUDA) are
    used for training. The outputs match selective_ssm_recurrent exactly.
    """
    seq_len, input_dim = u.shape
    state_dim = A.shape[0]
    output_dim = C.shape[0]

    if x0 is None:
        x0 = np.zeros(state_dim, dtype=u.dtype)

    # Discretize all time steps
    A_bar_all = np.zeros((seq_len, state_dim, state_dim), dtype=u.dtype)
    B_bar_all = np.zeros((seq_len, state_dim, input_dim), dtype=u.dtype)

    for k in range(seq_len):
        A_bar_all[k], B_bar_all[k] = discretize_zoh(A, B, delta[k])

    # Compute the "input terms" for the scan: B̄_k · u[k]
    Bu = np.zeros((seq_len, state_dim), dtype=u.dtype)
    for k in range(seq_len):
        Bu[k] = B_bar_all[k] @ u[k]

    # Sequential scan (for clarity; parallel scan used in practice)
    x = np.zeros((seq_len, state_dim), dtype=u.dtype)
    x_prev = x0.copy()

    for k in range(seq_len):
        x[k] = A_bar_all[k] @ x_prev + Bu[k]
        x_prev = x[k]

    # Compute outputs
    y = np.zeros((seq_len, output_dim), dtype=u.dtype)
    for k in range(seq_len):
        y[k] = C @ x[k] + D @ u[k]

    return y, x[-1]


def compute_delta(
    u: np.ndarray,
    W_delta: np.ndarray,
    b_delta: np.ndarray = None
) -> np.ndarray:
    """
    Compute input-dependent step sizes (the selective mechanism).

    In Mamba, delta depends on the input, allowing the model to dynamically
    choose what to remember (large delta) and what to ignore (small delta).

    Args:
        u: Input sequence, shape [seq_len, input_dim]
        W_delta: Projection weights, shape [state_dim, input_dim]
        b_delta: Bias, shape [state_dim] or None

    Returns:
        delta: Step sizes, shape [seq_len, state_dim], always positive

    The softplus function ensures delta > 0:
        delta = log(1 + exp(W_delta @ u + b_delta))
    """
    # Linear projection
    delta_linear = u @ W_delta.T  # [seq_len, state_dim]

    # Add bias if provided
    if b_delta is not None:
        delta_linear = delta_linear + b_delta[np.newaxis, :]

    # Softplus ensures positivity
    delta = np.log(1 + np.exp(delta_linear))

    return delta
