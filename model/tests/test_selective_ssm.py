"""
Correctness tests for the selective SSM reference implementation.

These tests verify that:
1. Both recurrent and scan implementations produce identical outputs
2. All functions have correct input/output shapes
3. The discretization and delta computation work as expected
4. Edge cases are handled properly

This is the ground truth that all future implementations (PyTorch, CUDA) must match.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from model.reference.selective_ssm import (
    discretize_zoh,
    selective_ssm_recurrent,
    selective_ssm_scan,
    compute_delta,
)


class TestDiscretizeZOH:
    """Tests for the ZOH discretization function."""

    def test_output_shapes(self):
        """Discretization should return matrices with correct shapes."""
        state_dim = 4
        input_dim = 8

        A = np.random.randn(state_dim, state_dim)
        B = np.random.randn(state_dim, input_dim)
        delta = np.abs(np.random.randn(state_dim))  # positive step sizes

        A_bar, B_bar = discretize_zoh(A, B, delta)

        assert A_bar.shape == (state_dim, state_dim)
        assert B_bar.shape == (state_dim, input_dim)

    def test_small_delta_approximation(self):
        """For small delta, A_bar should be close to identity + A*delta."""
        state_dim = 4
        input_dim = 8

        A = np.random.randn(state_dim, state_dim) * 0.1  # small values
        B = np.random.randn(state_dim, input_dim)
        delta = np.ones(state_dim) * 0.01  # very small step size

        A_bar, B_bar = discretize_zoh(A, B, delta)

        # For small delta, A_bar ≈ I + A*delta
        expected_A_bar = np.eye(state_dim) + A * delta[:, np.newaxis]
        assert_allclose(A_bar, expected_A_bar, rtol=1e-5)

        # B_bar ≈ B*delta
        expected_B_bar = B * delta[:, np.newaxis]
        assert_allclose(B_bar, expected_B_bar, rtol=1e-5)

    def test_zero_delta(self):
        """Zero delta should give identity for A_bar and zeros for B_bar."""
        state_dim = 4
        input_dim = 8

        A = np.random.randn(state_dim, state_dim)
        B = np.random.randn(state_dim, input_dim)
        delta = np.zeros(state_dim)

        A_bar, B_bar = discretize_zoh(A, B, delta)

        assert_allclose(A_bar, np.eye(state_dim))
        assert_allclose(B_bar, np.zeros_like(B))


class TestComputeDelta:
    """Tests for the input-dependent step size computation."""

    def test_output_shape(self):
        """Delta should have shape [seq_len, state_dim]."""
        seq_len = 16
        input_dim = 8
        state_dim = 4

        u = np.random.randn(seq_len, input_dim)
        W_delta = np.random.randn(state_dim, input_dim)
        b_delta = np.random.randn(state_dim)

        delta = compute_delta(u, W_delta, b_delta)

        assert delta.shape == (seq_len, state_dim)

    def test_positivity(self):
        """Delta should always be non-negative (softplus ensures this)."""
        seq_len = 16
        input_dim = 8
        state_dim = 4

        u = np.random.randn(seq_len, input_dim) * 10  # large values
        W_delta = np.random.randn(state_dim, input_dim)
        b_delta = np.random.randn(state_dim) * 10

        delta = compute_delta(u, W_delta, b_delta)

        # Softplus ensures non-negativity, but floating point can produce exact zeros
        # for very negative inputs (where exp(x) underflows to 0)
        assert np.all(delta >= 0)
        # At least some values should be strictly positive
        assert np.any(delta > 0)

    def test_without_bias(self):
        """Delta computation should work without bias term."""
        seq_len = 16
        input_dim = 8
        state_dim = 4

        u = np.random.randn(seq_len, input_dim)
        W_delta = np.random.randn(state_dim, input_dim)

        delta = compute_delta(u, W_delta, b_delta=None)

        assert delta.shape == (seq_len, state_dim)
        assert np.all(delta > 0)


class TestSelectiveSSMRecurrent:
    """Tests for the recurrent (sequential) SSM implementation."""

    def test_output_shapes(self):
        """Recurrent SSM should return correct output shapes."""
        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = np.random.randn(seq_len, input_dim)
        A = np.random.randn(state_dim, state_dim)
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)
        delta = np.abs(np.random.randn(seq_len, state_dim))

        y, x_final = selective_ssm_recurrent(u, A, B, C, D, delta)

        assert y.shape == (seq_len, output_dim)
        assert x_final.shape == (state_dim,)

    def test_with_initial_state(self):
        """Recurrent SSM should accept custom initial state."""
        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = np.random.randn(seq_len, input_dim)
        A = np.random.randn(state_dim, state_dim)
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)
        delta = np.abs(np.random.randn(seq_len, state_dim))
        x0 = np.random.randn(state_dim)

        y, x_final = selective_ssm_recurrent(u, A, B, C, D, delta, x0=x0)

        assert y.shape == (seq_len, output_dim)
        assert x_final.shape == (state_dim,)

    def test_zero_input(self):
        """Zero input should produce zero output (with zero initial state)."""
        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = np.zeros((seq_len, input_dim))
        A = np.random.randn(state_dim, state_dim) * 0.1
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.zeros((output_dim, input_dim))  # no feedthrough
        delta = np.abs(np.random.randn(seq_len, state_dim))

        y, x_final = selective_ssm_recurrent(u, A, B, C, D, delta)

        # With zero input and zero initial state, output should be zero
        assert_allclose(y, np.zeros_like(y), atol=1e-10)
        assert_allclose(x_final, np.zeros_like(x_final), atol=1e-10)


class TestSelectiveSSMScan:
    """Tests for the scan (parallel) SSM implementation."""

    def test_output_shapes(self):
        """Scan SSM should return correct output shapes."""
        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = np.random.randn(seq_len, input_dim)
        A = np.random.randn(state_dim, state_dim)
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)
        delta = np.abs(np.random.randn(seq_len, state_dim))

        y, x_final = selective_ssm_scan(u, A, B, C, D, delta)

        assert y.shape == (seq_len, output_dim)
        assert x_final.shape == (state_dim,)

    def test_with_initial_state(self):
        """Scan SSM should accept custom initial state."""
        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = np.random.randn(seq_len, input_dim)
        A = np.random.randn(state_dim, state_dim)
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)
        delta = np.abs(np.random.randn(seq_len, state_dim))
        x0 = np.random.randn(state_dim)

        y, x_final = selective_ssm_scan(u, A, B, C, D, delta, x0=x0)

        assert y.shape == (seq_len, output_dim)
        assert x_final.shape == (state_dim,)


class TestRecurrentVsScanEquivalence:
    """
    Critical tests: recurrent and scan implementations must produce identical outputs.

    This is the most important test suite — it verifies that the two different
    computation strategies (sequential vs. parallel) are mathematically equivalent.
    """

    def test_outputs_match_random_input(self):
        """Both implementations should produce identical outputs for random input."""
        np.random.seed(42)

        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = np.random.randn(seq_len, input_dim)
        A = np.random.randn(state_dim, state_dim) * 0.1
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)
        delta = np.abs(np.random.randn(seq_len, state_dim)) + 0.1

        y_recurrent, x_final_recurrent = selective_ssm_recurrent(u, A, B, C, D, delta)
        y_scan, x_final_scan = selective_ssm_scan(u, A, B, C, D, delta)

        assert_allclose(y_recurrent, y_scan, rtol=1e-6, atol=1e-8)
        assert_allclose(x_final_recurrent, x_final_scan, rtol=1e-6, atol=1e-8)

    def test_outputs_match_with_initial_state(self):
        """Both implementations should match when using custom initial state."""
        np.random.seed(123)

        seq_len = 32
        input_dim = 16
        state_dim = 8
        output_dim = 16

        u = np.random.randn(seq_len, input_dim)
        A = np.random.randn(state_dim, state_dim) * 0.1
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)
        delta = np.abs(np.random.randn(seq_len, state_dim)) + 0.1
        x0 = np.random.randn(state_dim)

        y_recurrent, x_final_recurrent = selective_ssm_recurrent(
            u, A, B, C, D, delta, x0=x0
        )
        y_scan, x_final_scan = selective_ssm_scan(u, A, B, C, D, delta, x0=x0)

        assert_allclose(y_recurrent, y_scan, rtol=1e-6, atol=1e-8)
        assert_allclose(x_final_recurrent, x_final_scan, rtol=1e-6, atol=1e-8)

    def test_outputs_match_long_sequence(self):
        """Both implementations should match for longer sequences."""
        np.random.seed(456)

        seq_len = 256
        input_dim = 32
        state_dim = 16
        output_dim = 32

        u = np.random.randn(seq_len, input_dim)
        A = np.random.randn(state_dim, state_dim) * 0.05
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)
        delta = np.abs(np.random.randn(seq_len, state_dim)) + 0.1

        y_recurrent, x_final_recurrent = selective_ssm_recurrent(u, A, B, C, D, delta)
        y_scan, x_final_scan = selective_ssm_scan(u, A, B, C, D, delta)

        assert_allclose(y_recurrent, y_scan, rtol=1e-5, atol=1e-7)
        assert_allclose(x_final_recurrent, x_final_scan, rtol=1e-5, atol=1e-7)

    def test_outputs_match_different_dtypes(self):
        """Both implementations should match for float32 and float64."""
        np.random.seed(789)

        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        for dtype in [np.float32, np.float64]:
            u = np.random.randn(seq_len, input_dim).astype(dtype)
            A = (np.random.randn(state_dim, state_dim) * 0.1).astype(dtype)
            B = np.random.randn(state_dim, input_dim).astype(dtype)
            C = np.random.randn(output_dim, state_dim).astype(dtype)
            D = np.random.randn(output_dim, input_dim).astype(dtype)
            delta = (np.abs(np.random.randn(seq_len, state_dim)) + 0.1).astype(dtype)

            y_recurrent, x_final_recurrent = selective_ssm_recurrent(
                u, A, B, C, D, delta
            )
            y_scan, x_final_scan = selective_ssm_scan(u, A, B, C, D, delta)

            # Use looser tolerance for float32
            rtol = 1e-5 if dtype == np.float32 else 1e-6
            atol = 1e-7 if dtype == np.float32 else 1e-8

            assert_allclose(y_recurrent, y_scan, rtol=rtol, atol=atol)
            assert_allclose(x_final_recurrent, x_final_scan, rtol=rtol, atol=atol)


class TestEndToEndWithDelta:
    """End-to-end tests using the full selective SSM pipeline with compute_delta."""

    def test_full_pipeline(self):
        """Test the complete flow: input -> delta -> SSM -> output."""
        np.random.seed(999)

        seq_len = 32
        input_dim = 16
        state_dim = 8
        output_dim = 16

        # Input sequence
        u = np.random.randn(seq_len, input_dim)

        # SSM parameters
        A = np.random.randn(state_dim, state_dim) * 0.1
        B = np.random.randn(state_dim, input_dim)
        C = np.random.randn(output_dim, state_dim)
        D = np.random.randn(output_dim, input_dim)

        # Projection weights for delta
        W_delta = np.random.randn(state_dim, input_dim)
        b_delta = np.random.randn(state_dim)

        # Compute input-dependent delta
        delta = compute_delta(u, W_delta, b_delta)

        # Run both implementations
        y_recurrent, x_final_recurrent = selective_ssm_recurrent(u, A, B, C, D, delta)
        y_scan, x_final_scan = selective_ssm_scan(u, A, B, C, D, delta)

        # Check shapes
        assert y_recurrent.shape == (seq_len, output_dim)
        assert x_final_recurrent.shape == (state_dim,)
        assert y_scan.shape == (seq_len, output_dim)
        assert x_final_scan.shape == (state_dim,)

        # Check equivalence
        assert_allclose(y_recurrent, y_scan, rtol=1e-6, atol=1e-8)
        assert_allclose(x_final_recurrent, x_final_scan, rtol=1e-6, atol=1e-8)

        # Check delta is positive
        assert np.all(delta > 0)
