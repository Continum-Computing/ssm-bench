"""
Tests verifying PyTorch implementation matches NumPy reference exactly.

These tests are critical: they ensure the PyTorch version (which enables GPU
profiling) produces identical outputs to the NumPy reference (which is the
ground truth).
"""

import torch
import numpy as np
import pytest
from numpy.testing import assert_allclose

from model.reference.selective_ssm import (
    discretize_zoh as discretize_zoh_numpy,
    selective_ssm_recurrent as selective_ssm_recurrent_numpy,
    selective_ssm_scan as selective_ssm_scan_numpy,
    compute_delta as compute_delta_numpy,
)

from model.pytorch_impl.selective_ssm import (
    discretize_zoh as discretize_zoh_torch,
    selective_ssm_recurrent as selective_ssm_recurrent_torch,
    selective_ssm_scan as selective_ssm_scan_torch,
    compute_delta as compute_delta_torch,
)


class TestDiscretizeZOH:
    """Tests for ZOH discretization: PyTorch vs NumPy."""

    def test_matches_numpy(self):
        """PyTorch discretization should match NumPy exactly."""
        np.random.seed(42)

        state_dim = 4
        input_dim = 8

        A_np = np.random.randn(state_dim, state_dim)
        B_np = np.random.randn(state_dim, input_dim)
        delta_np = np.abs(np.random.randn(state_dim))

        A_torch = torch.from_numpy(A_np.copy())
        B_torch = torch.from_numpy(B_np.copy())
        delta_torch = torch.from_numpy(delta_np.copy())

        A_bar_np, B_bar_np = discretize_zoh_numpy(A_np, B_np, delta_np)
        A_bar_torch, B_bar_torch = discretize_zoh_torch(A_torch, B_torch, delta_torch)

        assert_allclose(A_bar_torch.numpy(), A_bar_np, rtol=1e-6)
        assert_allclose(B_bar_torch.numpy(), B_bar_np, rtol=1e-6)


class TestComputeDelta:
    """Tests for delta computation: PyTorch vs NumPy."""

    def test_matches_numpy(self):
        """PyTorch delta computation should match NumPy exactly."""
        np.random.seed(123)

        seq_len = 16
        input_dim = 8
        state_dim = 4

        u_np = np.random.randn(seq_len, input_dim)
        W_delta_np = np.random.randn(state_dim, input_dim)
        b_delta_np = np.random.randn(state_dim)

        u_torch = torch.from_numpy(u_np.copy())
        W_delta_torch = torch.from_numpy(W_delta_np.copy())
        b_delta_torch = torch.from_numpy(b_delta_np.copy())

        delta_np = compute_delta_numpy(u_np, W_delta_np, b_delta_np)
        delta_torch = compute_delta_torch(u_torch, W_delta_torch, b_delta_torch)

        assert_allclose(delta_torch.numpy(), delta_np, rtol=1e-6)


class TestSelectiveSSMRecurrent:
    """Tests for recurrent SSM: PyTorch vs NumPy."""

    def test_matches_numpy_random_input(self):
        """PyTorch recurrent SSM should match NumPy for random inputs."""
        np.random.seed(456)

        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u_np = np.random.randn(seq_len, input_dim)
        A_np = np.random.randn(state_dim, state_dim) * 0.1
        B_np = np.random.randn(state_dim, input_dim)
        C_np = np.random.randn(output_dim, state_dim)
        D_np = np.random.randn(output_dim, input_dim)
        delta_np = np.abs(np.random.randn(seq_len, state_dim)) + 0.1

        u_torch = torch.from_numpy(u_np.copy())
        A_torch = torch.from_numpy(A_np.copy())
        B_torch = torch.from_numpy(B_np.copy())
        C_torch = torch.from_numpy(C_np.copy())
        D_torch = torch.from_numpy(D_np.copy())
        delta_torch = torch.from_numpy(delta_np.copy())

        y_np, x_final_np = selective_ssm_recurrent_numpy(
            u_np, A_np, B_np, C_np, D_np, delta_np
        )
        y_torch, x_final_torch = selective_ssm_recurrent_torch(
            u_torch, A_torch, B_torch, C_torch, D_torch, delta_torch
        )

        assert_allclose(y_torch.numpy(), y_np, rtol=1e-6, atol=1e-8)
        assert_allclose(x_final_torch.numpy(), x_final_np, rtol=1e-6, atol=1e-8)

    def test_matches_numpy_with_initial_state(self):
        """PyTorch recurrent SSM should match NumPy with custom initial state."""
        np.random.seed(789)

        seq_len = 32
        input_dim = 16
        state_dim = 8
        output_dim = 16

        u_np = np.random.randn(seq_len, input_dim)
        A_np = np.random.randn(state_dim, state_dim) * 0.1
        B_np = np.random.randn(state_dim, input_dim)
        C_np = np.random.randn(output_dim, state_dim)
        D_np = np.random.randn(output_dim, input_dim)
        delta_np = np.abs(np.random.randn(seq_len, state_dim)) + 0.1
        x0_np = np.random.randn(state_dim)

        u_torch = torch.from_numpy(u_np.copy())
        A_torch = torch.from_numpy(A_np.copy())
        B_torch = torch.from_numpy(B_np.copy())
        C_torch = torch.from_numpy(C_np.copy())
        D_torch = torch.from_numpy(D_np.copy())
        delta_torch = torch.from_numpy(delta_np.copy())
        x0_torch = torch.from_numpy(x0_np.copy())

        y_np, x_final_np = selective_ssm_recurrent_numpy(
            u_np, A_np, B_np, C_np, D_np, delta_np, x0=x0_np
        )
        y_torch, x_final_torch = selective_ssm_recurrent_torch(
            u_torch, A_torch, B_torch, C_torch, D_torch, delta_torch, x0=x0_torch
        )

        assert_allclose(y_torch.numpy(), y_np, rtol=1e-6, atol=1e-8)
        assert_allclose(x_final_torch.numpy(), x_final_np, rtol=1e-6, atol=1e-8)


class TestSelectiveSSMScan:
    """Tests for scan SSM: PyTorch vs NumPy."""

    def test_matches_numpy_random_input(self):
        """PyTorch scan SSM should match NumPy for random inputs."""
        np.random.seed(999)

        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u_np = np.random.randn(seq_len, input_dim)
        A_np = np.random.randn(state_dim, state_dim) * 0.1
        B_np = np.random.randn(state_dim, input_dim)
        C_np = np.random.randn(output_dim, state_dim)
        D_np = np.random.randn(output_dim, input_dim)
        delta_np = np.abs(np.random.randn(seq_len, state_dim)) + 0.1

        u_torch = torch.from_numpy(u_np.copy())
        A_torch = torch.from_numpy(A_np.copy())
        B_torch = torch.from_numpy(B_np.copy())
        C_torch = torch.from_numpy(C_np.copy())
        D_torch = torch.from_numpy(D_np.copy())
        delta_torch = torch.from_numpy(delta_np.copy())

        y_np, x_final_np = selective_ssm_scan_numpy(
            u_np, A_np, B_np, C_np, D_np, delta_np
        )
        y_torch, x_final_torch = selective_ssm_scan_torch(
            u_torch, A_torch, B_torch, C_torch, D_torch, delta_torch
        )

        assert_allclose(y_torch.numpy(), y_np, rtol=1e-6, atol=1e-8)
        assert_allclose(x_final_torch.numpy(), x_final_np, rtol=1e-6, atol=1e-8)

    def test_matches_numpy_long_sequence(self):
        """PyTorch scan SSM should match NumPy for long sequences."""
        np.random.seed(111)

        seq_len = 256
        input_dim = 32
        state_dim = 16
        output_dim = 32

        u_np = np.random.randn(seq_len, input_dim)
        A_np = np.random.randn(state_dim, state_dim) * 0.05
        B_np = np.random.randn(state_dim, input_dim)
        C_np = np.random.randn(output_dim, state_dim)
        D_np = np.random.randn(output_dim, input_dim)
        delta_np = np.abs(np.random.randn(seq_len, state_dim)) + 0.1

        u_torch = torch.from_numpy(u_np.copy())
        A_torch = torch.from_numpy(A_np.copy())
        B_torch = torch.from_numpy(B_np.copy())
        C_torch = torch.from_numpy(C_np.copy())
        D_torch = torch.from_numpy(D_np.copy())
        delta_torch = torch.from_numpy(delta_np.copy())

        y_np, x_final_np = selective_ssm_scan_numpy(
            u_np, A_np, B_np, C_np, D_np, delta_np
        )
        y_torch, x_final_torch = selective_ssm_scan_torch(
            u_torch, A_torch, B_torch, C_torch, D_torch, delta_torch
        )

        assert_allclose(y_torch.numpy(), y_np, rtol=1e-5, atol=1e-7)
        assert_allclose(x_final_torch.numpy(), x_final_np, rtol=1e-5, atol=1e-7)


class TestEndToEndPipeline:
    """End-to-end tests: full pipeline with delta computation."""

    def test_full_pipeline_matches_numpy(self):
        """Full pipeline (delta + SSM) should match NumPy exactly."""
        np.random.seed(222)

        seq_len = 32
        input_dim = 16
        state_dim = 8
        output_dim = 16

        u_np = np.random.randn(seq_len, input_dim)
        A_np = np.random.randn(state_dim, state_dim) * 0.1
        B_np = np.random.randn(state_dim, input_dim)
        C_np = np.random.randn(output_dim, state_dim)
        D_np = np.random.randn(output_dim, input_dim)
        W_delta_np = np.random.randn(state_dim, input_dim)
        b_delta_np = np.random.randn(state_dim)

        u_torch = torch.from_numpy(u_np.copy())
        A_torch = torch.from_numpy(A_np.copy())
        B_torch = torch.from_numpy(B_np.copy())
        C_torch = torch.from_numpy(C_np.copy())
        D_torch = torch.from_numpy(D_np.copy())
        W_delta_torch = torch.from_numpy(W_delta_np.copy())
        b_delta_torch = torch.from_numpy(b_delta_np.copy())

        # NumPy pipeline
        delta_np = compute_delta_numpy(u_np, W_delta_np, b_delta_np)
        y_np, x_final_np = selective_ssm_recurrent_numpy(
            u_np, A_np, B_np, C_np, D_np, delta_np
        )

        # PyTorch pipeline
        delta_torch = compute_delta_torch(u_torch, W_delta_torch, b_delta_torch)
        y_torch, x_final_torch = selective_ssm_recurrent_torch(
            u_torch, A_torch, B_torch, C_torch, D_torch, delta_torch
        )

        assert_allclose(delta_torch.numpy(), delta_np, rtol=1e-6)
        assert_allclose(y_torch.numpy(), y_np, rtol=1e-6, atol=1e-8)
        assert_allclose(x_final_torch.numpy(), x_final_np, rtol=1e-6, atol=1e-8)


class TestPyTorchSpecificFeatures:
    """Tests for PyTorch-specific capabilities (GPU, gradients)."""

    def test_device_handling(self):
        """Implementation should work on both CPU and GPU (if available)."""
        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = torch.randn(seq_len, input_dim)
        A = torch.randn(state_dim, state_dim) * 0.1
        B = torch.randn(state_dim, input_dim)
        C = torch.randn(output_dim, state_dim)
        D = torch.randn(output_dim, input_dim)
        delta = torch.abs(torch.randn(seq_len, state_dim)) + 0.1

        # Test on CPU
        y_cpu, x_cpu = selective_ssm_recurrent_torch(
            u, A, B, C, D, delta
        )
        assert y_cpu.device.type == "cpu"
        assert x_cpu.device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            u_gpu = u.cuda()
            A_gpu = A.cuda()
            B_gpu = B.cuda()
            C_gpu = C.cuda()
            D_gpu = D.cuda()
            delta_gpu = delta.cuda()

            y_gpu, x_gpu = selective_ssm_recurrent_torch(
                u_gpu, A_gpu, B_gpu, C_gpu, D_gpu, delta_gpu
            )
            assert y_gpu.device.type == "cuda"
            assert x_gpu.device.type == "cuda"

            # Results should match CPU version
            assert_allclose(y_gpu.cpu().numpy(), y_cpu.numpy(), rtol=1e-5, atol=1e-7)
            assert_allclose(x_gpu.cpu().numpy(), x_cpu.numpy(), rtol=1e-5, atol=1e-7)

    def test_gradient_flow(self):
        """Implementation should support gradient computation."""
        seq_len = 16
        input_dim = 8
        state_dim = 4
        output_dim = 8

        u = torch.randn(seq_len, input_dim, requires_grad=True)
        A = torch.randn(state_dim, state_dim) * 0.1
        B = torch.randn(state_dim, input_dim, requires_grad=True)
        C = torch.randn(output_dim, state_dim, requires_grad=True)
        D = torch.randn(output_dim, input_dim, requires_grad=True)
        delta = torch.abs(torch.randn(seq_len, state_dim)) + 0.1

        y, x_final = selective_ssm_recurrent_torch(u, A, B, C, D, delta)

        # Compute a simple loss and backprop
        loss = y.sum()
        loss.backward()

        # Check that gradients exist
        assert u.grad is not None
        assert B.grad is not None
        assert C.grad is not None
        assert D.grad is not None

        # Check gradient shapes
        assert u.grad.shape == u.shape
        assert B.grad.shape == B.shape
        assert C.grad.shape == C.shape
        assert D.grad.shape == D.shape
