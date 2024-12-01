import os
import random
import numpy as np
import torch
import triton
import triton.testing

from pathlib import Path
from typing import Callable

from layers import (
    MultiheadAttn,
    MultiheadFlashAttn,
    MultiheadDiffAttn,
    MultiheadFlashDiffAttn,
    MultiheadDiffAttnKernel,
)

# Benchmarking parameters
FOLDER_PREFIX = "A100"                                  # Prefix for saving results
NUM_HEADS = 16                                          # Number of heads
HEAD_DIM = 32                                           # Actual head_dim is 2 * HEAD_DIM
MODE = "fwd+bwd"                                        # Benchmark mode: "fwd" or "fwd+bwd"
RMS_NORM = True                                         # Compile kernel with RMS normalization
SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]         # Sequence lengths


def set_deterministic(seed=0):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_tensor_generator(*shape, dtype: torch.dtype, device: str):
    x = (
        torch.empty(*shape, dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )
    x.retain_grad()
    return x


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=SEQ_LENS,
        y_log=True,
        ylabel="Runtime (ms)",
        line_arg="implementation",
        line_vals=[
            "A (PyTorch)",
            "A (FlashAttention-2)",
            "DA (PyTorch)",
            "DA (FlashAttention-2)",
            "DA (Kernel)",
        ],
        line_names=[
            "A (PyTorch)",
            "A (FlashAttention-2)",
            "DA (PyTorch)",
            "DA (FlashAttention-2)",
            "DA (Kernel)",
        ],
        plot_name="runtime",
        args={
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "mode": MODE,
            "rms_norm": RMS_NORM,
        },
    )
)
def benchmark_runtime(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    mode: str,
    rms_norm: bool,
    implementation: str,
    dtype=torch.float16,
    device="cuda",
):

    quantiles = [0.5, 0.05, 0.95]
    batch_size = int(16384 / seq_len)

    random_tensor = lambda *shape: random_tensor_generator(
        *shape, dtype=dtype, device=device
    )

    if implementation == "A (PyTorch)":
        fn = MultiheadAttn
    elif implementation == "A (FlashAttention-2)":
        fn = MultiheadFlashAttn
    elif implementation == "DA (PyTorch)":
        fn = MultiheadDiffAttn
    elif implementation == "DA (FlashAttention-2)":
        fn = MultiheadFlashDiffAttn
    elif implementation == "DA (Kernel)":
        fn = MultiheadDiffAttnKernel
    else:
        raise ValueError("Invalid implementation")

    if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn, MultiheadDiffAttnKernel]:
        q1 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        q2 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        k1 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        k2 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        v = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
        lambda_scale = random_tensor(1)
    else:
        q = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
        k = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
        v = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)

    dout = torch.rand_like(v)

    def run(mode: str):
        if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn, MultiheadDiffAttnKernel]:
            y = fn(q1, q2, k1, k2, v, lambda_scale=lambda_scale, rms_norm=rms_norm)

            if mode == "fwd+bwd":
                y.backward(dout)
        else:
            y = fn(q, k, v, rms_norm=rms_norm)

            if mode == "fwd+bwd":
                y.backward(dout)

    try:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: run(mode), quantiles=quantiles
        )
    except RuntimeError as _:
        torch.cuda.empty_cache()
        ms, min_ms, max_ms = np.nan, np.nan, np.nan

    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=SEQ_LENS,
        y_log=True,
        ylabel="Peak Memory Usage (MB)",
        line_arg="implementation",
        line_vals=[
            "A (PyTorch)",
            "A (FlashAttention-2)",
            "DA (PyTorch)",
            "DA (FlashAttention-2)",
            "DA (Kernel)",
        ],
        line_names=[
            "A (PyTorch)",
            "A (FlashAttention-2)",
            "DA (PyTorch)",
            "DA (FlashAttention-2)",
            "DA (Kernel)",
        ],
        plot_name="memory",
        args={
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "mode": MODE,
            "rms_norm": RMS_NORM,
        },
    )
)
def benchmark_memory_usage(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    mode: str,
    rms_norm: bool,
    implementation: str,
    dtype=torch.float16,
    device="cuda",
):
    # NOTE: Using Triton's autotune produces incorrect measurements! Disable before running this benchmark.
    batch_size = int(16384 / seq_len)

    random_tensor = lambda *shape: random_tensor_generator(
        *shape, dtype=dtype, device=device
    )

    if implementation == "A (PyTorch)":
        fn = MultiheadAttn
    elif implementation == "A (FlashAttention-2)":
        fn = MultiheadFlashAttn
    elif implementation == "DA (PyTorch)":
        fn = MultiheadDiffAttn
    elif implementation == "DA (FlashAttention-2)":
        fn = MultiheadFlashDiffAttn
    elif implementation == "DA (Kernel)":
        fn = MultiheadDiffAttnKernel
    else:
        raise ValueError("Invalid implementation")

    if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn, MultiheadDiffAttnKernel]:
        q1 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        q2 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        k1 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        k2 = random_tensor(batch_size, num_heads, seq_len, head_dim)
        v = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
        lambda_scale = random_tensor(1)
    else:
        q = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
        k = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
        v = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)

    dout = torch.rand_like(v)

    torch.cuda.reset_peak_memory_stats()

    if fn in [MultiheadDiffAttn, MultiheadFlashDiffAttn, MultiheadDiffAttnKernel]:
        try:
            y = fn(q1, q2, k1, k2, v, lambda_scale=lambda_scale, rms_norm=rms_norm)

            if mode == "fwd+bwd":
                y.backward(dout)

        except RuntimeError as _:
            torch.cuda.empty_cache()
            return np.nan
    else:
        try:
            y = fn(q, k, v, rms_norm=rms_norm)

            if mode == "fwd+bwd":
                y.backward(dout)

        except RuntimeError as _:
            torch.cuda.empty_cache()
            return np.nan

    max_memory = torch.cuda.max_memory_allocated() / (1024**2)

    return max_memory


def benchmark_difference(
    fn1: Callable = MultiheadDiffAttn,
    fn2: Callable = MultiheadDiffAttnKernel,
    seq_len: int = 256,
    head_dim: int = HEAD_DIM,
    num_heads: int = NUM_HEADS,
    batch_size: int = 1,
    mode: str = MODE,
    rms_norm: bool = RMS_NORM,
    N: int = 100,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    save_path: Path = "./results/difference/",
    benchmark_name: str = "difference",
):

    random_tensor = lambda *shape: random_tensor_generator(
        *shape, dtype=dtype, device=device
    )

    diff = np.zeros(N)

    for i in range(N):
        if (fn1 and fn2) in [
            MultiheadDiffAttn,
            MultiheadFlashDiffAttn,
            MultiheadDiffAttnKernel,
        ]:
            q1 = random_tensor(batch_size, num_heads, seq_len, head_dim)
            q2 = random_tensor(batch_size, num_heads, seq_len, head_dim)
            k1 = random_tensor(batch_size, num_heads, seq_len, head_dim)
            k2 = random_tensor(batch_size, num_heads, seq_len, head_dim)
            v = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
            lambda_scale = random_tensor(1)

            dout = torch.rand_like(v)

            y1 = fn1(q1, q2, k1, k2, v, lambda_scale=lambda_scale, rms_norm=rms_norm)
            y2 = fn2(q1, q2, k1, k2, v, lambda_scale=lambda_scale, rms_norm=rms_norm)

            assert torch.allclose(y1, y2, atol=1e-2, rtol=0)

            if mode == "fwd+bwd":
                y1.backward(dout)

                y1_dv, v.grad = v.grad.clone(), None
                y1_dk1, k1.grad = k1.grad.clone(), None
                y1_dk2, k2.grad = k2.grad.clone(), None
                y1_dq1, q1.grad = q1.grad.clone(), None
                y1_dq2, q2.grad = q2.grad.clone(), None

                y2.backward(dout)

                y2_dv, v.grad = v.grad.clone(), None
                y2_dk1, k1.grad = k1.grad.clone(), None
                y2_dk2, k2.grad = k2.grad.clone(), None
                y2_dq1, q1.grad = q1.grad.clone(), None
                y2_dq2, q2.grad = q2.grad.clone(), None

                assert torch.allclose(y1_dv, y2_dv, atol=1e-2, rtol=0)
                assert torch.allclose(y1_dk1, y2_dk1, atol=1e-2, rtol=0)
                assert torch.allclose(y1_dk2, y2_dk2, atol=1e-2, rtol=0)
                assert torch.allclose(y1_dq1, y2_dq1, atol=1e-2, rtol=0)
                assert torch.allclose(y1_dq2, y2_dq2, atol=1e-2, rtol=0)

        else:
            q = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
            k = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)
            v = random_tensor(batch_size, num_heads, seq_len, 2 * head_dim)

            y1 = fn1(q, k, v, rms_norm=rms_norm)
            y2 = fn2(q, k, v, rms_norm=rms_norm)

            assert torch.allclose(y1, y2, atol=1e-2, rtol=0)

            if mode == "fwd+bwd":
                y1.backward(dout)

                y1_dv, v.grad = v.grad.clone(), None
                y1_dk, k.grad = k.grad.clone(), None
                y1_dq, q.grad = q.grad.clone(), None

                y2.backward(dout)

                y2_dv, v.grad = v.grad.clone(), None
                y2_dk, k.grad = k.grad.clone(), None
                y2_dq, q.grad = q.grad.clone(), None

                assert torch.allclose(y1_dv, y2_dv, atol=1e-2, rtol=0)
                assert torch.allclose(y1_dk, y2_dk, atol=1e-2, rtol=0)
                assert torch.allclose(y1_dq, y2_dq, atol=1e-2, rtol=0)

        diff[i] = torch.abs(y1 - y2).mean().item()

    os.makedirs(save_path, exist_ok=True)

    np.savetxt(os.path.join(save_path, f"{benchmark_name}.csv"), diff, delimiter=",")


if __name__ == "__main__":

    print("### Note: Disable Triton Autotune for accurate peak memory measurements ###")
    if MODE == "fwd+bwd":
        print("### Forward + Backward Pass Benchmark ###")
        folder = f"{FOLDER_PREFIX}_backward_{NUM_HEADS}_{HEAD_DIM}"
    else:
        print("### Forward Pass Benchmark ###")
        folder = f"{FOLDER_PREFIX}_forward_{NUM_HEADS}_{HEAD_DIM}"

    if RMS_NORM:
        print("### RMS Normalization Benchmark ###")
        folder += "_RMS_norm"

    benchmark_difference(save_path=f"./results/{folder}/difference/")
    benchmark_memory_usage.run(print_data=True, save_path=f"./results/{folder}/memory/")
    benchmark_runtime.run(print_data=True, save_path=f"./results/{folder}/runtime/")
