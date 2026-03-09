"""
Roofline benchmark for fused MoE triton kernel.

Measures kernel time across batch sizes and TP values to visualize
SM utilization bottleneck for small-batch MoE workloads.

Usage:
    python roofline_fused_moe_triton.py \
        --hidden-size 4096 --moe-intermediate-size 1024 \
        --num-experts 8 --topk 2 --dtype both \
        --save-path ./roofline_results/
"""

import argparse
import os
from typing import List, Optional

import torch
import triton

from sglang.srt.layers.moe.fused_moe_triton import override_config
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import is_hip

_is_hip = is_hip()


def get_best_config(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype_str: str,
    block_shape: Optional[List[int]] = None,
):
    """Get the best config for the given shape, falling back to default."""
    op_config = get_moe_configs(
        num_experts,
        shard_intermediate_size // 2,
        dtype_str,
        0,
        0,
        False,
    )
    if op_config is not None:
        return op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]
    return get_default_config(
        num_tokens,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype_str,
        False,
        block_shape,
    )


def benchmark_one(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8: bool,
) -> Optional[float]:
    """Benchmark a single configuration. Returns kernel time in us, or None on OOM."""
    try:
        init_dtype = torch.float16 if use_fp8 else dtype
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
        w1 = torch.randn(
            num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device="cuda"
        )
        w2 = torch.randn(
            num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype, device="cuda"
        )

        w1_scale, w2_scale, a1_scale, a2_scale = None, None, None, None
        if use_fp8:
            fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w1 = w1.to(fp8_dtype)
            w2 = w2.to(fp8_dtype)
            w1_scale = torch.ones(num_experts, dtype=torch.float32, device="cuda")
            w2_scale = torch.ones(num_experts, dtype=torch.float32, device="cuda")
            a1_scale = torch.ones(1, dtype=torch.float32, device="cuda")
            a2_scale = torch.ones(1, dtype=torch.float32, device="cuda")

        dtype_str = get_config_dtype_str(
            use_fp8_w8a8=use_fp8, dtype=dtype
        )
        config = get_best_config(
            num_tokens, num_experts, shard_intermediate_size,
            hidden_size, topk, dtype_str,
        )

        topk_config = TopKConfig(top_k=topk, renormalize=True)
        input_gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
        topk_output = select_experts(x, input_gating, topk_config)
        moe_runner_config = MoeRunnerConfig(inplace=True)

        def run():
            with override_config(config):
                fused_moe(
                    x, w1, w2, topk_output,
                    moe_runner_config=moe_runner_config,
                    use_fp8_w8a8=use_fp8,
                    w1_scale=w1_scale, w2_scale=w2_scale,
                    a1_scale=a1_scale, a2_scale=a2_scale,
                )

        # Warmup
        run()
        torch.cuda.synchronize()

        # Benchmark
        ms = triton.testing.do_bench(run, warmup=10, rep=50)
        return ms * 1000  # convert to us

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise


def run_benchmark(args):
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    tp_sizes = [1, 2, 4]
    dtypes = []
    if args.dtype in ("bf16", "both"):
        dtypes.append(("bf16", torch.bfloat16, False))
    if args.dtype in ("fp8", "both"):
        dtypes.append(("fp8", torch.bfloat16, True))

    results = {}  # (dtype_name, tp) -> {batch: time_us}

    for dtype_name, dtype, use_fp8 in dtypes:
        for tp in tp_sizes:
            shard_intermediate_size = 2 * args.moe_intermediate_size // tp
            key = f"{dtype_name}_tp{tp}"
            results[key] = {}
            print(f"\n--- {key}: N={shard_intermediate_size}, K={args.hidden_size} ---")

            for bs in batch_sizes:
                time_us = benchmark_one(
                    bs, args.num_experts, shard_intermediate_size,
                    args.hidden_size, args.topk, dtype, use_fp8,
                )
                results[key][bs] = time_us
                status = f"{time_us:.1f} us" if time_us is not None else "OOM"
                print(f"  batch={bs:>3d}: {status}")
                torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary (kernel time in us)")
    print("=" * 80)
    header = f"{'batch':>6s}"
    for key in results:
        header += f" | {key:>12s}"
    print(header)
    print("-" * len(header))
    for bs in batch_sizes:
        row = f"{bs:>6d}"
        for key in results:
            val = results[key].get(bs)
            row += f" | {val:>12.1f}" if val is not None else f" | {'OOM':>12s}"
        print(row)

    # Plot
    if args.save_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(args.save_path, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            markers = ["o", "s", "^", "D", "v", "p"]
            linestyles = ["-", "--"]

            for idx, (key, data) in enumerate(results.items()):
                xs = [bs for bs in batch_sizes if data.get(bs) is not None]
                ys = [data[bs] for bs in xs]
                if xs:
                    dtype_idx = 0 if "bf16" in key else 1
                    ax.plot(
                        xs, ys,
                        marker=markers[idx % len(markers)],
                        linestyle=linestyles[dtype_idx],
                        label=key, linewidth=2, markersize=8,
                    )

            ax.set_xlabel("Batch Size", fontsize=12)
            ax.set_ylabel("Kernel Time (us)", fontsize=12)
            ax.set_title(
                f"Fused MoE Triton Roofline\n"
                f"hidden={args.hidden_size}, moe_inter={args.moe_intermediate_size}, "
                f"E={args.num_experts}, topk={args.topk}",
                fontsize=13,
            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            save_file = os.path.join(args.save_path, "roofline_fused_moe.png")
            fig.savefig(save_file, dpi=150, bbox_inches="tight")
            print(f"\nPlot saved to {save_file}")
            plt.close(fig)
        except ImportError:
            print("\nmatplotlib not available, skipping plot generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roofline benchmark for fused MoE triton kernel")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--moe-intermediate-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--dtype", type=str, choices=["bf16", "fp8", "both"], default="both",
        help="Data type to benchmark",
    )
    parser.add_argument("--save-path", type=str, default=None, help="Directory to save plot")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    run_benchmark(args)
