#!/usr/bin/env python3
"""Plot VelvetOptimizer benchmark results.

Usage:
    python plot_benchmark.py benchmark_report.json [output.png]
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmark.py <report.json> [output.png]")
        sys.exit(1)

    report_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else report_path.with_suffix(".png")

    with open(report_path) as f:
        report = json.load(f)

    velvet = report["velvet"]
    adamw = report["adamw"]
    comparison = report["comparison"]

    # Extract step metrics
    v_steps = [m["step"] for m in velvet["step_metrics"]]
    v_losses = [m["loss"] for m in velvet["step_metrics"]]
    a_steps = [m["step"] for m in adamw["step_metrics"]]
    a_losses = [m["loss"] for m in adamw["step_metrics"]]

    # Velvet-specific metrics
    v_entropy = [m.get("entropy_scale") for m in velvet["step_metrics"]]
    v_ppl_scale = [m.get("ppl_scale") for m in velvet["step_metrics"]]
    v_grad_norm = [m.get("grad_norm") for m in velvet["step_metrics"]]

    has_entropy = any(x is not None for x in v_entropy)
    has_grad_norm = any(x is not None for x in v_grad_norm)

    # Determine subplot layout
    n_plots = 2  # loss + perplexity always
    if has_entropy:
        n_plots += 1
    if has_grad_norm:
        n_plots += 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(
        f"VelvetOptimizer Benchmark — {velvet['model_size']} | {velvet['epochs']} epochs",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plot_idx = 0

    # --- Plot 1: Per-step loss ---
    ax = axes[plot_idx]
    ax.plot(a_steps, a_losses, alpha=0.15, color="tab:blue", linewidth=0.5)
    ax.plot(v_steps, v_losses, alpha=0.15, color="tab:green", linewidth=0.5)
    if len(a_losses) > 5:
        ax.plot(a_steps, smooth(a_losses), color="tab:blue", linewidth=2, label="AdamW (smoothed)")
    if len(v_losses) > 5:
        ax.plot(v_steps, smooth(v_losses), color="tab:green", linewidth=2, label="Velvet (smoothed)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # --- Plot 2: Perplexity per epoch ---
    ax = axes[plot_idx]
    epochs_range = list(range(1, len(velvet["perplexity_history"]) + 1))
    ax.plot(epochs_range, adamw["perplexity_history"], "o-", color="tab:blue",
            linewidth=2, markersize=8, label="AdamW")
    ax.plot(epochs_range, velvet["perplexity_history"], "o-", color="tab:green",
            linewidth=2, markersize=8, label="Velvet")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity per Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate final values
    for data, color, name in [(velvet, "tab:green", "Velvet"), (adamw, "tab:blue", "AdamW")]:
        if data["perplexity_history"]:
            final_ppl = data["perplexity_history"][-1]
            ax.annotate(f"{final_ppl:.1f}", (len(data["perplexity_history"]), final_ppl),
                        textcoords="offset points", xytext=(10, 0),
                        fontsize=10, fontweight="bold", color=color)
    plot_idx += 1

    # --- Plot 3: Adaptive scales (Velvet only) ---
    if has_entropy:
        ax = axes[plot_idx]
        entropy_vals = [x for x in v_entropy if x is not None]
        ppl_vals = [x for x in v_ppl_scale if x is not None]
        steps_e = v_steps[:len(entropy_vals)]
        steps_p = v_steps[:len(ppl_vals)]

        if entropy_vals:
            ax.plot(steps_e, smooth(entropy_vals, 0.95), color="tab:orange",
                    linewidth=1.5, label="Entropy Scale")
        if ppl_vals:
            ax.plot(steps_p, smooth(ppl_vals, 0.95), color="tab:purple",
                    linewidth=1.5, label="Perplexity Scale")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1.0)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Scale Factor")
        ax.set_title("Velvet Adaptive Scales")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # --- Plot 4: Gradient norms ---
    if has_grad_norm:
        ax = axes[plot_idx]
        grad_vals = [x for x in v_grad_norm if x is not None]
        steps_g = v_steps[:len(grad_vals)]

        if grad_vals:
            ax.plot(steps_g, grad_vals, alpha=0.2, color="tab:red", linewidth=0.5)
            ax.plot(steps_g, smooth(grad_vals, 0.95), color="tab:red",
                    linewidth=1.5, label="Velvet Grad Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norms (Velvet)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Add comparison text box
    textstr = (
        f"Loss improvement: {comparison['loss_improvement_pct']:.1f}%\n"
        f"Perplexity improvement: {comparison['perplexity_improvement_pct']:.1f}%\n"
        f"Speed difference: {comparison['speed_difference_pct']:.1f}%\n"
        f"Velvet: {comparison['velvet_final_loss']:.4f} loss / {comparison['velvet_final_ppl']:.1f} ppl\n"
        f"AdamW:  {comparison['adamw_final_loss']:.4f} loss / {comparison['adamw_final_ppl']:.1f} ppl"
    )
    fig.text(0.02, 0.01, textstr, fontsize=9, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
