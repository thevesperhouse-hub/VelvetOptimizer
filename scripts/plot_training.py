#!/usr/bin/env python3
"""Plot training loss from training_log.json files.

Usage:
    # Single run
    python plot_training.py test-checkpoints/training_log.json

    # Compare Velvet vs AdamW (two separate training runs)
    python plot_training.py velvet-checkpoints/training_log.json adamw-checkpoints/training_log.json

    # Custom output
    python plot_training.py logs/*.json -o results.png
"""

import json
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("logs", nargs="+", help="Path(s) to training_log.json files")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path (default: auto)")
    args = parser.parse_args()

    colors = ["tab:green", "tab:blue", "tab:orange", "tab:red"]
    n_logs = len(args.logs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for i, log_path in enumerate(args.logs):
        with open(log_path) as f:
            log = json.load(f)

        optimizer = log.get("optimizer", f"Run {i+1}")
        model_size = log.get("model_size", "?")
        steps_data = log["steps"]

        steps = [s["step"] for s in steps_data]
        losses = [s["loss"] for s in steps_data]
        color = colors[i % len(colors)]

        # Plot 1: Loss per step
        ax = axes[0]
        ax.plot(steps, losses, alpha=0.15, color=color, linewidth=0.5)
        if len(losses) > 5:
            ax.plot(steps, smooth(losses), color=color, linewidth=2,
                    label=f"{optimizer} ({model_size})")

        # Plot 2: Perplexity per step (smoothed)
        ax2 = axes[1]
        ppls = [np.exp(l) for l in losses]
        if len(ppls) > 5:
            ax2.plot(steps, smooth(ppls, 0.95), color=color, linewidth=2,
                     label=f"{optimizer} ({model_size})")

    # Format loss plot
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Format perplexity plot
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Training Perplexity (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Summary text
    for i, log_path in enumerate(args.logs):
        with open(log_path) as f:
            log = json.load(f)
        steps_data = log["steps"]
        if steps_data:
            final_loss = steps_data[-1]["loss"]
            final_ppl = np.exp(final_loss)
            total_steps = steps_data[-1]["step"]
            name = log.get("optimizer", f"Run {i+1}")
            fig.text(0.02, 0.01 + i * 0.02,
                     f"{name}: {total_steps} steps | final loss: {final_loss:.4f} | final ppl: {final_ppl:.1f}",
                     fontsize=9, fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.02 + n_logs * 0.02, 1, 1])

    output = args.output or str(Path(args.logs[0]).with_suffix(".png"))
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output}")


if __name__ == "__main__":
    main()
