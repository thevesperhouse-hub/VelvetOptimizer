"""
Velvet GPU Optimizer

High-performance GPU optimizer for PyTorch with custom CUDA kernels.
Approximately 36% faster than standard Adam on RTX GPUs.

Usage:
    from velvet import VelvetOptimizer

    optimizer = VelvetOptimizer(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

The Vesper House
"""

__version__ = "1.0.0"
__author__ = "The Vesper House"

from .velvet_optimizer import VelvetOptimizer, Velvet

__all__ = [
    "VelvetOptimizer",
    "Velvet",
]
