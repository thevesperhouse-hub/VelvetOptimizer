"""VesperLM — PyTorch training framework with custom CUDA/Triton kernels."""

__version__ = "0.1.0"

from .config import VesperConfig
from .model import VesperLM
from .optimizer import VelvetOptimizer
