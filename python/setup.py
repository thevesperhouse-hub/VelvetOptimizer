"""VesperLM — PyTorch training framework with custom CUDA/Triton kernels."""

from setuptools import setup, find_packages

setup(
    name="vesper",
    version="0.1.0",
    description="VesperLM training framework with Velvet optimizer, ERA activation, and MoE",
    author="The Vesper House",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # NOTE: torch with CUDA must be installed separately:
        #   pip install torch --index-url https://download.pytorch.org/whl/cu128
        # The base torch requirement here may install CPU-only.
        "torch>=2.10.0",
        "tokenizers>=0.22.2",
        "tqdm",
    ],
    extras_require={
        "triton": ["triton>=3.6.0"],
        "flash-attn": ["flash-attn>=2.8.3"],
        "wandb": ["wandb"],
        "all": [
            "triton>=3.6.0",
            "flash-attn>=2.8.3",
            "huggingface-hub>=1.4.1",
            "wandb",
        ],
    },
)
