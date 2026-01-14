"""
Velvet GPU Optimizer - Setup Script

Build and install Velvet as a PyTorch extension.

Installation:
    pip install -e .

Requirements:
    - PyTorch with CUDA support
    - CUDA Toolkit 11.0+
    - C++17 compiler
    - cuBLAS

The Vesper House
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# Check CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Velvet requires CUDA to run.")
    print("Please install PyTorch with CUDA support.")
    sys.exit(1)

if CUDA_HOME is None:
    print("ERROR: CUDA_HOME is not set. Please install CUDA Toolkit.")
    sys.exit(1)

print(f"CUDA Home: {CUDA_HOME}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Project root
ROOT_DIR = Path(__file__).parent.absolute()

# Source files
EXTENSION_SOURCES = [
    'python/csrc/bindings.cpp',
    'src/tensor.cpp',
    'src/cuda_mem.cu',
    'src/velvet_cuda.cu',
]

# Include directories
INCLUDE_DIRS = [
    str(ROOT_DIR / 'include'),
    str(ROOT_DIR / 'python' / 'csrc'),
]

# Library directories
LIBRARY_DIRS = []

# Libraries to link
LIBRARIES = ['cublas']

# Compiler flags
EXTRA_COMPILE_ARGS = {
    'cxx': [
        '-O3',
        '-std=c++17',
    ],
    'nvcc': [
        '-O3',
        '--std=c++17',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-use_fast_math',
    ]
}

# Auto-detect CUDA architecture
# Get compute capability from current GPU
try:
    major, minor = torch.cuda.get_device_capability()
    compute_capability = f"{major}{minor}"
    arch_flags = [
        f'-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}',
        f'-gencode=arch=compute_{compute_capability},code=compute_{compute_capability}',
    ]
    EXTRA_COMPILE_ARGS['nvcc'].extend(arch_flags)
    print(f"Auto-detected GPU compute capability: {major}.{minor}")
except Exception as e:
    print(f"Warning: Could not auto-detect GPU architecture: {e}")
    print("Using default CUDA architectures (may be slower)")

# Create extension
ext_modules = [
    CUDAExtension(
        name='velvet._velvet_cpp',
        sources=EXTENSION_SOURCES,
        include_dirs=INCLUDE_DIRS,
        library_dirs=LIBRARY_DIRS,
        libraries=LIBRARIES,
        extra_compile_args=EXTRA_COMPILE_ARGS,
    )
]

# Read README for long description
README_PATH = ROOT_DIR / 'README.md'
if README_PATH.exists():
    with open(README_PATH, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Velvet GPU Optimizer - High-performance optimizer for PyTorch"

# Setup configuration
setup(
    name='velvet-optimizer',
    version='1.0.0',
    author='The Vesper House',
    description='High-performance GPU optimizer for PyTorch with custom CUDA kernels',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/thevesperhouse-hub/Velvet',
    license='MIT',

    # Package configuration
    packages=['velvet'],
    package_dir={'velvet': 'python'},

    # Extension modules
    ext_modules=ext_modules,

    # Build command
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },

    # Requirements
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
    ],

    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    # Keywords
    keywords='pytorch optimizer cuda gpu deep-learning machine-learning adam',

    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/thevesperhouse-hub/Velvet/issues',
        'Source': 'https://github.com/thevesperhouse-hub/Velvet',
    },
)

print("\n" + "="*70)
print("Velvet GPU Optimizer - Build Configuration")
print("="*70)
print(f"CUDA Home: {CUDA_HOME}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"Sources: {len(EXTENSION_SOURCES)} files")
print(f"Include dirs: {len(INCLUDE_DIRS)}")
print("="*70 + "\n")
