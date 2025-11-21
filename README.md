# CUDA Cross-Attention Optimization

High-performance CUDA implementations of cross-attention with progressive optimizations and batch processing support. Optimizes the standard cross-attention operation `softmax(Q @ K^T / sqrt(d)) @ V` where Q queries attend to K/V keys and values. Demonstrates kernel optimization techniques from naive baseline to fully vectorized implementations, with benchmarking for both single and multi-batch cases.

The project originated from optimizing the Perceiver architecture, where cross-attention between latent queries and input keys/values was identified as the primary performance bottleneck. Rather than optimizing the entire Perceiver pipeline, we focused on developing highly optimized CUDA kernels for the cross-attention operation itself. While the kernels are general-purpose and can be used in any cross-attention context, they are particularly well-suited for Perceiver-style architectures where latent vectors attend to large input sequences.

## Features

- **4 Progressive Kernel Implementations**:
  - `kernel_naive.cu` - Baseline (one thread per output element)
    - *Logic*: Each thread computes one output element. For each latent vector, computes Q·K^T scores for all input keys, applies softmax, then weighted sum with V. No parallelism or memory optimization.

  - `kernel_warp_parallel.cu` - Warp-level cooperation
    - *Logic*: Each warp (32 threads) processes one latent vector. Threads cooperate using warp shuffles to compute dot products in parallel. Q is distributed across threads, K/V are loaded per key, results reduced via shuffle operations.

  - `kernel_tiled.cu` - Shared memory tiling + online softmax
    - *Logic*: Similar to warp parallel, but caches K in shared memory tiles (16 keys at a time). Implements online softmax with running max/sum to avoid storing all scores, improving memory locality and reducing register pressure.

  - `kernel_vectorized.cu` - Full optimization (vectorization + all techniques)
    - *Logic*: Combines tiling with float4 vectorization (4 floats per load). Loads Q, K, V as vectorized types to reduce memory transactions by 4x. Uses online softmax and tiled processing for maximum throughput.

- **Batch Processing Support**: All kernels support arbitrary batch sizes
- **Comprehensive Benchmarking**:
  - Single batch benchmarking (`benchmark.py`)
  - Multi-batch benchmarking (`benchmark_batching.py`) across batch sizes [4, 8, 16, 32]
- **Output Verification**: Automatic correctness checking against PyTorch reference
- **Performance Analysis**: Detailed timing, speedup, scaling analysis, and throughput metrics

## Quick Start

### Prerequisites

- CUDA Toolkit (tested with CUDA 13.0)
- Python 3.x with NumPy
- PyTorch (required for generating reference data)

### Build

```bash
make
```

This compiles all CUDA kernels into `kernels.so`.

### Generate Reference Data

**Note**: Reference data files are not included in the repository. You must generate them before running benchmarks.

**Single Batch References:**
```bash
pip install torch numpy
python generate_references.py
```
Creates reference outputs in `data/N_latent_*/` directories.

**Batch References (for multi-batch benchmarking):**
```bash
python generate_batch_references.py
```
Creates reference outputs in `data/batches/batch_*/N_latent_*/` directories for batch sizes [4, 8, 16, 32].

### Run Benchmarks

**Single Batch (batch_size=1):**
```bash
python benchmark.py
```

**Multi-Batch Analysis:**
```bash
python benchmark_batching.py
```

The benchmarks will:
- Test all 4 kernel implementations
- Verify outputs against reference data
- Display performance summary and speedups
- Show scaling analysis (time per latent vector)
- Multi-batch benchmark includes throughput analysis (samples/sec)

**Note**: Make sure you've generated the required reference data before running benchmarks.

## Project Structure

```
.
├── kernel_naive.cu              # Baseline CUDA kernel
├── kernel_warp_parallel.cu      # Warp-parallel optimization
├── kernel_tiled.cu              # Tiled + online softmax
├── kernel_vectorized.cu         # Fully optimized kernel
├── cuda_wrappers.py             # Python wrappers (ctypes, batch support)
├── benchmark.py                 # Single batch benchmarking (batch_size=1)
├── benchmark_batching.py        # Multi-batch benchmarking
├── generate_references.py       # Single batch reference generator
├── generate_batch_references.py # Multi-batch reference generator
├── Makefile                     # Build configuration
└── requirements.txt             # Python dependencies
```

## Requirements

- **Runtime**: `numpy>=1.19.0`
- **Reference Generation**: `torch`, `numpy`
- **CUDA**: Compatible GPU with CUDA support
- **Tested Hardware**: RTX GPUs (tested on RTX machines with CUDA 13.0)

## Usage

The benchmarks test cross-attention computation:
- **Operation**: `softmax(Q @ K^T / sqrt(d)) @ V`
- **Input**: Q [batch_size, N_latent, D], K [batch_size, N_input, D], V [batch_size, N_input, D]
- **Output**: Cross-Attention(Q, K, V) [batch_size, N_latent, D]
- **Fixed dimensions**: N_input=784, D=768
- **Variable**:
  - N_latent ∈ {64, 128, 256, 512, 1024} (number of query vectors)
  - Batch sizes: 1 (single batch) or [4, 8, 16, 32] (multi-batch)

**Single Batch Benchmark** (`benchmark.py`):
- Tests batch_size=1 across all N_latent values
- Results: execution time, speedup, scaling analysis

**Multi-Batch Benchmark** (`benchmark_batching.py`):
- Tests all batch sizes [4, 8, 16, 32] across all N_latent values
- Results: execution time, speedup, throughput (samples/sec), scaling analysis
- Demonstrates batch processing efficiency

All results include correctness verification against PyTorch references.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project was developed as part of **CSCI-GA 3033-025: Graphics Processing Units (GPUs): Architecture & Programming** (Fall 2025) coursework at NYU Courant.
