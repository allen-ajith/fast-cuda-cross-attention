"""
Benchmark script for CUDA kernel implementations across multiple batch sizes
Tests all kernels across different batch sizes and N_latent values
"""

import numpy as np
import time
import os
import argparse

try:
    from cuda_wrappers import (
        run_naive_kernel,
        run_warp_parallel_kernel,
        run_tiled_kernel,
        run_vectorized_kernel
    )
    CUDA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CUDA kernels not available: {e}")
    print("Run 'make' to compile the kernels.")
    CUDA_AVAILABLE = False

N_input = 784
D = 768
DATA_DIR = 'data'
BATCHES_DIR = 'batches'

BATCH_SIZES = [4, 8, 16, 32]

N_latent_values = [64, 128, 256, 512, 1024]

def verify_output(output, reference, name, tolerance=5e-3):
    """Verify output matches reference.

    Uses a more lenient tolerance (5e-3) to account for floating-point
    differences across different GPU architectures (e.g., RTX 4070 vs Titan V).
    Different architectures may have slightly different implementations of
    expf(), sqrtf(), and accumulation order, leading to small numerical differences.
    """
    max_diff = np.max(np.abs(output - reference))
    mean_diff = np.mean(np.abs(output - reference))

    if max_diff > tolerance:
        print(f"  {name}: WARNING - Max difference {max_diff:.6f} exceeds tolerance {tolerance}")
        print(f"    Mean difference: {mean_diff:.6f}")
        return False
    else:
        print(f"  {name}: Verification passed (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f})")
        return True

def benchmark_kernel(kernel_func, Q, K, V, N_latent, N_input, D, num_iterations=100, warmup=10):
    """Benchmark a single kernel."""
    for _ in range(warmup):
        _ = kernel_func(Q, K, V, N_latent, N_input, D)

    start = time.time()
    for _ in range(num_iterations):
        output = kernel_func(Q, K, V, N_latent, N_input, D)
    elapsed = (time.time() - start) / num_iterations * 1000

    return output, elapsed

def benchmark_naive(Q, K, V, N_latent, N_input, D):
    """Benchmark naive CUDA kernel."""
    for _ in range(3):
        _ = run_naive_kernel(Q, K, V, N_latent, N_input, D)

    start = time.time()
    for _ in range(10):
        output = run_naive_kernel(Q, K, V, N_latent, N_input, D)
    elapsed = (time.time() - start) / 10 * 1000

    return output, elapsed

def load_batch_reference_data(BATCH_SIZE, N_latent):
    """Load Q, K, V matrices and reference output from batch subdirectory."""
    batch_dir = os.path.join(DATA_DIR, BATCHES_DIR, f'batch_{BATCH_SIZE}')
    subdir = os.path.join(batch_dir, f'N_latent_{N_latent}')
    q_path = os.path.join(subdir, 'Q_matrix.npy')

    if not os.path.exists(q_path):
        return None, None, None, None

    Q = np.load(os.path.join(subdir, 'Q_matrix.npy'))
    K = np.load(os.path.join(subdir, 'K_matrix.npy'))
    V = np.load(os.path.join(subdir, 'V_matrix.npy'))
    reference = np.load(os.path.join(subdir, 'output_reference.npy'))

    return Q, K, V, reference

def run_batch_benchmark():
    """Run benchmark testing all kernels across all batch sizes and N_latent values."""
    print("="*80)
    print("Perceiver Cross-Attention CUDA Kernel Benchmark - Batch Size Analysis")
    print("="*80)
    print(f"Fixed dimensions:")
    print(f"  N_input:  {N_input}")
    print(f"  D:         {D}")
    print(f"\nBatch sizes: {BATCH_SIZES}")
    print(f"N_latent values: {N_latent_values}")
    print("Kernels: Naive, Warp Parallel, Tiled, Vectorized")
    print("Reference verification: Using pre-generated references from data/batches/ subdirectories")
    print("="*80)

    if not CUDA_AVAILABLE:
        print("\nCUDA kernels not available. Cannot run benchmarks.")
        return

    missing_data = []
    for BATCH_SIZE in BATCH_SIZES:
        for N_latent in N_latent_values:
            ref_path = os.path.join(DATA_DIR, BATCHES_DIR, f'batch_{BATCH_SIZE}', f'N_latent_{N_latent}', 'output_reference.npy')
            if not os.path.exists(ref_path):
                missing_data.append((BATCH_SIZE, N_latent))

    if missing_data:
        print(f"\nError: Pre-generated batch reference data not found for:")
        for BATCH_SIZE, N_latent in missing_data:
            print(f"  Batch size {BATCH_SIZE}, N_latent {N_latent}")
        print(f"Please run 'python generate_batch_references.py' to generate the required data.")
        return

    all_results = {}
    for BATCH_SIZE in BATCH_SIZES:
        all_results[BATCH_SIZE] = {
            'N_latent': [],
            'naive': [],
            'parallel': [],
            'tiled': [],
            'optimized': []
        }

    for BATCH_SIZE in BATCH_SIZES:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE = {BATCH_SIZE}")
        print(f"{'='*80}")

        for N_latent in N_latent_values:
            print(f"\n  N_latent = {N_latent}")
            print(f"  {'-'*76}")

            Q, K, V, reference = load_batch_reference_data(BATCH_SIZE, N_latent)

            if Q is None:
                print(f"    Error: Pre-generated data not found")
                all_results[BATCH_SIZE]['naive'].append(None)
                all_results[BATCH_SIZE]['parallel'].append(None)
                all_results[BATCH_SIZE]['tiled'].append(None)
                all_results[BATCH_SIZE]['optimized'].append(None)
                all_results[BATCH_SIZE]['N_latent'].append(N_latent)
                continue

            print(f"    Loaded data: Q{Q.shape}, K{K.shape}, V{V.shape}, Ref{reference.shape}")

            print(f"    Naive...")
            try:
                output_naive, time_naive = benchmark_naive(Q, K, V, N_latent, N_input, D)
                all_results[BATCH_SIZE]['naive'].append(time_naive)
                print(f"      Time: {time_naive:.3f} ms")
                print(f"      Output shape: {output_naive.shape}")
                print(f"      Reference shape: {reference.shape}")
                verify_output(output_naive, reference, "Naive", tolerance=1e-3)
            except Exception as e:
                print(f"      Error: {e}")
                all_results[BATCH_SIZE]['naive'].append(None)

            print(f"    Warp Parallel...")
            try:
                output_parallel, time_parallel = benchmark_kernel(
                    run_warp_parallel_kernel, Q, K, V, N_latent, N_input, D
                )
                all_results[BATCH_SIZE]['parallel'].append(time_parallel)
                speedup = time_naive / time_parallel if time_naive is not None and time_naive > 0 else 0
                print(f"      Time: {time_parallel:.3f} ms (Speedup: {speedup:.2f})")
                verify_output(output_parallel, reference, "Warp Parallel", tolerance=1e-3)
            except Exception as e:
                print(f"      Error: {e}")
                all_results[BATCH_SIZE]['parallel'].append(None)

            print(f"    Tiled...")
            try:
                output_tiled, time_tiled = benchmark_kernel(
                    run_tiled_kernel, Q, K, V, N_latent, N_input, D
                )
                all_results[BATCH_SIZE]['tiled'].append(time_tiled)
                speedup = time_naive / time_tiled if time_naive is not None and time_naive > 0 else 0
                print(f"      Time: {time_tiled:.3f} ms (Speedup: {speedup:.2f})")
                verify_output(output_tiled, reference, "Tiled", tolerance=1e-3)
            except Exception as e:
                print(f"      Error: {e}")
                all_results[BATCH_SIZE]['tiled'].append(None)

            print(f"    Vectorized...")
            try:
                output_optimized, time_optimized = benchmark_kernel(
                    run_vectorized_kernel, Q, K, V, N_latent, N_input, D
                )
                all_results[BATCH_SIZE]['optimized'].append(time_optimized)
                speedup = time_naive / time_optimized if time_naive is not None and time_naive > 0 else 0
                print(f"      Time: {time_optimized:.3f} ms (Speedup: {speedup:.2f})")
                verify_output(output_optimized, reference, "Vectorized", tolerance=1e-3)
            except Exception as e:
                print(f"      Error: {e}")
                all_results[BATCH_SIZE]['optimized'].append(None)

            all_results[BATCH_SIZE]['N_latent'].append(N_latent)

    for BATCH_SIZE in BATCH_SIZES:
        print(f"\n{'='*80}")
        print(f"Performance Summary - Batch Size {BATCH_SIZE}")
        print(f"{'='*80}")
        print(f"{'N_latent':<12} {'Naive (ms)':<15} {'Warp Parallel (ms)':<20} {'Tiled (ms)':<15} {'Vectorized (ms)':<18}")
        print("-"*80)

        results = all_results[BATCH_SIZE]
        for i, n_latent in enumerate(results['N_latent']):
            naive = results['naive'][i] if results['naive'][i] is not None else float('inf')
            parallel = results['parallel'][i] if results['parallel'][i] is not None else float('inf')
            tiled = results['tiled'][i] if results['tiled'][i] is not None else float('inf')
            optimized = results['optimized'][i] if results['optimized'][i] is not None else float('inf')

            print(f"{n_latent:<12} {naive:<15.3f} {parallel:<20.3f} {tiled:<15.3f} {optimized:<18.3f}")

        print(f"\nThroughput (samples/sec) - Batch Size {BATCH_SIZE}")
        print(f"{'='*80}")
        print(f"{'N_latent':<12} {'Naive':<15} {'Warp Parallel':<18} {'Tiled':<15} {'Vectorized':<18}")
        print("-"*80)

        for i, n_latent in enumerate(results['N_latent']):
            naive = results['naive'][i]
            parallel = results['parallel'][i]
            tiled = results['tiled'][i]
            optimized = results['optimized'][i]

            throughput_naive = (BATCH_SIZE * 1000 / naive) if naive is not None and naive > 0 else 0
            throughput_parallel = (BATCH_SIZE * 1000 / parallel) if parallel is not None and parallel > 0 else 0
            throughput_tiled = (BATCH_SIZE * 1000 / tiled) if tiled is not None and tiled > 0 else 0
            throughput_optimized = (BATCH_SIZE * 1000 / optimized) if optimized is not None and optimized > 0 else 0

            print(f"{n_latent:<12} {throughput_naive:<15.1f} {throughput_parallel:<18.1f} {throughput_tiled:<15.1f} {throughput_optimized:<18.1f}")

        print(f"\nSpeedup vs Naive - Batch Size {BATCH_SIZE}")
        print(f"{'='*80}")
        print(f"{'N_latent':<12} {'Warp Parallel':<18} {'Tiled':<15} {'Vectorized':<18}")
        print("-"*80)

        for i, n_latent in enumerate(results['N_latent']):
            naive = results['naive'][i]
            parallel = results['parallel'][i]
            tiled = results['tiled'][i]
            optimized = results['optimized'][i]

            if naive is not None and naive > 0:
                speedup_parallel = naive / parallel if parallel is not None else 0
                speedup_tiled = naive / tiled if tiled is not None else 0
                speedup_optimized = naive / optimized if optimized is not None else 0

                print(f"{n_latent:<12} {speedup_parallel:<18.2f} {speedup_tiled:<15.2f} {speedup_optimized:<18.2f}")

    print("="*80)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark CUDA kernel implementations across multiple batch sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This benchmark tests all 4 kernel implementations (Naive, Warp Parallel, Tiled, Vectorized)
across multiple batch sizes and N_latent values to show batch processing performance.

Example:
  python benchmark_batching.py
        """
    )

    args = parser.parse_args()
    run_batch_benchmark()

if __name__ == "__main__":
    main()
