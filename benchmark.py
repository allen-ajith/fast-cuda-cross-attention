"""
Benchmark script for CUDA kernel implementations
Tests all kernels across multiple N_latent values with optimization and scaling analysis
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

N_latent_values = [64, 128, 256, 512, 1024]

def verify_output(output, reference, name, tolerance=5e-3):
    """Verify output matches reference.

    Uses a more lenient tolerance (5e-3) to account for floating-point
    differences
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

def load_reference_data(N_latent):
    """Load Q, K, V matrices and reference output from subdirectory."""
    subdir = os.path.join(DATA_DIR, f'N_latent_{N_latent}')
    q_path = os.path.join(subdir, 'Q_matrix.npy')

    if not os.path.exists(q_path):
        return None, None, None, None

    Q = np.load(os.path.join(subdir, 'Q_matrix.npy'))
    K = np.load(os.path.join(subdir, 'K_matrix.npy'))
    V = np.load(os.path.join(subdir, 'V_matrix.npy'))
    reference = np.load(os.path.join(subdir, 'output_reference.npy'))

    return Q, K, V, reference

def run_unified_benchmark():
    """Run unified benchmark testing all kernels across all N_latent values."""
    print("="*80)
    print("Perceiver Cross-Attention CUDA Kernel Benchmark")
    print("Optimization Comparison & Scaling Analysis")
    print("="*80)
    print(f"Fixed dimensions:")
    print(f"  N_input:  {N_input}")
    print(f"  D:         {D}")
    print(f"\nTesting N_latent values: {N_latent_values}")
    print("Kernels: Naive, Warp Parallel, Tiled, Vectorized")
    print("Reference verification: Using pre-generated references from data/ subdirectories")
    print("="*80)

    if not CUDA_AVAILABLE:
        print("\nCUDA kernels not available. Cannot run benchmarks.")
        return

    missing_refs = []
    for n in N_latent_values:
        ref_path = os.path.join(DATA_DIR, f'N_latent_{n}', 'output_reference.npy')
        if not os.path.exists(ref_path):
            missing_refs.append(n)

    if missing_refs:
        print(f"\nError: Pre-generated reference data not found for N_latent values: {missing_refs}")
        print(f"Please run 'python generate_references.py' to generate the required reference data.")
        print(f"\nExpected directory structure:")
        for n in missing_refs:
            print(f"  data/N_latent_{n}/")
            print(f"    Q_matrix.npy")
            print(f"    K_matrix.npy")
            print(f"    V_matrix.npy")
            print(f"    output_reference.npy")
        return

    all_results = {
        'N_latent': [],
        'naive': [],
        'parallel': [],
        'tiled': [],
        'optimized': []
    }

    for N_latent in N_latent_values:
        print(f"\n{'='*80}")
        print(f"N_latent = {N_latent}")
        print(f"{'='*80}")

        Q, K, V, reference = load_reference_data(N_latent)

        if Q is None:
            print(f"  Error: Pre-generated data not found for N_latent={N_latent}")
            print(f"  Expected: data/N_latent_{N_latent}/Q_matrix.npy")
            print(f"  Please run 'python generate_references.py' to generate the required data.")
            all_results['naive'].append(None)
            all_results['parallel'].append(None)
            all_results['tiled'].append(None)
            all_results['optimized'].append(None)
            all_results['N_latent'].append(N_latent)
            continue

        if len(reference.shape) == 2:
            reference = reference.reshape(1, N_latent, D)

        print(f"  Loaded pre-generated Q, K, V and reference from data/N_latent_{N_latent}/")

        print(f"\n  Naive...")
        try:
            output_naive, time_naive = benchmark_naive(Q, K, V, N_latent, N_input, D)
            all_results['naive'].append(time_naive)
            print(f"    Time: {time_naive:.3f} ms")
            print(f"    Output shape: {output_naive.shape}")
            print(f"    Reference shape: {reference.shape}")
            verify_output(output_naive, reference, "Naive", tolerance=5e-3)
        except Exception as e:
            print(f"    Error: {e}")
            all_results['naive'].append(None)

        print(f"  Warp Parallel...")
        try:
            output_parallel, time_parallel = benchmark_kernel(
                run_warp_parallel_kernel, Q, K, V, N_latent, N_input, D
            )
            all_results['parallel'].append(time_parallel)
            speedup = time_naive / time_parallel if time_naive is not None and time_naive > 0 else 0
            print(f"    Time: {time_parallel:.3f} ms")
            print(f"    Speedup over Naive: {speedup:.2f}")
            verify_output(output_parallel, reference, "Warp Parallel", tolerance=1e-3)
        except Exception as e:
            print(f"    Error: {e}")
            all_results['parallel'].append(None)

        print(f"  Tiled...")
        try:
            output_tiled, time_tiled = benchmark_kernel(
                run_tiled_kernel, Q, K, V, N_latent, N_input, D
            )
            all_results['tiled'].append(time_tiled)
            speedup = time_naive / time_tiled if time_naive is not None and time_naive > 0 else 0
            print(f"    Time: {time_tiled:.3f} ms")
            print(f"    Speedup over Naive: {speedup:.2f}")
            verify_output(output_tiled, reference, "Tiled", tolerance=1e-3)
        except Exception as e:
            print(f"    Error: {e}")
            all_results['tiled'].append(None)

        print(f"  Vectorized...")
        try:
            output_optimized, time_optimized = benchmark_kernel(
                run_vectorized_kernel, Q, K, V, N_latent, N_input, D
            )
            all_results['optimized'].append(time_optimized)
            speedup = time_naive / time_optimized if time_naive is not None and time_naive > 0 else 0
            print(f"    Time: {time_optimized:.3f} ms")
            print(f"    Speedup over Naive: {speedup:.2f}")
            verify_output(output_optimized, reference, "Vectorized", tolerance=1e-3)
        except Exception as e:
            print(f"    Error: {e}")
            all_results['optimized'].append(None)

        all_results['N_latent'].append(N_latent)

    print(f"\n{'='*80}")
    print("Performance Summary - All Kernels Across All N_latent Values")
    print(f"{'='*80}")
    print(f"{'N_latent':<12} {'Naive (ms)':<15} {'Warp Parallel (ms)':<20} {'Tiled (ms)':<15} {'Vectorized (ms)':<18}")
    print("-"*80)

    for i, n_latent in enumerate(all_results['N_latent']):
        naive = all_results['naive'][i] if all_results['naive'][i] is not None else float('inf')
        parallel = all_results['parallel'][i] if all_results['parallel'][i] is not None else float('inf')
        tiled = all_results['tiled'][i] if all_results['tiled'][i] is not None else float('inf')
        optimized = all_results['optimized'][i] if all_results['optimized'][i] is not None else float('inf')

        print(f"{n_latent:<12} {naive:<15.3f} {parallel:<20.3f} {tiled:<15.3f} {optimized:<18.3f}")

    print(f"\n{'='*80}")
    print("Optimization Speedups (vs Naive)")
    print(f"{'='*80}")
    print(f"{'N_latent':<12} {'Warp Parallel':<18} {'Tiled':<15} {'Vectorized':<18}")
    print("-"*80)

    for i, n_latent in enumerate(all_results['N_latent']):
        naive = all_results['naive'][i]
        parallel = all_results['parallel'][i]
        tiled = all_results['tiled'][i]
        optimized = all_results['optimized'][i]

        if naive is not None and naive > 0:
            speedup_parallel = naive / parallel if parallel is not None else 0
            speedup_tiled = naive / tiled if tiled is not None else 0
            speedup_optimized = naive / optimized if optimized is not None else 0

            print(f"{n_latent:<12} {speedup_parallel:<18.2f} {speedup_tiled:<15.2f} {speedup_optimized:<18.2f}")

    print(f"\n{'='*80}")
    print("Scaling Analysis - Time per Latent Vector (μs)")
    print(f"{'='*80}")
    print(f"{'N_latent':<12} {'Naive':<15} {'Warp Parallel':<18} {'Tiled':<15} {'Vectorized':<18}")
    print("-"*80)

    for i, n_latent in enumerate(all_results['N_latent']):
        naive = all_results['naive'][i]
        parallel = all_results['parallel'][i]
        tiled = all_results['tiled'][i]
        optimized = all_results['optimized'][i]

        time_per_latent_naive = (naive * 1000 / n_latent) if naive is not None else 0
        time_per_latent_parallel = (parallel * 1000 / n_latent) if parallel is not None else 0
        time_per_latent_tiled = (tiled * 1000 / n_latent) if tiled is not None else 0
        time_per_latent_optimized = (optimized * 1000 / n_latent) if optimized is not None else 0

        print(f"{n_latent:<12} {time_per_latent_naive:<15.2f} {time_per_latent_parallel:<15.2f} {time_per_latent_tiled:<15.2f} {time_per_latent_optimized:<15.2f}")

    print("="*80)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Benchmark CUDA kernel implementations with optimization and scaling analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This benchmark tests all 4 kernel implementations (Naive, Warp Parallel, Tiled, Vectorized)
across multiple N_latent values to show both optimization improvements and scaling behavior.

Example:
  python benchmark.py
        """
    )

    args = parser.parse_args()
    run_unified_benchmark()

if __name__ == "__main__":
    main()
