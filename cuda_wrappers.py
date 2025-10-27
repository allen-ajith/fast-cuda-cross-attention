"""
Python wrappers for CUDA kernels using ctypes and raw CUDA runtime API
No PyTorch dependency
"""

import ctypes
import numpy as np
import os

try:
    cuda_paths = [
        '/usr/local/cuda/lib64/libcudart.so',
        '/usr/local/cuda/lib/libcudart.so',
        '/opt/cuda/lib64/libcudart.so',
        '/opt/cuda/lib/libcudart.so',
    ]

    cudart = None
    for path in cuda_paths:
        if os.path.exists(path):
            cudart = ctypes.CDLL(path)
            break

    if cudart is None:
        try:
            cudart = ctypes.CDLL('libcudart.so')
        except:
            try:
                cudart = ctypes.CDLL('libcudart.dylib')
            except:
                raise RuntimeError("Could not find CUDA runtime library. Please ensure CUDA is installed.")

    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int

    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int

    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    cudart.cudaGetLastError.argtypes = []
    cudart.cudaGetLastError.restype = ctypes.c_int

    cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    cudart.cudaMemset.restype = ctypes.c_int

    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

except Exception as e:
    raise RuntimeError(f"Failed to load CUDA runtime: {e}")

try:
    lib = ctypes.CDLL('./kernels.so')
except OSError:
    raise RuntimeError("Failed to load kernels.so. Please compile the CUDA kernels first using 'make'.")

lib.launch_naive_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_naive_kernel.restype = None

lib.launch_parallel_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_parallel_kernel.restype = None

lib.launch_tiled_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_tiled_kernel.restype = None

lib.launch_optimized_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_optimized_kernel.restype = None

def _allocate_gpu_memory(size_bytes):
    """Allocate GPU memory."""
    ptr = ctypes.c_void_p()
    err = cudart.cudaMalloc(ctypes.byref(ptr), size_bytes)
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed with error {err}")
    return ptr

def _copy_to_gpu(host_data, gpu_ptr, size_bytes):
    """Copy data from host to GPU."""
    host_ptr = host_data.ctypes.data_as(ctypes.c_void_p)
    err = cudart.cudaMemcpy(gpu_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy H2D failed with error {err}")

def _copy_from_gpu(gpu_ptr, host_data, size_bytes):
    """Copy data from GPU to host."""
    host_ptr = host_data.ctypes.data_as(ctypes.c_void_p)
    err = cudart.cudaMemcpy(host_ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy D2H failed with error {err}")

def _free_gpu_memory(ptr):
    """Free GPU memory."""
    cudart.cudaFree(ptr)

def _run_kernel(kernel_func, Q, K, V, batch_size, N_latent, N_input, D):
    """Helper function to run a CUDA kernel with batch support."""

    Q = np.ascontiguousarray(Q, dtype=np.float32)
    K = np.ascontiguousarray(K, dtype=np.float32)
    V = np.ascontiguousarray(V, dtype=np.float32)

    Q_size = Q.size * 4
    K_size = K.size * 4
    V_size = V.size * 4
    output_size = batch_size * N_latent * D * 4

    Q_gpu = _allocate_gpu_memory(Q_size)
    K_gpu = _allocate_gpu_memory(K_size)
    V_gpu = _allocate_gpu_memory(V_size)
    output_gpu = _allocate_gpu_memory(output_size)

    try:
        err = cudart.cudaMemset(output_gpu, 0, output_size)
        if err != 0:
            raise RuntimeError(f"cudaMemset failed with error {err}")

        _copy_to_gpu(Q, Q_gpu, Q_size)
        _copy_to_gpu(K, K_gpu, K_size)
        _copy_to_gpu(V, V_gpu, V_size)

        kernel_func(
            ctypes.cast(Q_gpu, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(K_gpu, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(V_gpu, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(output_gpu, ctypes.POINTER(ctypes.c_float)),
            batch_size,
            N_latent,
            N_input,
            D,
            ctypes.c_void_p(0)
        )

        err = cudart.cudaGetLastError()
        if err != 0:
            raise RuntimeError(f"Kernel launch failed with CUDA error {err}. "
                            f"This usually indicates a kernel configuration problem, "
                            f"invalid memory access, or architecture incompatibility.")

        err = cudart.cudaDeviceSynchronize()
        if err != 0:
            err_detail = cudart.cudaGetLastError()
            raise RuntimeError(f"[ERROR] cudaDeviceSynchronize failed with error {err}, "
                            f"last CUDA error: {err_detail}. "
                            f"This usually indicates a kernel execution error, "
                            f"such as invalid memory access or illegal instruction.")

        output = np.zeros((batch_size, N_latent, D), dtype=np.float32)
        _copy_from_gpu(output_gpu, output, output_size)


        return output
    finally:
        _free_gpu_memory(Q_gpu)
        _free_gpu_memory(K_gpu)
        _free_gpu_memory(V_gpu)
        _free_gpu_memory(output_gpu)

def run_naive_kernel(Q, K, V, N_latent, N_input, D):
    """Run naive CUDA kernel (baseline - one thread per output element)."""
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_naive_kernel, Q, K, V, batch_size, N_latent, N_input, D)

def run_warp_parallel_kernel(Q, K, V, N_latent, N_input, D):
    """Run warp-parallel CUDA kernel (warp-level cooperation)."""
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_parallel_kernel, Q, K, V, batch_size, N_latent, N_input, D)

def run_tiled_kernel(Q, K, V, N_latent, N_input, D):
    """Run tiled CUDA kernel (shared memory tiling)."""
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_tiled_kernel, Q, K, V, batch_size, N_latent, N_input, D)

def run_vectorized_kernel(Q, K, V, N_latent, N_input, D):
    """Run vectorized CUDA kernel (online softmax + vectorization)."""
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_optimized_kernel, Q, K, V, batch_size, N_latent, N_input, D)
