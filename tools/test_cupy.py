# Test of cupy functionality for parallel computations

import numpy as np
import cupy as cp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    x_gpu = cp.array([1, 2, 3])
    print("l2_gpu = cp.linalg.norm(x_gpu): ", cp.linalg.norm(x_gpu))
    
    with cp.cuda.Device(0):
        x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
        print("x_on_gpu0:", x_on_gpu0)
        print("device: ", x_on_gpu0.device)
        
    # move array to device
    x_cpu = np.array([1, 2, 3])
    x_gpu = cp.asarray(x_cpu)  # move the data to the current device.
    
    x_cpu1 = cp.asnumpy(x_gpu)  # move the array to the host.
    print("x_cpu1 type: ", x_cpu1.__class__)
    
    # CPU/GPU agnostic code
    def softplus(x):
        xp = cp.get_array_module(x)  # 'xp' is a standard usage in the community
        print("Using:", xp.__name__)
        return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))
    
    x_plus = softplus(x_cpu)
    x_plus_gpu = softplus(x_gpu)
    
    
    # raw kernels 
    add_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_add(const float* x1, const float* x2, float* y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }
    ''', 'my_add')

    x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
    y = cp.zeros((5, 5), dtype=cp.float32)
    add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
    print("y: ", y)
    
    # raw kernel matrix multiplication
    matmul_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_add(T x1, T x2, T y) {
        # int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] * x2[tid];
    }
    ''', 'my_matmul')

    x1 = cp.arange(50, dtype=cp.float32).reshape(2, 5, 5)
    x2 = cp.arange(50, dtype=cp.float32).reshape(2, 5, 5)
    y = cp.zeros((2, 5, 5), dtype=cp.float32)
    matmul_kernel((2,), (5,5), (x1, x2, y))  # grid, block and arguments
    print("y: ", y)