#include "group.cu"

#include <assert.h> 
#include <iostream>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>
#include <cooperative_groups.h>

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;
using namespace cooperative_groups;

// Kernel launch parameters
static constexpr size_t BLOCKS = 256;
static constexpr size_t THREADS = 256;
static constexpr size_t POINTS = 1 << 16;

/* -------------------------- Kernel Functions ---------------------------------------------- */

/* -------------------------- Kernel 1 ---------------------------------------------- */

// Sum reduction with warp divergence
__global__ void sum_reduction_1(int *v, int *v_r) { 
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction in shared memory
    __shared__ int partial_sum[256];

    // Partial_sum array is being used to accumulate partial sums
    partial_sum[threadIdx.x] = v[tid];

    // Sychronization barrier
    __syncthreads();
    
    // Warp divergence to determine active threads based on stride
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Accumulate result into current block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

/* -------------------------- Kernel 2 ---------------------------------------------- */

// Sum reduction using sequential threads (eliminating warp divergence and modulo operation). 
// This performs ~2x compared to sum_reduction_1. 
__global__ void sum_reduction_2(int *v, int *v_r) { 
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction in shared memory
    __shared__ int partial_sum[256];

    // Partial_sum array is being used to accumulate partial sums
    partial_sum[threadIdx.x] = v[tid];

    // Sychronization barrier
    __syncthreads();
    
    for (int s = 1; s < blockDim.x; s *= 2) {
        // Change the indexing to be sequential threads (i.e. divide threads into groups)
        int index = 2 * s * threadIdx.x;

        // Each thread does work unless the idex goes off the block
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index + s];
        }
        __syncthreads();
    }

    // Accumulate result into current block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

/* -------------------------- Kernel 3 ---------------------------------------------- */

// Contiguous memory access instead of strided access, avoiding shared memory bank conflicts.
// Bank conflicts arise because of some specific access pattern of data in shared memory.
__global__ void sum_reduction_3(int *v, int *v_r) { 
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction in shared memory
    __shared__ int partial_sum[256];

    // Partial_sum array is being used to accumulate partial sums
    partial_sum[threadIdx.x] = v[tid];

    // Sychronization barrier
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless the index goes off the block
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Accumulate result into current block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

/* -------------------------- Kernel 4 ---------------------------------------------- */

// Half the threads are idle after the first iteration. Instead,
// launch half the number of threads and pack more work. 
__global__ void sum_reduction_4(int *v, int *v_r) { 
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction in shared memory
    __shared__ int partial_sum[256];

    // Load elements and perform first pass of the reduction,
    // Scale i since vector is 2x as long as the number of threads
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Partial_sum array is being used to accumulate partial sums
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];

    // Sychronization barrier
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Each thread does work unless the idex goes off the block
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Accumulate result into current block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

/* -------------------------- Kernel 4 ---------------------------------------------- */

// Used for last iteration to save useless work
// 'volatile' to prevent caching in registers (compiler optimization)
__device__ void warpReduce(volatile int *shared_mem_ptr, int t) {
    shared_mem_ptr[t] += shared_mem_ptr[t + 32];
    shared_mem_ptr[t] += shared_mem_ptr[t + 16];
    shared_mem_ptr[t] += shared_mem_ptr[t + 8];
    shared_mem_ptr[t] += shared_mem_ptr[t + 4];
    shared_mem_ptr[t] += shared_mem_ptr[t + 2];
    shared_mem_ptr[t] += shared_mem_ptr[t + 1];
}

// Warp reduce with loop unrolling
__global__ void sum_reduction_5(int *v, int *v_r) { 
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction in shared memory
    __shared__ int partial_sum[256];

    // Load elements and perform first pass of the reduction,
    // Scale i since vector is 2x as long as the number of threads
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Partial_sum array is being used to accumulate partial sums
    partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];

    // Sychronization barrier
    __syncthreads();
    
    // Do all iterations until reaching the last warp, otherwise
    // results in a lot of useless checks
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        // Each thread does work unless it is further than the stride
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReduce(partial_sum, threadIdx.x);
    }

    // Accumulate result into current block
    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

/* -------------------------- Helper Functions ---------------------------------------------- */

/**
 * Print finite field tests
 */
void print_field_tests(var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for each limb
    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n\n", result[3]);
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

// Execute sum reduction kernel
void execute_sum_reduction(var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z) {    
    size_t bytes = POINTS * sizeof(int);

    // Allocate dynamic memory
    int *h_v, *h_v_r, *d_v, *d_v_r;
    h_v = (int *)malloc(POINTS * sizeof(int));
    h_v_r = (int *)malloc(POINTS * sizeof(int));
    cudaMalloc(&d_v, POINTS * sizeof(int));
    cudaMalloc(&d_v_r, POINTS * sizeof(int));

    // Populate array
    for (int i = 0; i < POINTS; i++) {
        h_v[i] = 1;
    }

    // Copy array to device
    cudaMemcpy(d_v, h_v, POINTS * sizeof(int), cudaMemcpyHostToDevice);

    // Grid size
    int GRID_SIZE = (int)ceil(POINTS / 256);

    // Launch kernels
    // sum_reduction_1<<<GRID_SIZE, 256>>>(d_v, d_v_r);
    // sum_reduction_1<<<1, GRID_SIZE>>>(d_v_r, d_v_r);

    // sum_reduction_2<<<GRID_SIZE, 256>>>(d_v, d_v_r);
    // sum_reduction_2<<<1, GRID_SIZE>>>(d_v_r, d_v_r);

    // sum_reduction_3<<<GRID_SIZE, 256>>>(d_v, d_v_r);
    // sum_reduction_3<<<1, GRID_SIZE>>>(d_v_r, d_v_r);

    // sum_reduction_4<<<GRID_SIZE / 2, 256>>>(d_v, d_v_r);
    // sum_reduction_4<<<1, GRID_SIZE / 2>>>(d_v_r, d_v_r);

    sum_reduction_5<<<GRID_SIZE / 2, 256>>>(d_v, d_v_r);
    sum_reduction_5<<<1, GRID_SIZE / 2>>>(d_v_r, d_v_r);

    // Copy results to host
    cudaMemcpy(h_v_r, d_v_r, POINTS * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Accumulated result is: " << h_v_r[0] << endl;
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *c, *d, *result, *res_x, *res_y, *res_z;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS * sizeof(var));
    cudaMallocManaged(&b, LIMBS * sizeof(var));
    cudaMallocManaged(&c, LIMBS * sizeof(var));
    cudaMallocManaged(&d, LIMBS * sizeof(var));
    cudaMallocManaged(&result, LIMBS * sizeof(var));
    cudaMallocManaged(&res_x, LIMBS * sizeof(var));
    cudaMallocManaged(&res_y, LIMBS * sizeof(var));
    cudaMallocManaged(&res_z, LIMBS * sizeof(var));

    // Execute kernel functions
    execute_sum_reduction(a, b, c, d, result, res_x, res_y, res_z);

    // Successfull execution of unit tests
    cout << "******* All 'MSM' unit tests passed! **********" << endl;

    // End timer
    auto stop = high_resolution_clock::now();

    // Calculate duraion of execution time 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds\n" << endl; 

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(result);

    cout << "Completed sucessfully!" << endl;

    return 0;
}

/** Developer Doscs 
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * https://enccs.github.io/CUDA/3.01_ParallelReduction/#unroll-the-last-warp
*/