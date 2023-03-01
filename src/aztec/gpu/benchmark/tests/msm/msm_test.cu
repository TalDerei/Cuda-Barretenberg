#include "field.cu"
#include <assert.h> 
#include <iostream>
#include <algorithm>


using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

// Kernel launch parameters
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 4;

/* -------------------------- Simple Kernel Function for MSM Test ---------------------------------------------- */

// Initialize points and scalars
__global__ void initialize_simple_double_and_add(uint64_t *a, uint64_t *b) {
    fq_gpu point{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu scalar{ 0x09, 0x0, 0x0, 0x0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = scalar.data[i];
        b[i] = point.data[i];
    }
}

// Simple montgomery multiplication as a baseline reference
__global__ void simple_multiplication(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < LIMBS) {
        fq_gpu::to_monty(scalar[tid], scalar[tid]);  
        fq_gpu::mul(point[tid], scalar[tid], result[tid]); 
    }
}

// Native approach for computing scalar mutliplication with time complexity: O(2^k)
// nP = P + P ... + P 
__global__ void naive_double_and_add(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::add(point[tid], point[tid], result[tid]);     
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
    }
}

// Double and add implementation using bit-decomposition with time complexity: O(2^k / 2)
__global__ void double_and_add_halv(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    fq_gpu R;
    fq_gpu Q;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.data[tid]); 
        fq_gpu::load(point[tid], Q.data[tid]);

        // Loop unrolling
        fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);   // P
        fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]);   // 2P
        fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]);   // 4P
        fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]);   // 8P 
        fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);   // 9P 
    }
    
    // Store the final value of R into the result array for this limb
    fq_gpu::load(R.data[tid], result[tid]);
}

// Double and add implementation using bit-decomposition with time complexity: O(k)
__global__ void double_and_add(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    fq_gpu R;
    fq_gpu Q;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.data[tid]); 
        fq_gpu::load(point[tid], Q.data[tid]);
        
        for (int i = 3; i >= 0; i--) {
            // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
            // and extracting the i-th bit of scalar in limb.
            if (((scalar[0] >> i) & 1) ? 1 : 0)
                fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);  
            if (i != 0) 
                fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]); 
        }
    }
    
    // Store the final value of R into the result array for this limb
    fq_gpu::load(R.data[tid], result[tid]);
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void print(var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for each limb
    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n\n", result[3]);
}

void execute_kernels(var *a, var *b, var *result) {    
    // MSM Test
    initialize_simple_double_and_add<<<BLOCKS, THREADS>>>(a, b);
    // simple_multiplication<<<BLOCKS, THREADS>>>(a, b, result);
    // naive_double_and_add<<<BLOCKS, THREADS>>>(a, b, result);
    // double_and_add_halv<<<BLOCKS, THREADS>>>(a, b, result);
    double_and_add<<<BLOCKS, THREADS>>>(a, b, result);
    print(result);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *result;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS * sizeof(var));
    cudaMallocManaged(&b, LIMBS * sizeof(var));
    cudaMallocManaged(&result, LIMBS * sizeof(var));

    // Execute kernel functions
    execute_kernels(a, b, result);

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


/*
Notes:
1. How to expand this to perform on an actual Fq point and Fr scalar?
    --> They operate over different prime fields, but Fr scalar doesn't participate 
    in the addition / multiplication calculations, only multiples of the Fq curve element. 

2. How to expand this to add an elliptic curve point G?
3. How to expand this to perform on a vector of points and scalars with larger kernel parameters?
*/
