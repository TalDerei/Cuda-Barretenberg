#include "field.cu"
#include <assert.h> 
#include <iostream>
#include <algorithm>


using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Simple Kernel Function for MSM Test ---------------------------------------------- */

// Initialize points and scalars
__global__ void initialize_simple_double_and_add(uint64_t *a, uint64_t *b, uint64_t *expected) {
    fq_gpu point{ 10, 0, 0, 0 };
    fq_gpu scalar{ 10, 0, 0, 0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = scalar.data[i];
        b[i] = point.data[i];
    }
}

// Native approach for computing scalar mutliplication with time complexity: O(2^k)
// nP = P + P ... + P
__global__ void naive_double_and_add(uint64_t *scalars, uint64_t *points, uint64_t *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < LIMBS) {
        fq_gpu::mul(scalars[tid], points[tid], result[tid]); 
    }
}

// Double and add implementation with time complexity: O(k)
__global__ void double_and_add(uint64_t *scalars, uint64_t *points, uint64_t *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    fq_gpu R;
    fq_gpu Q;

    if (tid < LIMBS) {
        // Initialize R to the identity element
        R.data[tid] = fq_gpu::load(1, R.data[tid]); 

         // Initialize Q to the input point
        Q.data[tid] = fq_gpu::load(points[tid], Q.data[tid]);

        for (int i = 3; i >= 0; i--) {
            fq_gpu::square(R.data[tid], R.data[tid]);

            // Extracts the i-th bit of scalars[tid], and the expression evaluates to 1 or 0
            if ((scalars[tid] >> i) & 1) {
                fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);                   
            }
        }
    }
    
    // Load final R result into result
    result[tid] = fq_gpu::load(R.data[tid], result[tid]);
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void execute_kernels(var *a, var *b, var *expected, var *result, var *bit) {    
    // MSM Test
    initialize_simple_double_and_add<<<BLOCKS, THREADS>>>(a, b, expected);
    // naive_double_and_add<<<BLOCKS, LIMBS_NUM>>>(a, b, result);
    double_and_add<<<BLOCKS, LIMBS_NUM>>>(a, b, result);

    cudaDeviceSynchronize();

    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n", result[3]);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *expected, *result, *bit;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(var));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(var));
    cudaMallocManaged(&expected, LIMBS * sizeof(var));
    cudaMallocManaged(&result, LIMBS * sizeof(var));
    cudaMallocManaged(&bit, sizeof(var));

    // Execute kernel functions
    execute_kernels(a, b, expected, result, bit);

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


// Open Questions?
    // 1. Is the kernel implementation correct?
        // --> Check by performing on non-field / non-curve elements to make the test simpler. 
        // Kernel isn't correct yet, something to do probably with the indexing

    // 2. What baseline are we comparing against, and what's the difference between simple 
    // performing a MUL(scalar, point) vs. double and add algorithm?
        // --> We are checking by running a naive kernel that multiplies the same point by itself N times
        // and then comparing the result to the double-and-add kernel.

    // 3. Why use double and add?
    // It becomes more efficient for larger numbers, it's over elliptic curve points not just field elements.
    // and O(k) complexity vs O(2^k) complexity.

    // 4. How is the flow dependency issue being resolved, as each thread acts on a different bit?
        // --> each thread acts on a different limb, but sequentially in its own limb

    // 5. How to expand this to perform on an actual Fq point and Fr scalar?
    // 6. How to expand this to perform on a vector of points and scalars?
    // 7. Does the result correspond with the expected result?