#include "field.cu"
#include <assert.h> 

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Montgomery Multiplication Test ---------------------------------------------- */

__global__ void initialize_msm(uint64_t *a, uint64_t *b, uint64_t *expected) {

}

__global__ void msm(uint64_t *a, uint64_t *b, uint64_t *result) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {}
}


/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void assert_checks(var *expected, var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Assert clause
    assert(expected[0] == result[0]);
    assert(expected[1] == result[1]);
    assert(expected[2] == result[2]);
    assert(expected[3] == result[3]);
}

void execute_kernels(var *a, var *b, var *expected, var *result) {    
    // MSM Test
    initialize_msm<<<BLOCKS, THREADS>>>(a, b, expected);
    msm<<<BLOCKS, LIMBS_NUM>>>(a, b, result);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *expected, *result;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(var));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(var));
    cudaMallocManaged(&expected, LIMBS * sizeof(var));
    cudaMallocManaged(&result, LIMBS * sizeof(var));

    // Execute kernel functions
    execute_kernels(a, b, expected, result);

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