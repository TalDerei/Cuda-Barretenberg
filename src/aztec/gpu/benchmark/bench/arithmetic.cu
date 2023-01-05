#include "field.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

// Maximum 2^31 - 1 blocks in the x-dimension
constexpr size_t POINTS = 1UL << 30;
// Number of limbs 
static constexpr size_t LIMBS_NUM = 4;

/* -------------------------- Montgomery Multiplication Test ---------------------------------------------- */

__global__ void initialize_mont_mult(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu b_field{ 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };
    fq_gpu const_expected{ 0x7ed4174114b521c4, 0x58f5bd1d4279fdc2, 0x6a73ac09ee843d41, 0x687a76ae9b3425c };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

// Each thread performs a single multiplication, so the total number of multiplications is 
// equal to the total number of threads launched, which is NUM_POINTS * 4.
__global__ void mont_mult(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Calculate global thread ID 
    size_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    // Boundry check
    if (tid < LIMBS) {
        fq_gpu::mul(a[tid % 4], b[tid % 4], res[tid % 4]);
    }
}

/* -------------------------- Main -- Executing Kernels ---------------------------------------------- */
int main(int, char**) {
    // CUDA Event API
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Define pointers to uint64_t type
    var *a, *b, *res;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&res, LIMBS * sizeof(uint64_t));

    // Dim3 variables
    dim3 THREADS(LIMBS_NUM);
    dim3 GRID(POINTS);

    // Start event timestamp
    cudaEventRecord(start);

    // Launch initialization and workload kernels
    initialize_mont_mult<<<1, LIMBS_NUM>>>(a, b, res);
    mont_mult<<<GRID, THREADS>>>(a, b, res);

    // End event timestamp
    cudaEventRecord(stop);
    
    // Synchronization barrier that blocks CPU execution until the specified event is recorded
    cudaEventSynchronize(stop);

    // Print results
    printf("result[0] is: %zu\n", res[0]);
    printf("result[1] is: %zu\n", res[1]);
    printf("result[2] is: %zu\n", res[2]);
    printf("result[3] is: %zu\n", res[3]);
    
    // Calculate duraion of execution time 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time taken by function: " << milliseconds << " milliseconds\n" << endl; 

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(res);

    cout << "Completed sucessfully!" << endl;

    return 0;
}