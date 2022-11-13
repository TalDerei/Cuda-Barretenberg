#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <iostream>

#include "../fields/field.cu"

using namespace std;
using namespace gpu_barretenberg;

const unsigned int ELEMENTS = 4;
const unsigned int BYTES_PER_ELEM = 32;

__global__ void initialize(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Initialize field elements
    fq_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu b_field{ 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };

    for (int i = 0; i < ELEMENTS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void test(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < ELEMENTS) {
        fq_gpu::mul(a[tid], b[tid], res[tid]);
    }

    // Calculate global thread ID, and boundry check
    // int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    // if (tid < ELEMENTS) {
    //     c[tid] = a[tid] + b[tid];
    // }
}

int main(int, char**) {
    // Define pointers to uint64_t
    var *a, *b, *res;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, ELEMENTS * sizeof(uint64_t));
    cudaMallocManaged(&b, ELEMENTS * sizeof(uint64_t));
    cudaMallocManaged(&res, ELEMENTS * sizeof(uint64_t));

    // initialize<<<1, ELEMENTS>>>(a, b, mod);
    initialize<<<1, 1>>>(a, b, res);

    test<<<1, 4>>>(a, b, res);

    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    printf("result is: %zu\n", res[0]);
    printf("result is: %zu\n", res[1]);
    printf("result is: %zu\n", res[2]);
    printf("result is: %zu\n", res[3]);

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(res);

    cout << "Completed sucessfully!" << endl;

    return 0;
}