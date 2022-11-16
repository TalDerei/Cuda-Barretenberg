#include "../fields/field.cu"

using namespace std;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_TEST = 4;

/* -------------------------- Montgomery Multiplication Test ---------------------------------------------- */

__global__ void initialize_mont_mult(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu b_field{ 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };
    fq_gpu const_expected{ 0x7ed4174114b521c4, 0x58f5bd1d4279fdc2, 0x6a73ac09ee843d41, 0x687a76ae9b3425c };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void mont_mult(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        // res[tid] = a[tid] + b[tid];
        fq_gpu::mul(a[tid], b[tid], res[tid]);
    }
}

/* -------------------------- Montgomery Multiplication Test -- Short Integers ---------------------------------------------- */

__global__ void initialize_mont_mult_short(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0xa, 0, 0, 0 };
    fq_gpu b_field{ 0xb, 0, 0, 0 };
    fq_gpu const_expected = { 0x65991a6dc2f3a183, 0xe3ba1f83394a2d08, 0x8401df65a169db3f, 0x1727099643607bba };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void mont_mult_short(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::mul(a[tid], b[tid], res[tid]);
    }
}

/* -------------------------- Multiply - Square Consistency ---------------------------------------------- */

// __global__ void initialize_mul_square_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
//     fq_gpu a_field{ 0x2523b6fa3956f037, 0x158aa08ecdd9ec1c, 0xf48216a4c74738d8, 0x2514cc93d6f0a1ba };
//     fq_gpu b_field{ 0xb68aee5e4c8fc17a, 0xc5193de7f401d5e7, 0xb8777d4dde671db2, 0xe513e75c087b0ba };

//     for (int i = 0; i < LIMBS; i++) {
//         a[i] = a_field.data[i];
//         b[i] = b_field.data[i];
//     }
// }

// __global__ void mul_square_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
//     fq_gpu t1;
//     fq_gpu t2;
//     fq_gpu mul_result;
//     fq_gpu sqr_result;
    
//     // Calculate global thread ID, and boundry check
//     int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
//     if (tid < LIMBS) {
//         t1[tid] = a[tid] - b[tid];
//         t2[tid] = a[tid] + b[tid];
//         mul_result[tid] = t1[tid] * t2[tid];
//         // t1[tid] = a[tid].sqr();
//         // t2[tid] = b[tid].sqr();
//         // sqr_result[tid] = t1[tid] - t2[tid];
//         // res[tid] = sqr_result[tid];
//         res[tid] = mul_result[tid];
//     }
// }

int main(int, char**) {
    // Define pointers to uint64_t
    var *a, *b, *res;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_TEST * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS_TEST * sizeof(uint64_t));
    cudaMallocManaged(&res, LIMBS * sizeof(uint64_t));

    // Initialize field elements
    initialize_mont_mult<<<1, 1>>>(a, b, res);
    // initialize_mont_mult_short<<<1, 1>>>(a, b, res);
    // initialize_mul_square_consistency<<<1, 1>>>(a, b, res);

    // Montgomery multiplication test
    mont_mult<<<1, LIMBS_TEST>>>(a, b, res);
    // mont_mult_short<<<1, LIMBS_TEST>>>(a, b, res);
    // mul_square_consistency<<<1, LIMBS_TEST>>>(a, b, res);

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