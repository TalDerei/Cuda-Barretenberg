#include "group.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

// Maximum 2^31 - 1 blocks in the x-dimension
constexpr size_t POINTS = 1 << 16;
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
// equal to the total number of blocks launched and each thread per block, which is POINTS * 4.
// The launch parameters define 'GRID' blocks of 'THREADS' threads each.
__global__ void mont_mult(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Calculate global thread ID 
    size_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    // Boundry check
    if (tid < LIMBS) {
        fq_gpu::mul(a[tid % 4], b[tid % 4], res[tid % 4]);
    }
}

/* -------------------------- G1 Curve Addition Test ---------------------------------------------- */
__global__ void initialize_curve_addition(var *a, var *b, var *c, var *x, var *y, var *z) {
    fq_gpu a_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu a_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu a_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu b_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu b_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu b_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        x[i] = b_x.data[i];
        y[i] = b_y.data[i];
        z[i] = b_z.data[i];
    }
}

__global__ void curve_addition(var *a, var *b, var *c, var *x, var *y, var *z, var *res_x, var *res_y, var *res_z) {
    g1_gpu::element lhs;
    g1_gpu::element rhs;
    g1_gpu::element result;
    g1_gpu::element expected;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::load(a[tid], lhs.x.data[tid]);
        fq_gpu::load(b[tid], lhs.y.data[tid]);
        fq_gpu::load(c[tid], lhs.z.data[tid]);
        fq_gpu::load(x[tid], rhs.x.data[tid]);
        fq_gpu::load(y[tid], rhs.y.data[tid]);
        fq_gpu::load(z[tid], rhs.z.data[tid]);

        for (int i = 0; i < POINTS; i++) {
            // lhs + rhs
            g1_gpu::add(
                lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
                rhs.x.data[tid], rhs.y.data[tid], rhs.z.data[tid], 
                res_x[tid], res_y[tid], res_z[tid]
            );

            // Temporarily handle case where P = Q -- NEED TO MOVE TO 'group.cu' file
            if (fq_gpu::is_zero(res_x[tid]) && fq_gpu::is_zero(res_y[tid]) && fq_gpu::is_zero(res_z[tid])) {
                // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
                g1_gpu::doubling(
                    lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
                    res_x[tid], res_y[tid], res_z[tid]
                );
            }
        }
    }
}

/* -------------------------- Main -- Executing Kernels ---------------------------------------------- */
int main(int, char**) {
    // CUDA Event API
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Define pointers to uint64_t type
    var *a, *b, *c, *x, *y, *z, *res, *res_x, *res_y, *res_z;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&c, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_z, LIMBS * sizeof(uint64_t));

    // Dim3 variables
    dim3 THREADS(LIMBS_NUM);
    dim3 GRID(POINTS);

    // Start event timestamp
    cudaEventRecord(start);

    // Launch initialization and workload kernels
    // initialize_mont_mult<<<1, LIMBS_NUM>>>(a, b, res);
    // mont_mult<<<GRID, THREADS>>>(a, b, res);

    initialize_curve_addition<<<1, LIMBS_NUM>>>(a, b, c, x, y, z);
    curve_addition<<<GRID, THREADS>>>(a, b, c, x, y, z, res_x, res_y, res_z);

    // End event timestamp
    cudaEventRecord(stop);
    
    // Synchronization barrier that blocks CPU execution until the specified event is recorded
    cudaEventSynchronize(stop);

    // Print results
    // printf("result[0] is: %zu\n", res[0]);
    // printf("result[1] is: %zu\n", res[1]);
    // printf("result[2] is: %zu\n", res[2]);
    // printf("result[3] is: %zu\n", res[3]);

    printf("result[0] is: %zu\n", res_x[0]);
    printf("result[1] is: %zu\n", res_x[1]);
    printf("result[2] is: %zu\n", res_x[2]);
    printf("result[3] is: %zu\n", res_x[3]);

    printf("result[0] is: %zu\n", res_y[0]);
    printf("result[1] is: %zu\n", res_y[1]);
    printf("result[2] is: %zu\n", res_y[2]);
    printf("result[3] is: %zu\n", res_y[3]);

    printf("result[0] is: %zu\n", res_z[0]);
    printf("result[1] is: %zu\n", res_z[1]);
    printf("result[2] is: %zu\n", res_z[2]);
    printf("result[3] is: %zu\n", res_z[3]);
    
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

/**
 * TODO: Add assert check
*/