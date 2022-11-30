#include "../fields/group.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Mixed Addition ---------------------------------------------- */

__global__ void initialize_mixed_add_check_against_constants(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res) {
    fq_gpu a_x{ 0x92716caa6cac6d26, 0x1e6e234136736544, 0x1bb04588cde00af0, 0x9a2ac922d97e6f5 };
    fq_gpu a_y{ 0x9e693aeb52d79d2d, 0xf0c1895a61e5e975, 0x18cd7f5310ced70f, 0xac67920a22939ad };
    fq_gpu a_z{ 0xfef593c9ce1df132, 0xe0486f801303c27d, 0x9bbd01ab881dc08e, 0x2a589badf38ec0f9 };
    fq_gpu b_x{ 0xa1ec5d1398660db8, 0x6be3e1f6fd5d8ab1, 0x69173397dd272e11, 0x12575bbfe1198886 };
    fq_gpu b_y{ 0xcfbfd4441138823e, 0xb5f817e28a1ef904, 0xefb7c5629dcc1c42, 0x1a9ed3d6f846230e };
    fq_gpu exp_x{ 0x2a9d0201fccca20, 0x36f969b294f31776, 0xee5534422a6f646, 0x911dbc6b02310b6 };
    fq_gpu exp_y{ 0x14c30aaeb4f135ef, 0x9c27c128ea2017a1, 0xf9b7d80c8315eabf, 0x35e628df8add760 };
    fq_gpu exp_z{ 0xa43fe96673d10eb3, 0x88fbe6351753d410, 0x45c21cc9d99cb7d, 0x3018020aa6e9ede5 };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        x[i] = b_x.data[i];
        y[i] = b_y.data[i];
    }
}

__global__ void mixed_add_check_against_constants(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res) {
    g1::element lhs;
    g1::affine_element rhs;
    g1::element result;
    g1::element expected;
    
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        lhs.x.data[tid] = fq_gpu::to_monty(a[tid], res[tid]);
        lhs.y.data[tid] = fq_gpu::to_monty(b[tid], res[tid]);
        lhs.z.data[tid] = fq_gpu::to_monty(c[tid], res[tid]);
        rhs.x.data[tid] = fq_gpu::to_monty(x[tid], res[tid]);
        rhs.y.data[tid] = fq_gpu::to_monty(y[tid], res[tid]);

        g1::mixed_add(lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], rhs.x.data[tid], rhs.y.data[tid], expected_x[tid], expected_y[tid], expected_z[tid]);
        
        expected_x[tid] = fq_gpu::from_monty(expected_x[tid], expected_x[tid]);
        expected_y[tid] = fq_gpu::from_monty(expected_y[tid], expected_y[tid]);
        expected_z[tid] = fq_gpu::from_monty(expected_z[tid], expected_z[tid]);

        // EXPECT_EQ(result == expected, true);
    }
}

/* -------------------------- Doubling ---------------------------------------------- */

__global__ void initialize_dbl_check_against_constants(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res) {
    fq_gpu a_x{ 0x8d1703aa518d827f, 0xd19cc40779f54f63, 0xabc11ce30d02728c, 0x10938940de3cbeec };
    fq_gpu a_y{ 0xcf1798994f1258b4, 0x36307a354ad90a25, 0xcd84adb348c63007, 0x6266b85241aff3f };
    fq_gpu a_z{ 0xe213e18fd2df7044, 0xb2f42355982c5bc8, 0xf65cf5150a3a9da1, 0xc43bde08b03aca2 };
    fq_gpu exp_x{ 0xd5c6473044b2e67c, 0x89b185ea20951f3a, 0x4ac597219cf47467, 0x2d00482f63b12c86 };
    fq_gpu exp_y{ 0x4e7e6c06a87e4314, 0x906a877a71735161, 0xaa7b9893cc370d39, 0x62f206bef795a05 };
    fq_gpu exp_z{ 0x8813bdca7b0b115a, 0x929104dffdfabd22, 0x3fff575136879112, 0x18a299c1f683bdca };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
    }
}

__global__ void dbl_check_against_constants(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res) {
    g1::element lhs;
    g1::element result;
    g1::element expected;
    
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        lhs.x.data[tid] = fq_gpu::to_monty(a[tid], res[tid]);
        lhs.y.data[tid] = fq_gpu::to_monty(b[tid], res[tid]);
        lhs.z.data[tid] = fq_gpu::to_monty(c[tid], res[tid]);

        g1::doubling(lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], expected_x[tid], expected_y[tid], expected_z[tid]);
        g1::doubling(expected_x[tid], expected_y[tid], expected_z[tid], expected_x[tid], expected_y[tid], expected_z[tid]);
        g1::doubling(expected_x[tid], expected_y[tid], expected_z[tid], expected_x[tid], expected_y[tid], expected_z[tid]);

        // Converting from monty form gets back the correct result
        expected_x[tid] = fq_gpu::from_monty(expected_x[tid], res[tid]);
        expected_y[tid] = fq_gpu::from_monty(expected_y[tid], res[tid]);
        expected_z[tid] = fq_gpu::from_monty(expected_z[tid], res[tid]);

        // EXPECT_EQ(result == expected, true);
    }
}

/* -------------------------- Main -- Executing Kernels ---------------------------------------------- */

void execute_kernels(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res) {
    // Initialization kernels
    // initialize_mixed_add_check_against_constants<<<BLOCKS, THREADS>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z, res);
    initialize_dbl_check_against_constants<<<BLOCKS, THREADS>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z, res);

    // Workload kernels
    // mixed_add_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z, res);
    dbl_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z, res);
}

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *res;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&c, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, c, x, y, z, expected_x, expected_y, expected_z, res);

    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results
    // printf("res[0] is: %zu\n", res[0]);
    // printf("res[1] is: %zu\n", res[1]);
    // printf("res[2] is: %zu\n", res[2]);
    // printf("res[3] is: %zu\n", res[3]);

    printf("expected_x[0] is: %zu\n", expected_x[0]);
    printf("expected_x[1] is: %zu\n", expected_x[1]);
    printf("expected_x[2] is: %zu\n", expected_x[2]);
    printf("expected_x[3] is: %zu\n", expected_x[3]);

    printf("expected_y[0] is: %zu\n", expected_y[0]);
    printf("expected_y[1] is: %zu\n", expected_y[1]);
    printf("expected_y[2] is: %zu\n", expected_y[2]);
    printf("expected_y[3] is: %zu\n", expected_y[3]);

    printf("expected_z[0] is: %zu\n", expected_z[0]);
    printf("expected_z[1] is: %zu\n", expected_z[1]);
    printf("expected_z[2] is: %zu\n", expected_z[2]);
    printf("expected_z[3] is: %zu\n", expected_z[3]);

    // End timer
    auto stop = high_resolution_clock::now();

    // Calculate duraion of execution time 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds\n" << endl; 

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(expected_x);
    cudaFree(expected_y);
    cudaFree(expected_z);
    cudaFree(res);

    cout << "Completed sucessfully!" << endl;

    return 0;
}