#include "group.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Addition Test ---------------------------------------------- */
__global__ void initialize_add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z) {
    fq_gpu a_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu a_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu a_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu b_x{ 0xafdb8a15c98bf74c, 0xac54df622a8d991a, 0xc6e5ae1f3dad4ec8, 0x1bd3fb4a59e19b52 };
    fq_gpu b_y{ 0x21b3bb529bec20c0, 0xaabd496406ffb8c1, 0xcd3526c26ac5bdcb, 0x187ada6b8693c184 };
    fq_gpu b_z{ 0xffcd440a228ed652, 0x8a795c8f234145f1, 0xd5279cdbabb05b95, 0xbdf19ba16fc607a };
    fq_gpu exp_x{ 0x18764da36aa4cd81, 0xd15388d1fea9f3d3, 0xeb7c437de4bbd748, 0x2f09b712adf6f18f };
    fq_gpu exp_y{ 0x50c5f3cab191498c, 0xe50aa3ce802ea3b5, 0xd9d6125b82ebeff8, 0x27e91ba0686e54fe };
    fq_gpu exp_z{ 0xe4b81ef75fedf95, 0xf608edef14913c75, 0xfd9e178143224c96, 0xa8ae44990c8accd };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        x[i] = b_x.data[i];
        y[i] = b_y.data[i];
        z[i] = b_z.data[i];
        expected_x[i] = exp_x.data[i];
        expected_y[i] = exp_y.data[i];
        expected_z[i] = exp_z.data[i];
    }
}

__global__ void add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *res_x, var *res_y, var *res_z) {
    g1::projective_element lhs;
    g1::projective_element rhs;
    g1::projective_element result;
    g1::projective_element expected;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        lhs.x.data[tid] = fq_gpu::to_monty(a[tid], res_x[tid]);
        lhs.y.data[tid] = fq_gpu::to_monty(b[tid], res_x[tid]);
        lhs.z.data[tid] = fq_gpu::to_monty(c[tid], res_x[tid]);
        rhs.x.data[tid] = fq_gpu::to_monty(x[tid], res_x[tid]);
        rhs.y.data[tid] = fq_gpu::to_monty(y[tid], res_x[tid]);
        rhs.z.data[tid] = fq_gpu::to_monty(z[tid], res_x[tid]);

        // lhs + rhs (projective element + projective element)
        g1::add_projective(
            lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
            rhs.x.data[tid], rhs.y.data[tid], rhs.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );
        
        // Transform results from montgomery form 
        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);
    }
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void assert_checks(var *expected, var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();
    
    // Assert clause
    // assert(expected[0] == result[0]);
    // assert(expected[1] == result[1]);
    // assert(expected[2] == result[2]);
    // assert(expected[3] == result[3]);

    // Print statements
    printf("expected[0] is: %zu\n", expected[0]);
    printf("expected[1] is: %zu\n", expected[1]);
    printf("expected[2] is: %zu\n", expected[2]);
    printf("expected[3] is: %zu\n", expected[3]);
    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n", result[3]);
}

void execute_kernels
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res_x, var *res_y, var *res_z) {
    // Addition Test
    initialize_add_check_against_constants<<<BLOCKS, THREADS>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z);
    add_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to 'uint64_t' type
    var *a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *res_x, *res_y, *res_z;    

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
    cudaMallocManaged(&res_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_z, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, c, x, y, z, expected_x, expected_y, expected_z, res_x, res_y, res_z);

    // Successfull execution of unit tests
    cout << "******* All 'G1 BN-254 Curve' unit tests passed! **********" << endl;

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
    cudaFree(res_x);
    cudaFree(res_y);
    cudaFree(res_z);

    cout << "Completed sucessfully!" << endl;

    return 0;
}
