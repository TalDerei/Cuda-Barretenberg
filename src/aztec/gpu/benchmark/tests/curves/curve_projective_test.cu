#include "group.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

// Kernel launch parameters
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 4;
static constexpr size_t POINTS = 1 << 10;

/* -------------------------- Addition Test ---------------------------------------------- */
__global__ void initialize_add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, g1::element *t1, g1::element *t2, g1::element *expected) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    fq_gpu::load(0x0, a[tid]);
    fq_gpu::load(field_gpu<fq_gpu>::one().data[tid], b[tid]);
    fq_gpu::load(0x0, c[tid]);

    fq_gpu::load(a[tid], t1[0].x.data[tid]); 
    fq_gpu::load(b[tid], t1[0].y.data[tid]); 
    fq_gpu::load(c[tid], t1[0].z.data[tid]); 
}

__global__ void initialize_affine(g1::affine_element *aff) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    fq_gpu a_x{ 0x01, 0x0, 0x0, 0x0 };
    fq_gpu a_y{ 0x02, 0x0, 0x0, 0x0 };

    fq_gpu::load(a_x.data[tid], aff[0].x.data[tid]); 
    fq_gpu::load(a_y.data[tid], aff[0].y.data[tid]); 
}

__global__ void initialize_jacobian(g1::element *jac) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    fq_gpu a_x{ 0x1, 0x0, 0x0, 0x0 };
    fq_gpu a_y{ 0x2, 0x0, 0x0, 0x0 };
    fq_gpu a_z{ 0x0, 0x0, 0x0, 0x0 };

    fq_gpu::load(a_x.data[tid], jac[0].x.data[tid]); 
    fq_gpu::load(a_y.data[tid], jac[0].y.data[tid]); 
    fq_gpu::load(a_z.data[tid], jac[0].z.data[tid]); 
}

__global__ void initialize_projective(g1::projective_element *proj) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    fq_gpu a_x{ 0x1, 0x0, 0x0, 0x0 };
    fq_gpu a_y{ 0x2, 0x0, 0x0, 0x0 };
    fq_gpu a_z{ 0x0, 0x0, 0x0, 0x0 };

    fq_gpu::load(a_x.data[tid], proj[0].x.data[tid]); 
    fq_gpu::load(a_y.data[tid], proj[0].y.data[tid]); 
    fq_gpu::load(a_z.data[tid], proj[0].z.data[tid]); 
}


__global__ void add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *res_x, var *res_y, var *res_z, g1::element *t1, g1::projective_element *t2, g1::projective_element *t3) {
    g1::projective_element lhs;
    g1::projective_element rhs;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::to_monty(t1[0].x.data[tid], lhs.x.data[tid]);
        fq_gpu::to_monty(t1[0].y.data[tid], lhs.y.data[tid]);
        fq_gpu::to_monty(t1[0].z.data[tid], lhs.z.data[tid]);
        fq_gpu::to_monty(t2[0].x.data[tid], rhs.x.data[tid]);
        fq_gpu::to_monty(t2[0].y.data[tid], rhs.y.data[tid]);
        fq_gpu::to_monty(t2[0].z.data[tid], rhs.z.data[tid]);

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

        fq_gpu::load(res_x[tid], t3[0].x.data[tid]); 
        fq_gpu::load(res_y[tid], t3[0].y.data[tid]); 
        fq_gpu::load(res_z[tid], t3[0].z.data[tid]); 
    }
}

// Compare two elliptic curve elements
__global__ void comparator_kernel(g1::element *point, g1::projective_element *point_2, uint64_t *result) {     
    fq_gpu lhs_zz;
    fq_gpu lhs_zzz;
    fq_gpu rhs_zz;
    fq_gpu rhs_zzz;
    fq_gpu lhs_x;
    fq_gpu lhs_y;
    fq_gpu rhs_x;
    fq_gpu rhs_y;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    lhs_zz.data[tid] =  fq_gpu::square(point[0].z.data[tid], lhs_zz.data[tid]);
    lhs_zzz.data[tid] = fq_gpu::mul(lhs_zz.data[tid], point[0].z.data[tid], lhs_zzz.data[tid]);
    rhs_zz.data[tid] = fq_gpu::square(point_2[0].z.data[tid], rhs_zz.data[tid]);
    rhs_zzz.data[tid] = fq_gpu::mul(rhs_zz.data[tid], point_2[0].z.data[tid], rhs_zzz.data[tid]);
    lhs_x.data[tid] = fq_gpu::mul(point[0].x.data[tid], rhs_zz.data[tid], lhs_x.data[tid]);
    lhs_y.data[tid] = fq_gpu::mul(point[0].y.data[tid], rhs_zzz.data[tid], lhs_y.data[tid]);
    rhs_x.data[tid] = fq_gpu::mul(point_2[0].x.data[tid], lhs_zz.data[tid], rhs_x.data[tid]);
    rhs_y.data[tid] = fq_gpu::mul(point_2[0].y.data[tid], lhs_zzz.data[tid], rhs_y.data[tid]);
    result[tid] = ((lhs_x.data[tid] == rhs_x.data[tid]) && (lhs_y.data[tid] == rhs_y.data[tid]));
}

__global__ void affine_to_projective(g1::affine_element *point, g1::projective_element *point_2) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    fq_gpu::load(point[0].x.data[tid], point_2[0].x.data[tid]);
    fq_gpu::load(point[0].y.data[tid], point_2[0].y.data[tid]);
    fq_gpu::load(field_gpu<fq_gpu>::one().data[tid], point_2[0].z.data[tid]);
}

__global__ void jacobian_to_projective(g1::element *point, g1::projective_element *point_2) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fq_gpu t1; 

    fq_gpu::mul(point[0].x.data[tid], point[0].z.data[tid], point_2[0].x.data[tid]);
    fq_gpu::load(point[0].y.data[tid], point_2[0].y.data[tid]);
    fq_gpu::square(point[0].z.data[tid], t1.data[tid]);
    fq_gpu::mul(t1.data[tid], point[0].z.data[tid], point_2[0].z.data[tid]);
}

__global__ void projective_to_jacobian(g1::projective_element *point, g1::element *point_2) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fq_gpu t1; 

    fq_gpu::mul(point[0].x.data[tid], point[0].z.data[tid], point_2[0].x.data[tid]);
    fq_gpu::square(point[0].z.data[tid], t1.data[tid]);
    fq_gpu::mul(point[0].y.data[tid], t1.data[tid], point_2[0].y.data[tid]);
    fq_gpu::load(point[0].z.data[tid], point_2[0].z.data[tid]);
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
    printf("result[3] is: %zu\n\n", result[3]);
}

void print_affine(g1::affine_element *aff) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for projective point
    printf("expected[0] is: %zu\n", aff[0].x.data[0]);
    printf("expected[0] is: %zu\n", aff[0].x.data[1]);
    printf("expected[0] is: %zu\n", aff[0].x.data[2]);
    printf("expected[0] is: %zu\n\n", aff[0].x.data[3]);

    printf("expected[0] is: %zu\n", aff[0].y.data[0]);
    printf("expected[0] is: %zu\n", aff[0].y.data[1]);
    printf("expected[0] is: %zu\n", aff[0].y.data[2]);
    printf("expected[0] is: %zu\n\n", aff[0].y.data[3]);
}

void print_projective(g1::projective_element *proj) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for projective point
    printf("expected[0] is: %zu\n", proj[0].x.data[0]);
    printf("expected[0] is: %zu\n", proj[0].x.data[1]);
    printf("expected[0] is: %zu\n", proj[0].x.data[2]);
    printf("expected[0] is: %zu\n\n", proj[0].x.data[3]);

    printf("expected[0] is: %zu\n", proj[0].y.data[0]);
    printf("expected[0] is: %zu\n", proj[0].y.data[1]);
    printf("expected[0] is: %zu\n", proj[0].y.data[2]);
    printf("expected[0] is: %zu\n\n", proj[0].y.data[3]);

    printf("expected[0] is: %zu\n", proj[0].z.data[0]);
    printf("expected[0] is: %zu\n", proj[0].z.data[1]);
    printf("expected[0] is: %zu\n", proj[0].z.data[2]);
    printf("expected[0] is: %zu\n\n", proj[0].z.data[3]);
}

void print_jacobian(g1::element *jac) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for projective point
    printf("expected[0] is: %zu\n", jac[0].x.data[0]);
    printf("expected[0] is: %zu\n", jac[0].x.data[1]);
    printf("expected[0] is: %zu\n", jac[0].x.data[2]);
    printf("expected[0] is: %zu\n\n", jac[0].x.data[3]);

    printf("expected[0] is: %zu\n", jac[0].y.data[0]);
    printf("expected[0] is: %zu\n", jac[0].y.data[1]);
    printf("expected[0] is: %zu\n", jac[0].y.data[2]);
    printf("expected[0] is: %zu\n\n", jac[0].y.data[3]);

    printf("expected[0] is: %zu\n", jac[0].z.data[0]);
    printf("expected[0] is: %zu\n", jac[0].z.data[1]);
    printf("expected[0] is: %zu\n", jac[0].z.data[2]);
    printf("expected[0] is: %zu\n\n", jac[0].z.data[3]);
}

void print_field(var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for each limb
    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n\n", result[3]);
}

void execute_kernels
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *result, var *res_x, var *res_y, var *res_z) {
    // Allocate unified memory accessible by host and device
    g1::element *t1;
    g1::element *t2;
    g1::projective_element *t3;
    g1::element *jac;
    g1::affine_element *aff;
    g1::projective_element *proj;
    g1::element *expected;

    cudaMallocManaged(&t1, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&t2, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&t3, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&jac, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&aff, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&proj, 3 * 2 * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected, 3 * 2 * LIMBS * sizeof(uint64_t));

    // Initialize points
    initialize_add_check_against_constants<<<BLOCKS, THREADS>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z, t1, t2, expected);
    initialize_affine<<<BLOCKS, THREADS>>>(aff);

    // Affine conversion
    affine_to_projective<<<1, LIMBS>>>(aff, proj);
    cudaDeviceSynchronize();

    // Execute projective addition
    add_check_against_constants<<<1, LIMBS>>>(a, b, c, x, y, z, res_x, res_y, res_z, t1, proj, t3);
    print_projective(t3);

    // Compare results
    cudaDeviceSynchronize();
    comparator_kernel<<<1, LIMBS>>>(expected, t3, result);
    print_field(result);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to 'uint64_t' type
    var *a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *result, *res_x, *res_y, *res_z;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&c, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_z, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, c, x, y, z, expected_x, expected_y, expected_z, result, res_x, res_y, res_z);

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