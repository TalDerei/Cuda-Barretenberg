#include "field.cu"
#include <assert.h> 

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Montgomery Multiplication Test ---------------------------------------------- */

__global__ void initialize_mul(uint64_t *a, uint64_t *b, uint64_t *expected) {
    fr_gpu a_field{ 0x192f9ddc938ea63, 0x1db93d61007ec4fe, 0xc89284ec31fa49c0, 0x2478d0ff12b04f0f };
    fr_gpu b_field{ 0x7aade4892631231c, 0x8e7515681fe70144, 0x98edb76e689b6fd8, 0x5d0886b15fc835fa };
    fr_gpu expect{ 0xab961ef46b4756b6, 0xbc6b636fc29678c8, 0xd247391ed6b5bd16, 0x12e8538b3bde6784 };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
        expected[i] = expect.data[i];
    }
}

__global__ void mul(uint64_t *a, uint64_t *b, uint64_t *result) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fr_gpu::mul(a[tid], b[tid], result[tid]); 
    }
}

/* -------------------------- Square Test ---------------------------------------------- */

__global__ void initialize_sqr(uint64_t *a, uint64_t *expected) {
    fr_gpu a_field{ 0x95f946723a1fc34f, 0x641ec0482fc40bb9, 0xb8d645bc49dd513d, 0x1c1bffd317599dbc };
    fr_gpu b_field{ 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };
    fr_gpu expect{ 0xc787f7d9e2c72714, 0xcf21cf53d8f65f67, 0x8db109903dac0008, 0x26ab4dd65f46be5f };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_field.data[i];
        expected[i] = expect.data[i];
    }
}

__global__ void sqr(uint64_t *a, uint64_t *result) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fr_gpu::mul(a[tid], a[tid], result[tid]); 
    }
}

/* -------------------------- Add Test ---------------------------------------------- */

__global__ void initialize_add(uint64_t *a, uint64_t *b, uint64_t *expected) {
    fr_gpu a_field{ 0x20565a572c565a66, 0x7bccd0f01f5f7bff, 0x63ec2beaad64711f, 0x624953caaf44a814 };
    fr_gpu b_field{ 0xa17307a2108adeea, 0x74629976c14c5e2b, 0x9ce6f072ab1740ee, 0x398c753702b2bef0 };
    fr_gpu expect = { 0x7de76c654ce1394f, 0xc7fb821e66f26999, 0x4882d6a6d6fa59b0, 0x6b717a8ed0c5c6db };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
        expected[i] = expect.data[i];
    }
}

__global__ void add(uint64_t *a, uint64_t *b, uint64_t *result) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fr_gpu::add(a[tid], b[tid], result[tid]); 
    }
}

/* -------------------------- Subtract Test ---------------------------------------------- */

__global__ void initialize_subtract(uint64_t *a, uint64_t *b, uint64_t *expected) {
    fr_gpu a_field{ 0xcfbcfcf457cf2d38, 0x7b27af26ce62aa61, 0xf0378e90d48f2b92, 0x4734b22cb21ded };
    fr_gpu b_field{ 0x569fdb1db5198770, 0x446ddccef8347d52, 0xef215227182d22a, 0x8281b4fb109306 };
    fr_gpu expect{ 0xbcff176a92b5a5c9, 0x5eedbaa04fe79da0, 0x9995bf24e48db1c5, 0x3029017012d32b11 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
        expected[i] = expect.data[i];
    }
}

__global__ void subtract(uint64_t *a, uint64_t *b, uint64_t *result) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fr_gpu::sub(a[tid], b[tid], result[tid]); 
    }
}

/* -------------------------- Multiply - Square Against Constants ---------------------------------------------- */

__global__ void initialize_sqr_check_against_constants(uint64_t *a, uint64_t *expected) {
    fr_gpu a_field{ 0x329596aa978981e8, 0x8542e6e254c2a5d0, 0xc5b687d82eadb178, 0x2d242aaf48f56b8a };
    fr_gpu expecteded = { 0xbf4fb34e120b8b12, 0xf64d70efbf848328, 0xefbb6a533f2e7d89, 0x1de50f941425e4aa };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        expected[i] = expecteded.data[i];
    }
}

__global__ void sqr_check_against_constants(uint64_t *a, uint64_t *result) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fr_gpu::square(a[tid], result[tid]);
    }
}

/* -------------------------- Convert To Montgomery Form ---------------------------------------------- */

__global__ void initialize_to_montgomery_form(uint64_t *a, uint64_t *expected) {
    fr_gpu a_field{ 0x01, 0x00, 0x00, 0x00 };
    fr_gpu expecteded{ 0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0xe0a77c19a07df2f };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        expected[i] = expecteded.data[i];
    }
}

__global__ void to_montgomery_form(uint64_t *a, uint64_t *result) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fr_gpu::to_monty(a[tid], result[tid]);
    }
}

/* -------------------------- Convert From Montgomery Form ---------------------------------------------- */

__global__ void initialize_from_montgomery_form(uint64_t *a, uint64_t *expected) {
    fr_gpu a_field{ 0x01, 0x00, 0x00, 0x00 };
    fr_gpu expecteded{ 0x01, 0x00, 0x00, 0x00 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        expected[i] = expecteded.data[i];        
    }
}

__global__ void from_montgomery_form(uint64_t *a, uint64_t *result) {
    fr_gpu t1;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        t1.data[tid] = fr_gpu::to_monty(a[tid], result[tid]);
        fr_gpu::from_monty(t1.data[tid], result[tid]);
    }
}

/* -------------------------- Montgomery Consistency Check ---------------------------------------------- */

__global__ void initialize_montgomery_consistency_check(uint64_t *a, uint64_t *b) {
    fr_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fr_gpu b_field{ 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void montgomery_consistency_check(uint64_t *a, uint64_t *b, uint64_t *expected, uint64_t *result) {
    fr_gpu aR;
    fr_gpu bR;
    fr_gpu aRR;
    fr_gpu bRR;
    fr_gpu bRRR;
    fr_gpu result_a;
    fr_gpu result_b;
    fr_gpu result_c;
    fr_gpu result_d;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        aR.data[tid] = fr_gpu::to_monty(a[tid], result[tid]);
        aRR.data[tid] = fr_gpu::to_monty(aR.data[tid], result[tid]);
        bR.data[tid] = fr_gpu::to_monty(b[tid], result[tid]);
        bRR.data[tid] = fr_gpu::to_monty(bR.data[tid], result[tid]);
        bRRR.data[tid] = fr_gpu::to_monty(bRR.data[tid], result[tid]);

        result_a.data[tid] = fr_gpu::mul(aRR.data[tid], bRR.data[tid], result[tid]); // abRRR
        result_b.data[tid] = fr_gpu::mul(aR.data[tid], bRRR.data[tid], result[tid]); // abRRR
        result_c.data[tid] = fr_gpu::mul(aR.data[tid], bR.data[tid], result[tid]);   // abR
        result_d.data[tid] = fr_gpu::mul(a[tid], b[tid], result[tid]);               // abR^-1

        result_a.data[tid] = fr_gpu::from_monty(result_a.data[tid], result[tid]);    // abRR
        result_a.data[tid] = fr_gpu::from_monty(result_a.data[tid], result[tid]);    // abR
        result_a.data[tid] = fr_gpu::from_monty(result_a.data[tid], result[tid]);    // ab
        result_c.data[tid] = fr_gpu::from_monty(result_c.data[tid], expected[tid]);    // ab
        result_d.data[tid] = fr_gpu::to_monty(result_d.data[tid], expected[tid]);      // ab
    }
}

/* -------------------------- Add Multiplication Consistency ---------------------------------------------- */

__global__ void initialize_add_mul_consistency(uint64_t *a, uint64_t *b) {
    fr_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fr_gpu multiplicand = { 0x09, 0, 0, 0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = multiplicand.data[i];
    }
}

__global__ void add_mul_consistency(uint64_t *a, uint64_t *b, uint64_t *expected, uint64_t *res) {
    fr_gpu multiplicand;
    fr_gpu result;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        multiplicand.data[tid] = fr_gpu::to_monty(b[tid], res[tid]);    
        result.data[tid] = fr_gpu::add(a[tid], a[tid], res[tid]);                       // 2
        result.data[tid] = fr_gpu::add(result.data[tid], result.data[tid], res[tid]);   // 4
        result.data[tid] = fr_gpu::add(result.data[tid], result.data[tid], res[tid]);   // 8
        result.data[tid] = fr_gpu::add(result.data[tid], a[tid], res[tid]);             // 9

        fr_gpu::mul(a[tid], multiplicand.data[tid], expected[tid]);                     // 9        
    }
}

/* -------------------------- Subtract Multiplication Consistency ---------------------------------------------- */

__global__ void initialize_sub_mul_consistency(uint64_t *a, uint64_t *b) {
    fr_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fr_gpu multiplicand = { 0x05, 0, 0, 0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = multiplicand.data[i];
    }
}

__global__ void sub_mul_consistency(uint64_t *a, uint64_t *b, uint64_t *expected, uint64_t *res) {
    fr_gpu multiplicand;
    fr_gpu result;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        multiplicand.data[tid] = fr_gpu::to_monty(b[tid], res[tid]);    
        result.data[tid] = fr_gpu::add(a[tid], a[tid], res[tid]);                           // 2
        result.data[tid] = fr_gpu::add(result.data[tid], result.data[tid], res[tid]);       // 4
        result.data[tid] = fr_gpu::add(result.data[tid], result.data[tid], res[tid]);       // 8
        result.data[tid] = fr_gpu::sub(result.data[tid], a[tid], res[tid]);                 // 7
        result.data[tid] = fr_gpu::sub(result.data[tid], a[tid], res[tid]);                 // 6
        result.data[tid] = fr_gpu::sub(result.data[tid], a[tid], res[tid]);                 // 5

        fr_gpu::mul(a[tid], multiplicand.data[tid], expected[tid]);                         // 5       
    }
}

/* -------------------------- Cube Root ---------------------------------------------- */

__global__ void initialize_cube(uint64_t *a) {
    fr_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
    }
}

__global__ void cube(uint64_t *a, uint64_t *expected, uint64_t *result) {
    fr_gpu x_cubed;
    fr_gpu beta_x;
    fr_gpu beta_x_cubed;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        x_cubed.data[tid] = fr_gpu::mul(a[tid], a[tid], result[tid]);  
        x_cubed.data[tid] = fr_gpu::mul(x_cubed.data[tid], a[tid], result[tid]);  

        beta_x.data[tid] = fr_gpu::mul(a[tid], gpu_barretenberg::CUBE_ROOT_SCALAR[tid], expected[tid]);  
        beta_x_cubed.data[tid] = fr_gpu::mul(beta_x.data[tid], beta_x.data[tid], expected[tid]); 
        beta_x_cubed.data[tid] = fr_gpu::mul(beta_x_cubed.data[tid], beta_x.data[tid], expected[tid]); 
    }
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

    // Print statements
    // printf("expected[0] is: %zu\n", expected[0]);
    // printf("expected[1] is: %zu\n", expected[1]);
    // printf("expected[2] is: %zu\n", expected[2]);
    // printf("expected[3] is: %zu\n", expected[3]);
    // printf("result[0] is: %zu\n", result[0]);
    // printf("result[1] is: %zu\n", result[1]);
    // printf("result[2] is: %zu\n", result[2]);
    // printf("result[3] is: %zu\n", result[3]);
}

void execute_kernels(var *a, var *b, var *expected, var *result) {    
    // Montgomery Multiplication Test 
    initialize_mul<<<BLOCKS, THREADS>>>(a, b, expected);
    mul<<<BLOCKS, LIMBS_NUM>>>(a, b, result);
    assert_checks(expected, result);

    // Square Test
    initialize_sqr<<<BLOCKS, THREADS>>>(a, expected);
    sqr<<<BLOCKS, LIMBS_NUM>>>(a, result);
    assert_checks(expected, result);

    // Add Test
    initialize_add<<<BLOCKS, THREADS>>>(a, b, expected);
    add<<<BLOCKS, LIMBS_NUM>>>(a, b, result);
    assert_checks(expected, result);

    // Subtract Test
    initialize_subtract<<<BLOCKS, THREADS>>>(a, b, expected);
    subtract<<<BLOCKS, LIMBS_NUM>>>(a, b, result);
    assert_checks(expected, result);

    // Convert To Montgomery Form
    initialize_to_montgomery_form<<<BLOCKS, THREADS>>>(a, expected);
    to_montgomery_form<<<BLOCKS, LIMBS_NUM>>>(a, result);
    assert_checks(expected, result);

    // Convert From Montgomery Form
    initialize_from_montgomery_form<<<BLOCKS, THREADS>>>(a, expected);
    from_montgomery_form<<<BLOCKS, LIMBS_NUM>>>(a, result);
    assert_checks(expected, result);

    // Montgomery Consistency Check
    initialize_montgomery_consistency_check<<<BLOCKS, THREADS>>>(a, b);
    montgomery_consistency_check<<<BLOCKS, LIMBS_NUM>>>(a, b, expected, result);
    assert_checks(expected, result);

    // Add Multiplication Consistency
    initialize_add_mul_consistency<<<BLOCKS, THREADS>>>(a, b);
    add_mul_consistency<<<BLOCKS, LIMBS_NUM>>>(a, b, expected, result);
    assert_checks(expected, result);

    // Subtract Multiplication Consistency
    initialize_sub_mul_consistency<<<BLOCKS, THREADS>>>(a, b);
    sub_mul_consistency<<<BLOCKS, LIMBS_NUM>>>(a, b, expected, result);
    assert_checks(expected, result);

    // Cube Root
    initialize_cube<<<BLOCKS, THREADS>>>(a);
    cube<<<BLOCKS, LIMBS_NUM>>>(a, expected, result);
    assert_checks(expected, result);
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
    cout << "******* All 'Fr' unit tests passed! **********" << endl;

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