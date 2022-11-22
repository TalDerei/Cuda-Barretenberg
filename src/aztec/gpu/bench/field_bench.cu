#include "../fields/field.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

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

__global__ void initialize_mul_square_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x7d2e20e82f73d3e8, 0x8e50616a7a9d419d, 0xcdc833531508914b, 0xd510253a2ce62c };
    fq_gpu b_field{ 0x2829438b071fd14e, 0xb03ef3f9ff9274e, 0x605b671f6dc7b209, 0x8701f9d971fbc9 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void mul_square_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu t1;
    fq_gpu t2;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        t1.data[tid] = fq_gpu::sub(a[tid], b[tid], res[tid]);
        t2.data[tid] = fq_gpu::add(a[tid], b[tid], res[tid]);
        fq_gpu::mul(t1.data[tid], t2.data[tid], res[tid]);

        t1.data[tid] = fq_gpu::square(a[tid], res[tid]);
        t2.data[tid] = fq_gpu::square(b[tid], res[tid]);
        fq_gpu::sub(t1.data[tid], t2.data[tid], res[tid]);
    }
}

/* -------------------------- Multiply - Square Against Constants ---------------------------------------------- */

__global__ void initialize_sqr_check_against_constants(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x329596aa978981e8, 0x8542e6e254c2a5d0, 0xc5b687d82eadb178, 0x2d242aaf48f56b8a };
    fq_gpu b_field{ 0, 0, 0, 0 };
    fq_gpu const_expected{ 0xbf4fb34e120b8b12, 0xf64d70efbf848328, 0xefbb6a533f2e7d89, 0x1de50f941425e4aa };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void sqr_check_against_constants(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu t1;
    fq_gpu t2;
    
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        t1.data[tid] = fq_gpu::square(a[tid], res[tid]);
        t2.data[tid] = fq_gpu::square(b[tid], res[tid]);
        fq_gpu::sub(t1.data[tid], t2.data[tid], res[tid]);
    }
}

/* -------------------------- Add - Check Against Constants ---------------------------------------------- */

__global__ void initialize_add_check_against_constants(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x7d2e20e82f73d3e8, 0x8e50616a7a9d419d, 0xcdc833531508914b, 0xd510253a2ce62c };
    fq_gpu b_field{ 0x2829438b071fd14e, 0xb03ef3f9ff9274e, 0x605b671f6dc7b209, 0x8701f9d971fbc9 };
    fq_gpu const_expected{ 0xa55764733693a536, 0x995450aa1a9668eb, 0x2e239a7282d04354, 0x15c121f139ee1f6 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void add_check_against_constants(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::add(a[tid], b[tid], res[tid]);
    }
}

/* -------------------------- Subtract - Check Against Constants ---------------------------------------------- */

__global__ void initialize_sub_check_against_constants(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0xd68d01812313fb7c, 0x2965d7ae7c6070a5, 0x08ef9af6d6ba9a48, 0x0cb8fe2108914f53 };
    fq_gpu b_field{ 0x2cd2a2a37e9bf14a, 0xebc86ef589c530f6, 0x75124885b362b8fe, 0x1394324205c7a41d };
    fq_gpu const_expected{ 0xe5daeaf47cf50779, 0xd51ed34a5b0d0a3c, 0x4c2d9827a4d939a6, 0x29891a51e3fb4b5f };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void sub_check_against_constants(uint64_t *a, uint64_t *b, uint64_t *res) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::sub(a[tid], b[tid], res[tid]);
    }
}

/* -------------------------- Convert To Montgomery Form ---------------------------------------------- */

__global__ void initialize_to_montgomery_form(uint64_t *a, uint64_t *res) {
    fq_gpu a_field{ 0x01, 0x00, 0x00, 0x00 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
    }
}

__global__ void to_montgomery_form(uint64_t *a, uint64_t *res) {
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::to_monty(a[tid], res[tid]);
    }
}

/* -------------------------- Convert From Montgomery Form ---------------------------------------------- */

__global__ void initialize_from_montgomery_form(uint64_t *a, uint64_t *res) {
    fq_gpu a_field{ 0x01, 0x00, 0x00, 0x00 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
    }
}

__global__ void from_montgomery_form(uint64_t *a, uint64_t *res) {
    fq_gpu t;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        t.data[tid] = fq_gpu::to_monty(a[tid], res[tid]);
        fq_gpu::from_monty(t.data[tid], res[tid]);
    }
}

/* -------------------------- Montgomery Consistency Check ---------------------------------------------- */

__global__ void initialize_montgomery_consistency_check(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu b_field{ 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };
    fq_gpu const_expected{ 0x7ed4174114b521c4, 0x58f5bd1d4279fdc2, 0x6a73ac09ee843d41, 0x687a76ae9b3425c };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = b_field.data[i];
    }
}

__global__ void montgomery_consistency_check(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu aR;
    fq_gpu bR;
    fq_gpu aRR;
    fq_gpu bRR;
    fq_gpu bRRR;
    fq_gpu result_a;
    fq_gpu result_b;
    fq_gpu result_c;
    fq_gpu result_d;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        aR.data[tid] = fq_gpu::to_monty(a[tid], res[tid]);
        aRR.data[tid] = fq_gpu::to_monty(aR.data[tid], res[tid]);
        bR.data[tid] = fq_gpu::to_monty(b[tid], res[tid]);
        bRR.data[tid] = fq_gpu::to_monty(bR.data[tid], res[tid]);
        bRRR.data[tid] = fq_gpu::to_monty(bRR.data[tid], res[tid]);
        result_a.data[tid] = fq_gpu::mul(aRR.data[tid], bRR.data[tid], res[tid]); // abRRR
        result_b.data[tid] = fq_gpu::mul(aR.data[tid], bRRR.data[tid], res[tid]); // abRRR
        result_c.data[tid] = fq_gpu::mul(aR.data[tid], bR.data[tid], res[tid]);   // abR
        result_d.data[tid] = fq_gpu::mul(a[tid], b[tid], res[tid]);               // abR^-1
        fq_gpu::from_monty(result_a.data[tid], res[tid]); // abRR
        fq_gpu::from_monty(result_a.data[tid], res[tid]); // abR
        fq_gpu::from_monty(result_a.data[tid], res[tid]); // ab
        fq_gpu::from_monty(result_c.data[tid], res[tid]); // ab
        fq_gpu::from_monty(result_d.data[tid], res[tid]); // ab

        // EXPECT_EQ((result_a == result_b), true);
        // EXPECT_EQ((result_a == result_c), true);
        // EXPECT_EQ((result_a == result_d), true);
    }
}

/* -------------------------- Add Multiplication Consistency ---------------------------------------------- */

__global__ void initialize_add_mul_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu multiplicand = { 0x09, 0, 0, 0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = multiplicand.data[i];
    }
}

__global__ void add_mul_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu multiplicand;
    fq_gpu result;
    fq_gpu expected;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        multiplicand.data[tid] = fq_gpu::to_monty(b[tid], res[tid]);    
        result.data[tid] = fq_gpu::add(a[tid], a[tid], res[tid]);                       // 2
        result.data[tid] = fq_gpu::add(result.data[tid], result.data[tid], res[tid]);   // 4
        result.data[tid] = fq_gpu::add(result.data[tid], result.data[tid], res[tid]);   // 8
        result.data[tid] = fq_gpu::add(result.data[tid], a[tid], res[tid]);             // 9
        expected.data[tid] = fq_gpu::mul(a[tid], multiplicand.data[tid], res[tid]);     // 9
        
        // EXPECT_EQ((result == expected), true);
    }
}

/* -------------------------- Subtract Multiplication Consistency ---------------------------------------------- */

__global__ void initialize_sub_mul_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu multiplicand = { 0x05, 0, 0, 0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
        b[i] = multiplicand.data[i];
    }
}

__global__ void sub_mul_consistency(uint64_t *a, uint64_t *b, uint64_t *res) {
    fq_gpu multiplicand;
    fq_gpu result;
    fq_gpu expected;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        multiplicand.data[tid] = fq_gpu::to_monty(b[tid], res[tid]);    
        result.data[tid] = fq_gpu::add(a[tid], a[tid], res[tid]);                       // 2
        result.data[tid] = fq_gpu::add(result.data[tid], result.data[tid], res[tid]);   // 4
        result.data[tid] = fq_gpu::add(result.data[tid], result.data[tid], res[tid]);   // 8
        result.data[tid] = fq_gpu::sub(result.data[tid], a[tid], res[tid]);             // 7
        result.data[tid] = fq_gpu::sub(result.data[tid], a[tid], res[tid]);             // 6
        result.data[tid] = fq_gpu::sub(result.data[tid], a[tid], res[tid]);             // 5
        expected.data[tid] = fq_gpu::mul(a[tid], multiplicand.data[tid], res[tid]);     // 5
       
        // EXPECT_EQ((result == expected), true);
    }
}

/* -------------------------- Cube Root ---------------------------------------------- */

__global__ void initialize_cube(uint64_t *a, uint64_t *res) {
    fq_gpu a_field{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = a_field.data[i];
    }
}

__global__ void cube(uint64_t *a, uint64_t *res) {
    fq_gpu x_cubed;
    fq_gpu beta_x;
    fq_gpu beta_x_cubed;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        // compute x^3 and beta_x^3
        x_cubed.data[tid] = fq_gpu::mul(a[tid], a[tid], res[tid]);  
        x_cubed.data[tid] = fq_gpu::mul(x_cubed.data[tid], a[tid], res[tid]);  

        beta_x.data[tid] = fq_gpu::mul(a[tid], gpu_barretenberg::CUBE_ROOT_BASE[tid], res[tid]);  
        beta_x_cubed.data[tid] = fq_gpu::mul(beta_x.data[tid], beta_x.data[tid], res[tid]); 
        beta_x_cubed.data[tid] = fq_gpu::mul(beta_x_cubed.data[tid], beta_x.data[tid], res[tid]); 

        // EXPECT_EQ((x_cubed == beta_x_cubed), true);
    }
}

/* -------------------------- Main -- Executing Kernels ---------------------------------------------- */

void execute_kernels(var *a, var *b, var *res) {
    // Initialization kernels
    initialize_mont_mult<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_mont_mult_short<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_mul_square_consistency<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_sqr_check_against_constants<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_add_check_against_constants<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_sub_check_against_constants<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_to_montgomery_form<<<BLOCKS, THREADS>>>(a, res);
    initialize_from_montgomery_form<<<BLOCKS, THREADS>>>(a, res);
    initialize_montgomery_consistency_check<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_add_mul_consistency<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_sub_mul_consistency<<<BLOCKS, THREADS>>>(a, b, res);
    initialize_cube<<<BLOCKS, THREADS>>>(a, res);

    // Workload kernels
    mont_mult<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    mont_mult_short<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    mul_square_consistency<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    sqr_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    add_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    sub_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    to_montgomery_form<<<BLOCKS, LIMBS_NUM>>>(a, res);
    from_montgomery_form<<<BLOCKS, LIMBS_NUM>>>(a, res);
    montgomery_consistency_check<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    add_mul_consistency<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    sub_mul_consistency<<<BLOCKS, LIMBS_NUM>>>(a, b, res);
    cube<<<BLOCKS, LIMBS_NUM>>>(a, res);

    printf("result[0] is: %zu\n", res[0]);
    printf("result[1] is: %zu\n", res[1]);
    printf("result[2] is: %zu\n", res[2]);
    printf("result[3] is: %zu\n", res[3]);
}

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *res;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&res, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, res);

    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // End timer
    auto stop = high_resolution_clock::now();

    // Calculate duraion of execution time 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds\n" << endl; 

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(res);

    cout << "Completed sucessfully!" << endl;

    return 0;
}