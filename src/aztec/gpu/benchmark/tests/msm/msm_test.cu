#include "reference_string.cu"
#include <assert.h> 
#include <iostream>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

// Kernel launch parameters
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 4;
static constexpr size_t POINTS = 1 << 10;

/* -------------------------- Kernel Functions For Finite Field Tests ---------------------------------------------- */

// Initialize points and scalars
__global__ void initialize_simple_double_and_add_field(
uint64_t *a, uint64_t *b, uint64_t *expect_x) {
    fq_gpu point{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu scalar{ 0x09, 0x0, 0x0, 0x0 };
    fq_gpu expected_x{ 0xe57e2642f120824e, 0x34d7259cc9fe8db6, 0x46b12983f878ed43, 0x2b615a81474beec5 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = scalar.data[i];
        b[i] = point.data[i];
        expect_x[i] = expected_x.data[i];
    }
}

// Simple montgomery multiplication as a baseline reference
__global__ void simple_multiplication_field(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::to_monty(scalar[tid], scalar[tid]);  
        fq_gpu::mul(point[tid], scalar[tid], result[tid]); 
    }
}

// Native approach for computing scalar mutliplication with time complexity: O(2^k)
// nP = P + P ... + P 
__global__ void naive_double_and_add_field(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::add(point[tid], point[tid], result[tid]);     
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
        fq_gpu::add(point[tid], result[tid], result[tid]);
    }
}

// Double and add implementation using bit-decomposition with time complexity: O(2^k / 2)
__global__ void double_and_add_half_field(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    fq_gpu R;
    fq_gpu Q;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.data[tid]); 
        fq_gpu::load(point[tid], Q.data[tid]);

        // Loop unrolling
        fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);   // P
        fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]);   // 2P
        fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]);   // 4P
        fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]);   // 8P 
        fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);   // 9P 
    }
    
    // Store the final value of R into the result array for this limb
    fq_gpu::load(R.data[tid], result[tid]);
}

// Double and add implementation using bit-decomposition with time complexity: O(k)
__global__ void double_and_add_field(uint64_t *scalar, uint64_t *point, uint64_t *result) {
    fq_gpu R;
    fq_gpu Q;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.data[tid]); 
        fq_gpu::load(point[tid], Q.data[tid]);
        
        for (int i = 3; i >= 0; i--) {
            // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
            // and extracting the i-th bit of scalar in limb.
            if (((scalar[0] >> i) & 1) ? 1 : 0)
                fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);  
            if (i != 0) 
                fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]); 
        }
    }
    
    // Store the final value of R into the result array for this limb
    fq_gpu::load(R.data[tid], result[tid]);
}

/* -------------------------- Kernel Functions For Elliptic Curve Tests ---------------------------------------------- */

// Initialize points and scalars
__global__ void initialize_simple_double_and_add_curve(
uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *expect_x, uint64_t *expect_y, uint64_t *expect_z) {
    fq_gpu point_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu point_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu point_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu scalar{ 0x09, 0x0, 0x0, 0x0 };
    fq_gpu expected_x{ 0xb95b0df1fafbbf24, 0x848b28a0183c5cb7, 0x8158042f18cfd297, 0x124a5cddf43c0bc2 };
    fq_gpu expected_y{ 0x5769f9d04cd40953, 0x15f951f775281d56, 0x8b6b9be09b2bcd61, 0x1d2dbd94949735db };
    fq_gpu expected_z{ 0x5644e29729c3e1ce, 0xbf97116e02fc9117, 0x2f18c34822c7b2cd, 0x867c7e32dc19f38 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = point_x.data[i];
        b[i] = point_y.data[i];
        c[i] = point_z.data[i];
        d[i] = scalar.data[i];
        expect_x[i] = expected_x.data[i];
        expect_y[i] = expected_y.data[i];
        expect_z[i] = expected_z.data[i];
    }
}

// Native approach for computing scalar mutliplication with time complexity: O(2^k)
// nP = P + P ... + P 
__global__ void naive_double_and_add_curve(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, var *res_x, var *res_y, var *res_z) {
    g1::element ec;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::to_monty(a[tid], ec.x.data[tid]);
        fq_gpu::to_monty(b[tid], ec.y.data[tid]);
        fq_gpu::to_monty(c[tid], ec.z.data[tid]);

        // Jacobian addition
        g1::add(
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // ec + ec = 2ec
        if (fq_gpu::is_zero(res_x[tid]) && fq_gpu::is_zero(res_y[tid]) && fq_gpu::is_zero(res_z[tid])) {
            g1::doubling(
                ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
                res_x[tid], res_y[tid], res_z[tid]
            );
        }

        // 2ec + ec = 3ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 3ec + ec = 4ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 4ec + ec = 5ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 5ec + ec = 6ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 6ec + ec = 7ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 7ec + ec = 8ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 8ec + ec = 9ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);
    }
}

// Double and add implementation using bit-decomposition with time complexity: O(2^k / 2)
__global__ void double_and_add_half_curve(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, var *res_x, var *res_y, var *res_z) {
    g1::element ec;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::to_monty(a[tid], ec.x.data[tid]);
        fq_gpu::to_monty(b[tid], ec.y.data[tid]);
        fq_gpu::to_monty(c[tid], ec.z.data[tid]);

        // Jacobian addition
        g1::add(
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // ec + ec = 2ec
        if (fq_gpu::is_zero(res_x[tid]) && fq_gpu::is_zero(res_y[tid]) && fq_gpu::is_zero(res_z[tid])) {
            g1::doubling(
                ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
                res_x[tid], res_y[tid], res_z[tid]
            );
        }

        // 2ec + 2ec = 4ec
        g1::doubling(
            res_x[tid], res_y[tid], res_z[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 4ec + 4ec = 8ec
        g1::doubling(
            res_x[tid], res_y[tid], res_z[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // 8ec + ec = 9ec
        g1::add(
            res_x[tid], res_y[tid], res_z[tid], 
            ec.x.data[tid], ec.y.data[tid], ec.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);
    }
}

// Double and add implementation using bit-decomposition with time complexity: O(k)
__global__ void double_and_add_curve(
uint64_t *point_x, uint64_t *point_y, uint64_t *point_z, uint64_t *scalar, var *res_x, var *res_y, var *res_z) {
    g1::element R;
    g1::element Q;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.x.data[tid]); 
        fq_gpu::load(0, R.y.data[tid]); 
        fq_gpu::load(0, R.z.data[tid]); 

        fq_gpu::load(point_x[tid], Q.x.data[tid]);
        fq_gpu::load(point_y[tid], Q.y.data[tid]);
        fq_gpu::load(point_z[tid], Q.z.data[tid]);

        fq_gpu::to_monty(point_x[tid], Q.x.data[tid]);
        fq_gpu::to_monty(point_y[tid], Q.y.data[tid]);
        fq_gpu::to_monty(point_z[tid], Q.z.data[tid]);

        for (int i = 3; i >= 0; i--) {
            // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
            // and extracting the i-th bit of scalar in limb.
            if (((scalar[0] >> i) & 1) ? 1 : 0)
                g1::add(
                    R.x.data[tid], R.y.data[tid], R.z.data[tid], 
                    Q.x.data[tid], Q.y.data[tid], Q.z.data[tid], 
                    R.x.data[tid], R.y.data[tid], R.z.data[tid]
                );
            if (i != 0) 
                g1::doubling(
                    R.x.data[tid], R.y.data[tid], R.z.data[tid], 
                    R.x.data[tid], R.y.data[tid], R.z.data[tid]
                );
        }
    }
    
    // Store the final value of R into the result array for this limb
    fq_gpu::load(R.x.data[tid], res_x[tid]);
    fq_gpu::load(R.y.data[tid], res_y[tid]);
    fq_gpu::load(R.z.data[tid], res_z[tid]);

    fq_gpu::from_monty(res_x[tid], res_x[tid]);
    fq_gpu::from_monty(res_y[tid], res_y[tid]);
    fq_gpu::from_monty(res_z[tid], res_z[tid]);
}

/* -------------------------- Kernel Functions For Vector of Finite Field Tests ---------------------------------------------- */

// Convert result from montgomery form
__global__ void convert(fq_gpu *point, uint64_t *result) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::from_monty(point[0].data[tid], result[tid]);
    }

}

// Naive double and add using sequential implementation
__global__ void naive_double_and_add_field_vector_simple(fq_gpu *point, fq_gpu *result_vec, uint64_t *result) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    fq_gpu res{ 0, 0, 0, 0 };
    for (int i = 0; i < 1024; i++) {
        fq_gpu::add(res.data[tid], point[i].data[tid], res.data[tid]);
    }
    fq_gpu::load(res.data[tid], result[tid]);
    fq_gpu::from_monty(result[tid], result[tid]);
}

// Naive double and add using multiple kernel invocations with block-level grandularity
__global__ void naive_double_and_add_field_vector(fq_gpu *point, fq_gpu *result_vec, uint64_t *result) { 
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    fq_gpu::add(
        point[blockIdx.x * 2].data[threadIdx.x], point[(blockIdx.x * 2) + 1].data[threadIdx.x], result_vec[blockIdx.x].data[threadIdx.x]
    );
    
    if (threadIdx.x == 0) {
        fq_gpu::load(point[0].data[0], result[0]);
        fq_gpu::load(point[0].data[1], result[1]);
        fq_gpu::load(point[0].data[2], result[2]);
        fq_gpu::load(point[0].data[3], result[3]);
    }
}

/* -------------------------- Kernel Functions For Vector of Elliptic Curve Tests ---------------------------------------------- */


/* -------------------------- Helper Functions ---------------------------------------------- */

// Read curve points
template <class B>
B* read_points() {
    fq_gpu *points = new fq_gpu[POINTS];
    std::ifstream myfile ("../src/aztec/gpu/benchmark/tests/msm/points.txt"); 

    int i = 0;
    if ( myfile.is_open() ) {     
        for (size_t i = 0; i < POINTS * 4; ++i) {
            for (size_t j = 0; j < 4; j++) {
                myfile >> points[i].data[j];
            }
        }
    }
    return points;
} 

// Read scalars
template <class B>
B* read_scalars() {
    uint64_t temp[POINTS * 4];
    fr_gpu *scalars = new fr_gpu[POINTS];
    uint64_t scalar;

    // File stream
    ifstream stream;
    stream.open("../src/aztec/gpu/msm/scalars/scalars.txt", ios::in);

    // Read scalars
    if (stream.is_open()) {   
        int i = 0;  
        while (stream.good()) { 
            stream >> scalar;
            temp[i] = scalar;
            i++;
        }
    }
    
    for (size_t i = 0; i < (sizeof(temp) / sizeof(uint64_t)) / 4; ++i) {    
        fr_gpu element{ temp[i], temp[i + 1], temp[i + 2], temp[i + 3] };
        scalars[i] = element;        
    }
        
    return scalars;
}

// Print results
void print_field_tests(var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for each limb
    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n\n", result[3]);
}

void print_curve_tests(var *res_x, var *res_y, var *res_z) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for each limb
    printf("res_x[0] is: %zu\n", res_x[0]);
    printf("res_x[1] is: %zu\n", res_x[1]);
    printf("res_x[2] is: %zu\n", res_x[2]);
    printf("res_x[3] is: %zu\n\n", res_x[3]);

    printf("res_y[0] is: %zu\n", res_y[0]);
    printf("res_y[1] is: %zu\n", res_y[1]);
    printf("res_y[2] is: %zu\n", res_y[2]);
    printf("res_y[3] is: %zu\n\n", res_y[3]);

    printf("res_z[0] is: %zu\n", res_z[0]);
    printf("res_z[1] is: %zu\n", res_z[1]);
    printf("res_z[2] is: %zu\n", res_z[2]);
    printf("res_z[3] is: %zu\n\n", res_z[3]);
}

void print_field_vector_tests(fq_gpu *result_vec) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Print results for each limb
    printf("result[0] is: %zu\n", result_vec[0].data[0]);
    printf("result[1] is: %zu\n", result_vec[0].data[1]);
    printf("result[2] is: %zu\n", result_vec[0].data[2]);
    printf("result[3] is: %zu\n\n", result_vec[0].data[3]);
}

// Assert statements
void assert_checks(var *expected, var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();

    // Assert clause
    assert(expected[0] == result[0]);
    assert(expected[1] == result[1]);
    assert(expected[2] == result[2]);
    assert(expected[3] == result[3]);
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

// Execute kernel with finite field elements
void execute_kernels_finite_fields(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x,  var *expect_y,  var *expect_z) {    
    initialize_simple_double_and_add_field<<<BLOCKS, THREADS>>>(a, b, expect_x);
    
    double_and_add_field<<<BLOCKS, THREADS>>>(a, b, result);
    assert_checks(expect_x, result);
    print_field_tests(result);

    simple_multiplication_field<<<BLOCKS, THREADS>>>(a, b, result);
    assert_checks(expect_x, result);
    print_field_tests(result);

    naive_double_and_add_field<<<BLOCKS, THREADS>>>(a, b, result);
    assert_checks(expect_x, result);
    print_field_tests(result);
    
    double_and_add_half_field<<<BLOCKS, THREADS>>>(a, b, result);
    assert_checks(expect_x, result);
    print_field_tests(result);
}

/**
 * Execute kernel with curve elements
 */
void execute_kernels_curve(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    initialize_simple_double_and_add_curve<<<BLOCKS, THREADS>>>(a, b, c, d, expect_x, expect_y, expect_z);

    naive_double_and_add_curve<<<BLOCKS, THREADS>>>(a, b, c, d, res_x, res_y, res_z);
    print_curve_tests(res_x, res_y, res_z);

    double_and_add_half_curve<<<BLOCKS, THREADS>>>(a, b, c, d, res_x, res_y, res_z);
    assert_checks(expect_x, res_x);
    assert_checks(expect_y, res_y);
    assert_checks(expect_z, res_z);
    print_curve_tests(res_x, res_y, res_z);

    double_and_add_curve<<<BLOCKS, THREADS>>>(a, b, c, d, res_x, res_y, res_z);
    assert_checks(expect_x, res_x);
    assert_checks(expect_y, res_y);
    assert_checks(expect_z, res_z);
    print_curve_tests(res_x, res_y, res_z);
}

/**
 * Execute kernel with vector of finite field elements
 */
void execute_kernels_finite_fields_vector(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    // Read curve points and scalars
    fr_gpu *scalars = read_scalars<fr_gpu>();
    fq_gpu *points = read_points<fq_gpu>();

    // Define pointers to uint64_t type
    fq_gpu *points_alloc, *scalar_alloc, *result_vec;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&points_alloc, LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&scalar_alloc, LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&result_vec, LIMBS * POINTS * sizeof(var));

    // Load points
    for (int i = 0; i < POINTS; i++) {
        for (int j = 0; j < LIMBS; j++) {
            points_alloc[i].data[j] = points[i].data[j];
        }
    }

    // Load expected result
    expect_x[0] = 0x2ABC1016AF87ED0;
    expect_x[1] = 0xB606DF3AF98259F3;
    expect_x[2] = 0x9EE7391E20B296B4;
    expect_x[3] = 0x21E559B660EDBD92;

    naive_double_and_add_field_vector_simple<<<1, 4>>>(points_alloc, result_vec, result);
    assert_checks(expect_x, result);
    print_field_tests(result);

    // the issue with this is it requires log(NUM_POINTS) kernel invocations
    naive_double_and_add_field_vector<<<1024, 4>>>(points_alloc, result_vec, result);
    naive_double_and_add_field_vector<<<512, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<256, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<128, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<64, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<32, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<16, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<8, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<4, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<2, 4>>>(result_vec, result_vec, result);
    naive_double_and_add_field_vector<<<1, 4>>>(result_vec, result_vec, result);
    convert<<<1, 4>>>(result_vec, result);
    print_field_tests(result);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *c, *d, *result, *res_x, *res_y, *res_z, *expect_x, *expect_y, *expect_z;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS * sizeof(var));
    cudaMallocManaged(&b, LIMBS * sizeof(var));
    cudaMallocManaged(&c, LIMBS * sizeof(var));
    cudaMallocManaged(&d, LIMBS * sizeof(var));
    cudaMallocManaged(&result, LIMBS * sizeof(var));
    cudaMallocManaged(&res_x, LIMBS * sizeof(var));
    cudaMallocManaged(&res_y, LIMBS * sizeof(var));
    cudaMallocManaged(&res_z, LIMBS * sizeof(var));
    cudaMallocManaged(&expect_x, LIMBS * sizeof(var));
    cudaMallocManaged(&expect_y, LIMBS * sizeof(var));
    cudaMallocManaged(&expect_z, LIMBS * sizeof(var));

    // Execute kernel functions
    // execute_kernels_finite_fields(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);
    // execute_kernels_curve(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);
    execute_kernels_finite_fields_vector(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);

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