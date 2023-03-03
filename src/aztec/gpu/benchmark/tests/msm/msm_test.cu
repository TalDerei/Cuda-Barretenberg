#include "group.cu"
#include <assert.h> 
#include <iostream>
#include <algorithm>

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

// Kernel launch parameters
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 4;

/* -------------------------- Kernel Functions For Finite Field Tests ---------------------------------------------- */

// Initialize points and scalars
__global__ void initialize_simple_double_and_add_field(uint64_t *a, uint64_t *b) {
    fq_gpu point{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq_gpu scalar{ 0x09, 0x0, 0x0, 0x0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = scalar.data[i];
        b[i] = point.data[i];
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
__global__ void initialize_simple_double_and_add_curve(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d) {
    fq_gpu point_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu point_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu point_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu scalar{ 0x09, 0x0, 0x0, 0x0 };

    for (int i = 0; i < LIMBS; i++) {
        a[i] = point_x.data[i];
        b[i] = point_y.data[i];
        c[i] = point_z.data[i];
        d[i] = scalar.data[i];
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

/* -------------------------- Helper Functions ---------------------------------------------- */

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

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void execute_kernels(var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z) {    
    // Finite Field Tests
    initialize_simple_double_and_add_field<<<BLOCKS, THREADS>>>(a, b);
    simple_multiplication_field<<<BLOCKS, THREADS>>>(a, b, result);
    naive_double_and_add_field<<<BLOCKS, THREADS>>>(a, b, result);
    double_and_add_half_field<<<BLOCKS, THREADS>>>(a, b, result);
    double_and_add_field<<<BLOCKS, THREADS>>>(a, b, result);
    // print_field_tests(result);

    // Elliptic Curve Tests
    initialize_simple_double_and_add_curve<<<BLOCKS, THREADS>>>(a, b, c, d);
    naive_double_and_add_curve<<<BLOCKS, THREADS>>>(a, b, c, d, res_x, res_y, res_z);
    double_and_add_half_curve<<<BLOCKS, THREADS>>>(a, b, c, d, res_x, res_y, res_z);
    double_and_add_curve<<<BLOCKS, THREADS>>>(a, b, c, d, res_x, res_y, res_z);
    // print_curve_tests(res_x, res_y, res_z);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to uint64_t type
    var *a, *b, *c, *d, *result, *res_x, *res_y, *res_z;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS * sizeof(var));
    cudaMallocManaged(&b, LIMBS * sizeof(var));
    cudaMallocManaged(&c, LIMBS * sizeof(var));
    cudaMallocManaged(&d, LIMBS * sizeof(var));
    cudaMallocManaged(&result, LIMBS * sizeof(var));
    cudaMallocManaged(&res_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_z, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, c, d, result, res_x, res_y, res_z);

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


/*
Questions:
1. How to expand this to perform on an actual Fq point and Fr scalar?
    --> They operate over different prime fields, but Fr scalar doesn't participate 
    in the addition / multiplication calculations, only multiples of the Fq curve element. 

2. How to expand this to add an elliptic curve point G?
3. How to expand this to perform on a vector of points and scalars with larger kernel parameters?

Notes:
- Seems like barretenberg doesn't have methods to multipliy fq * g1 or g1 * g1, or fq * g1.x or g1.x * g1.x
- Performing a double and then add, e.g. ec + ec = 2ec, then 2ec + ec = 3ec yields the same results in gpu tests
and barretenberg. But ec + ec = 2ec, then 2ec + ec = 3ec, then 3ec + ec = 4ec isn't yielding the same results
as 2ec + 2ec with doubling. Need to investigate, because seems like a montgomery representation problem. 

Even on Barretenberg,the results might be different. It's converting everything to montgomery form before starting the calculation,
and the assert check still passes. Some conversions going on in the equality '==' check. 

To standardarize the results between barretenberg and my test suites, we'll do the following:
1. For FF code, don't need to convert to and from montgomery representation unless it's a multiplication operation.
2. For ECC code, always convert to and from montgomery representation code. 

We'll follow the same spec as Barretenberg, without all of the extra confusing operator overloading ops. 
*/

/**
 * TODO: Add expected results and assert statements
 * TODO: Resolve montgomery representation conflicts
*/
