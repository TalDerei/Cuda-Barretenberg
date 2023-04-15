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
__global__ void initialize_simple_double_and_add_curve_single(
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

// working double and add with a single point and scalar as baseline reference
__global__ void naive_multiplication_single(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, var *res_x, var *res_y, var *res_z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    g1::element one; 
    g1::element R;
    g1::element Q;

    fr_gpu exponent{ 0xb67299b792199cf0, 0xc1da7df1e7e12768, 0x692e427911532edf, 0x13dd85e87dc89978 };

    fq_gpu::load(gpu_barretenberg::one_x_bn_254[tid], one.x.data[tid]);
    fq_gpu::load(gpu_barretenberg::one_y_bn_254[tid], one.y.data[tid]);
    fq_gpu::load(fq_gpu::one().data[tid], one.z.data[tid]);

    if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.x.data[tid]); 
        fq_gpu::load(0, R.y.data[tid]); 
        fq_gpu::load(0, R.z.data[tid]); 

        fq_gpu::load(one.x.data[tid], Q.x.data[tid]);
        fq_gpu::load(one.y.data[tid], Q.y.data[tid]);
        fq_gpu::load(one.z.data[tid], Q.z.data[tid]);

        // Loop for each limb starting with the last limb
        for (int j = 3; j >= 0; j--) {
            // Loop for each bit of scalar
            for (int i = 64; i >= 0; i--) {
                // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
                // and extracting the i-th bit of scalar in limb.
                if (((exponent.data[j] >> i) & 1) ? 1 : 0)
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
    }

    // Store the final value of R into the result array for this limb
    fq_gpu::load(R.x.data[tid], res_x[tid]);
    fq_gpu::load(R.y.data[tid], res_y[tid]);
    fq_gpu::load(R.z.data[tid], res_z[tid]);

    // Convert back from montgomery form
    // fq_gpu::from_monty(res_x[tid], res_x[tid]);
    // fq_gpu::from_monty(res_y[tid], res_y[tid]);
    // fq_gpu::from_monty(res_z[tid], res_z[tid]);
}

// working double and add with multiple points and scalars as baseline reference
__global__ void naive_multiplication_multiple(fr_gpu *test_scalars, g1::element *test_points, g1::element *final_result) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;

    g1::element R;
    g1::element Q;

    // Initialize points and scalars
    fr_gpu scalar0 { 0xE49C36330BB35C4E, 0x22A5041C3B1B0B19, 0x37EDFE43AB6771EF, 0xCDA9012E9BF4459 };
    fr_gpu scalar1 { 0xCE3F41131DD7E353, 0xF089615FED1CDBFE, 0x9E0724A3FA817F99, 0xCB83233939D786B };
    fq_gpu point0_x { 0xD35D438DC58F0D9D, 0xA78EB28F5C70B3D, 0x666EA36F7879462C, 0xE0A77C19A07DF2F };
    fq_gpu point0_y { 0xA6BA871B8B1E1B3A, 0x14F1D651EB8E167B, 0xCCDD46DEF0F28C58, 0x1C14EF83340FBE5E };
    fq_gpu point0_z { 0xD35D438DC58F0D9D, 0xA78EB28F5C70B3D, 0x666EA36F7879462C, 0xE0A77C19A07DF2F };
    fq_gpu point1_x { 0x71930C11D782E155, 0xA6BB947CFFBE3323, 0xAA303344D4741444, 0x2C3B3F0D26594943 };
    fq_gpu point1_y { 0xD186911225DBDF54, 0x1A10FED0E5557E9E, 0xA3C3448E12102463, 0x44B3AD628E5381F4 };
    fq_gpu point1_z { 0xD35D438DC58F0D9D, 0xA78EB28F5C70B3D, 0x666EA36F7879462C, 0xE0A77C19A07DF2F };

    // Load points and scalars
    fq_gpu::load(scalar0.data[tid % 4], test_scalars[0].data[tid % 4]); 
    fq_gpu::load(scalar1.data[tid % 4], test_scalars[1].data[tid % 4]); 
    fq_gpu::load(point0_x.data[tid % 4], test_points[0].x.data[tid % 4]); 
    fq_gpu::load(point0_y.data[tid % 4], test_points[0].y.data[tid % 4]); 
    fq_gpu::load(point0_z.data[tid % 4], test_points[0].z.data[tid % 4]); 
    fq_gpu::load(point1_x.data[tid % 4], test_points[1].x.data[tid % 4]); 
    fq_gpu::load(point1_y.data[tid % 4], test_points[1].y.data[tid % 4]); 
    fq_gpu::load(point1_z.data[tid % 4], test_points[1].z.data[tid % 4]); 

    if (tid < LIMBS) {
        // Initialize result as 0
        fq_gpu::load(0, final_result[0].x.data[tid % 4]); 
        fq_gpu::load(0, final_result[0].y.data[tid % 4]); 
        fq_gpu::load(0, final_result[0].z.data[tid % 4]); 
        // Loop for each bucket module
        for (unsigned z = 0; z < 2; z++) {
            // Initialize 'R' to the identity element, Q to the curve point
            fq_gpu::load(0, R.x.data[tid % 4]); 
            fq_gpu::load(0, R.y.data[tid % 4]); 
            fq_gpu::load(0, R.z.data[tid % 4]); 

            // Load partial sums
            // change index to 'z'
            fq_gpu::load(test_points[z].x.data[tid % 4], Q.x.data[tid % 4]);
            fq_gpu::load(test_points[z].y.data[tid % 4], Q.y.data[tid % 4]);
            fq_gpu::load(test_points[z].z.data[tid % 4], Q.z.data[tid % 4]);

            // Sync loads
            __syncthreads();
    
            // Loop for each limb starting with the last limb
            for (int j = 3; j >= 0; j--) {
                // Loop for each bit of scalar
                for (int i = 64; i >= 0; i--) {   
                    // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
                    // and extracting the i-th bit of scalar in limb.
                    if (((test_scalars[z].data[j] >> i) & 1) ? 1 : 0)
                        g1::add(
                            Q.x.data[tid % 4], Q.y.data[tid % 4], Q.z.data[tid % 4], 
                            R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4], 
                            R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4]
                        );
                    if (i != 0) 
                        g1::doubling(
                            R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4], 
                            R.x.data[tid % 4], R.y.data[tid % 4], R.z.data[tid % 4]
                        );
                }
            }
            g1::add(
                R.x.data[tid % 4], 
                R.y.data[tid % 4], 
                R.z.data[tid % 4],
                final_result[0].x.data[tid % 4],
                final_result[0].y.data[tid % 4],
                final_result[0].z.data[tid % 4],
                final_result[0].x.data[tid % 4], 
                final_result[0].y.data[tid % 4], 
                final_result[0].z.data[tid % 4]
            );
        }
    }

    // Convert back from montgomery form
    // fq_gpu::from_monty(final_result[0].x.data[tid % 4], final_result[0].x.data[tid % 4]);
    // fq_gpu::from_monty(final_result[0].y.data[tid % 4], final_result[0].y.data[tid % 4]);
    // fq_gpu::from_monty(final_result[0].z.data[tid % 4], final_result[0].z.data[tid % 4]);
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
__global__ void convert_field(fq_gpu *point, uint64_t *result) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::from_monty(point[0].data[tid], result[tid]);
    }
}

__global__ void convert_curve(g1::element *point, uint64_t *res_x, uint64_t *res_y, uint64_t *res_z) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::from_monty(point[0].x.data[tid], res_x[tid]);
        fq_gpu::from_monty(point[0].y.data[tid], res_y[tid]);
        fq_gpu::from_monty(point[0].z.data[tid], res_z[tid]);
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
    fq_gpu::add(
        point[blockIdx.x * 2].data[threadIdx.x], point[(blockIdx.x * 2) + 1].data[threadIdx.x], result_vec[blockIdx.x].data[threadIdx.x]
    );

    __syncthreads();
    
    // Accumulate result into current block
    if (threadIdx.x == 0) {
        fq_gpu::load(result_vec[0].data[0], result[0]);
        fq_gpu::load(result_vec[0].data[1], result[1]);
        fq_gpu::load(result_vec[0].data[2], result[2]);
        fq_gpu::load(result_vec[0].data[3], result[3]);
    }
}

/* -------------------------- Kernel Functions For Vector of Elliptic Curve Tests ---------------------------------------------- */

// Naive double and add using sequential implementation
__global__ void naive_double_and_add_curve_vector_simple(g1::element *point, g1::element *result_vec, uint64_t *res_x, uint64_t *res_y,  uint64_t *res_z) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    g1::element res;
    fq_gpu res_x_temp{ 0, 0, 0, 0 };
    fq_gpu res_y_temp{ 0, 0, 0, 0 };
    fq_gpu res_z_temp{ 0, 0, 0, 0 };

    fq_gpu::load(res_x_temp.data[tid], res.x.data[tid]);
    fq_gpu::load(res_y_temp.data[tid], res.y.data[tid]);
    fq_gpu::load(res_z_temp.data[tid], res.z.data[tid]);

    fq_gpu::to_monty(res.x.data[tid], res.x.data[tid]);
    fq_gpu::to_monty(res.x.data[tid], res.y.data[tid]);
    fq_gpu::to_monty(res.x.data[tid], res.z.data[tid]);

    for (int i = 0; i < 1024; i++) {        
        g1::add(
            res.x.data[tid], res.y.data[tid], res.z.data[tid], 
            point[i].x.data[tid], point[i].y.data[tid], point[i].z.data[tid], 
            res.x.data[tid], res.y.data[tid], res.z.data[tid]
        );
    }
    
    fq_gpu::load(res.x.data[tid], res_x[tid]);
    fq_gpu::load(res.y.data[tid], res_y[tid]);
    fq_gpu::load(res.z.data[tid], res_z[tid]);

    fq_gpu::load(res.x.data[tid], result_vec[0].x.data[tid]);
    fq_gpu::load(res.y.data[tid], result_vec[0].y.data[tid]);
    fq_gpu::load(res.z.data[tid], result_vec[0].z.data[tid]);

    fq_gpu::from_monty(res_x[tid], res_x[tid]);
    fq_gpu::from_monty(res_y[tid], res_y[tid]);
    fq_gpu::from_monty(res_z[tid], res_z[tid]);
}

// Naive double and add using multiple kernel invocations with block-level grandularity
__global__ void naive_double_and_add_curve_vector(g1::element *point, g1::element *result_vec, uint64_t *res_x,  uint64_t *res_y,  uint64_t *res_z) {     
    g1::add(
        point[blockIdx.x * 2].x.data[threadIdx.x], point[blockIdx.x * 2].y.data[threadIdx.x], point[blockIdx.x * 2].z.data[threadIdx.x],
        point[(blockIdx.x * 2) + 1].x.data[threadIdx.x], point[(blockIdx.x * 2) + 1].y.data[threadIdx.x], point[(blockIdx.x * 2) + 1].z.data[threadIdx.x],
        result_vec[blockIdx.x].x.data[threadIdx.x], result_vec[blockIdx.x].y.data[threadIdx.x], result_vec[blockIdx.x].z.data[threadIdx.x]
    );
}

// Compare two elliptic curve elements
__global__ void comparator_kernel(g1::element *point, g1::element *point_2, uint64_t *result) {     
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

/* -------------------------- Kernel Functions For Vector of Finite Field With Scalars Tests ---------------------------------------------- */

// Naive double and add using sequential implementation using scalars
__global__ void naive_double_and_add_field_vector_with_scalars(fq_gpu *point, fr_gpu *scalar, fq_gpu *result_vec, uint64_t *result) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    fq_gpu temp{ 0, 0, 0, 0 };
    fq_gpu temp_accumulator{ 0, 0, 0, 0 };

    for (int i = 0; i < 1024; i++) {
        fq_gpu::mul(point[i].data[tid], scalar[i].data[tid], temp.data[tid]);
        fq_gpu::add(temp.data[tid], temp_accumulator.data[tid], temp_accumulator.data[tid]);
    }
    fq_gpu::load(temp_accumulator.data[tid], result_vec[0].data[tid]);
    fq_gpu::from_monty(result_vec[0].data[tid], result_vec[0].data[tid]);
    fq_gpu::load(temp_accumulator.data[tid], result[tid]);
    fq_gpu::from_monty(result[tid], result[tid]);
}

// Double and add implementation using bit-decomposition with time complexity: O(k)
__global__ void double_and_add_field_vector_with_scalars(fq_gpu *point, fr_gpu *scalar, uint64_t *result) {
    // Holders for points and scalars
    fq_gpu R;
    fq_gpu Q;
    
    // Inner and outer accumulators
    fq_gpu inner_accumulator;
    fq_gpu outer_accumulator;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        // Initialize the inner and outer accumulator that will store intermediate results
        fq_gpu::load(0, inner_accumulator.data[tid]); 
        fq_gpu::load(0, outer_accumulator.data[tid]); 

        // Loop over all curve points and scalars
        // Does indexing with int vs uint64_t or size_t matter?
        for (int i = 0; i < POINTS; i++) {
            // Load Q with curve point 'i', and 'R' with the identity element
            fq_gpu::load(point[i].data[tid], Q.data[tid]);
            // Loop for each limb
            for (int j = 0; j < LIMBS; j++) {
                fq_gpu::load(0, R.data[tid]); 
                // Loop for each bit of scalar
                for (int z = 63; z >= 0; z--) {
                    // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB,
                    // and extracting the i-th bit of scalar in limb.
                    if (((scalar[i].data[j] >> j) & 1) ? 1 : 0)
                        fq_gpu::add(R.data[tid], Q.data[tid], R.data[tid]);  
                    if (z != 0) 
                        // z != 0 assumes the odd case -- need to calculate if this is neccessary?
                        fq_gpu::add(R.data[tid], R.data[tid], R.data[tid]); 
                }
                // Inner accumulator 
                fq_gpu::add(R.data[tid], inner_accumulator.data[tid], inner_accumulator.data[tid]);
            }
            // Outer accumulator 
            fq_gpu::add(inner_accumulator.data[tid], outer_accumulator.data[tid], outer_accumulator.data[tid]);
        }
    }
    
    // Store the final value of R into the result array for this limb
    fq_gpu::load(outer_accumulator.data[tid], result[tid]);
}

/* -------------------------- Kernel Functions For Vector of Finite Field With Scalars Tests ---------------------------------------------- */

// Naive double and add using sequential implementation using scalars
__global__ void naive_double_and_add_curve_with_scalars(
g1::element *point, fr_gpu *scalar, g1::element *result_vec, uint64_t *res_x, uint64_t *res_y, uint64_t *res_z) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    g1::element temp;
    g1::element temp_accumulator;
    fq_gpu res_x_temp{ 0, 0, 0, 0 };
    fq_gpu res_y_temp{ 0, 0, 0, 0 };
    fq_gpu res_z_temp{ 0, 0, 0, 0 };

    fq_gpu::load(res_x_temp.data[tid], temp.x.data[tid]);
    fq_gpu::load(res_y_temp.data[tid], temp.y.data[tid]);
    fq_gpu::load(res_z_temp.data[tid], temp.z.data[tid]);
    fq_gpu::load(res_x_temp.data[tid], temp_accumulator.x.data[tid]);
    fq_gpu::load(res_y_temp.data[tid], temp_accumulator.y.data[tid]);
    fq_gpu::load(res_z_temp.data[tid], temp_accumulator.z.data[tid]);

    for (int i = 0; i < 1024; i++) {
        fq_gpu::mul(point[i].x.data[tid], scalar[i].data[tid], temp.x.data[tid]);
        fq_gpu::mul(point[i].y.data[tid], scalar[i].data[tid], temp.y.data[tid]);
        fq_gpu::mul(point[i].z.data[tid], scalar[i].data[tid], temp.z.data[tid]);
        fq_gpu::add(temp.x.data[tid], temp_accumulator.x.data[tid], temp_accumulator.x.data[tid]);
        fq_gpu::add(temp.y.data[tid], temp_accumulator.y.data[tid], temp_accumulator.y.data[tid]);
        fq_gpu::add(temp.z.data[tid], temp_accumulator.z.data[tid], temp_accumulator.z.data[tid]);
    }
    
    fq_gpu::load(temp_accumulator.x.data[tid], result_vec[0].x.data[tid]);
    fq_gpu::load(temp_accumulator.y.data[tid], result_vec[0].y.data[tid]);
    fq_gpu::load(temp_accumulator.z.data[tid], result_vec[0].z.data[tid]);

    fq_gpu::from_monty(result_vec[0].x.data[tid], result_vec[0].x.data[tid]);
    fq_gpu::from_monty(result_vec[0].y.data[tid], result_vec[0].y.data[tid]);
    fq_gpu::from_monty(result_vec[0].z.data[tid], result_vec[0].z.data[tid]);   

    fq_gpu::load(result_vec[0].x.data[tid], res_x[tid]);
    fq_gpu::load(result_vec[0].y.data[tid], res_y[tid]);
    fq_gpu::load(result_vec[0].z.data[tid], res_z[tid]); 
}

/* -------------------------- Helper Functions ---------------------------------------------- */

// Read field points
template <class B>
B* read_field_points() {
    fq_gpu *points = new fq_gpu[POINTS];
    std::ifstream myfile ("../src/aztec/gpu/benchmark/tests/msm/points/field_points.txt"); 

    if ( myfile.is_open() ) {     
        for (size_t i = 0; i < POINTS * 4; ++i) {
            for (size_t j = 0; j < 4; j++) {
                myfile >> points[i].data[j];
            }
        }
    }
    return points;
} 

// Read curve points
template <class B>
B* read_curve_points() {
    g1::element *points = new g1::element[3 * LIMBS * POINTS * sizeof(var)];
    std::ifstream myfile ("../src/aztec/gpu/benchmark/tests/msm/points/curve_points.txt"); 

    if ( myfile.is_open() ) {   
        for (size_t i = 0; i < POINTS; i++) {
            for (size_t j = 0; j < 4; j++) {
                myfile >> points[i].x.data[j];
            }
            for (size_t y = 0; y < 4; y++) {
                myfile >> points[i].y.data[y];
            }
            for (size_t z = 0; z < 4; z++) {
                myfile >> points[i].z.data[z];
            }
        }   
    }
    myfile.close();
    return points;
} 

// Read scalars
template <class B>
B* read_scalars() {
    fr_gpu *scalars = new fr_gpu[POINTS];

    // File stream
    ifstream stream;
    stream.open("../src/aztec/gpu/msm/points/scalars.txt", ios::in);

    // Read scalars
    if ( stream.is_open() ) {   
        for (size_t i = 0; i < POINTS; i++) {
            for (size_t j = 0; j < 4; j++) {
                stream >> scalars[i].data[j];
            }
        }   
    }
    stream.close();
        
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

// Execute kernel with curve elements
void execute_kernels_curve(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
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

// Execute kernel with vector of finite field elements
void execute_kernels_finite_fields_vector(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    // Read curve points
    fq_gpu *points = read_field_points<fq_gpu>();

    // Define pointers to uint64_t type
    fq_gpu *points_alloc, *result_vec;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&points_alloc, LIMBS * POINTS * sizeof(var));
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

    // Kernel invocation 1
    naive_double_and_add_field_vector_simple<<<1, 4>>>(points_alloc, result_vec, result);
    assert_checks(expect_x, result);
    print_field_tests(result);

    // Kernel invocation 2
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
    convert_field<<<1, 4>>>(result_vec, result);
    print_field_tests(result);
}

// Execute kernel with vector of finite field elements
void execute_kernels_curve_vector(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    // Read curve points
    g1::element *points = read_curve_points<g1::element>();

    // Define pointers to g1::element type
    g1::element *points_alloc, *result_vec_1, *result_vec_2;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&points_alloc, 3 * LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&result_vec_1, 3 * LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&result_vec_2, 3 * LIMBS * POINTS * sizeof(var));
 
    // Load curve elements 
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < LIMBS; j++) {
            points_alloc[i].x.data[j] = points[i].x.data[j];
            points_alloc[i].y.data[j] = points[i].y.data[j];
            points_alloc[i].z.data[j] = points[i].z.data[j];
        }
    }

    // Load expected result for 1024 points
    expect_x[0] = 0x80593DFBD43FBD19;
    expect_x[1] = 0x3D92F6F9AEF80647;
    expect_x[2] = 0x91BF58F40BC30FF8;
    expect_x[3] = 0x2686099C3F32E11B;

    expect_y[0] = 0xE68A1568BFC92B69;
    expect_y[1] = 0x749D8B0CDED593A8;
    expect_y[2] = 0x9038B56B64A1BDBD;
    expect_y[3] = 0x1E9B10F7538FBC6E;
    
    expect_z[0] = 0xB276DE627AF13A05;
    expect_z[1] = 0x86B9806C6F057AC6;
    expect_z[2] = 0x19440082AE9936D8;
    expect_z[3] = 0x14E512C471B5CDD4;

    // Kernel invocation 1
    naive_double_and_add_curve_vector_simple<<<1, 4>>>(points_alloc, result_vec_1, res_x, res_y, res_z);
    assert_checks(expect_x, res_x);
    assert_checks(expect_y, res_y);
    assert_checks(expect_z, res_z);
    print_curve_tests(res_x, res_y, res_z);

    // Kernel invocation 2
    naive_double_and_add_curve_vector<<<512, 4>>>(points_alloc, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<256, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<128, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<64, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<32, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<16, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<8, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<4, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<2, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    naive_double_and_add_curve_vector<<<1, 4>>>(result_vec_2, result_vec_2, res_x, res_y, res_z);
    convert_curve<<<1, 4>>>(result_vec_2, res_x, res_y, res_z);
    print_curve_tests(res_x, res_y, res_z);

    // Compare results
    cudaDeviceSynchronize();
    comparator_kernel<<<1, 4>>>(result_vec_1, result_vec_2, result);
    print_field_tests(result);
}

// Execute kernel with vector of finite field elements with scalars
void execute_kernels_finite_fields_vector_with_scalars(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    // Read curve points and scalars
    fr_gpu *scalars = read_scalars<fr_gpu>();
    fq_gpu *points = read_field_points<fq_gpu>();

    // Define pointers to uint64_t type
    fq_gpu *points_alloc, *result_vec_1, *result_vec_2;
    fr_gpu *scalars_alloc;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&points_alloc, LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&scalars_alloc, LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&result_vec_1, LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&result_vec_2, LIMBS * POINTS * sizeof(var));

    // Load field elements 
    for (int i = 0; i < POINTS; i++) {
        for (int j = 0; j < LIMBS; j++) {
            points_alloc[i].data[j] = points[i].data[j];
        }
    }

    // Load scalars
    for (int i = 0; i < POINTS; i++) {
        for (int j = 0; j < LIMBS; j++) {
            scalars_alloc[i].data[j] = scalars[i].data[j];
        }
    }

    // Load expected result
    expect_x[0] = 0x5B5ECFF24EEE567B;
    expect_x[1] = 0xE0E4CCD174FA7CD3;
    expect_x[2] = 0x2EDFE1054FF06F7D;
    expect_x[3] = 0x93075569BFD611A;

    // Kernel invocation 1
    naive_double_and_add_field_vector_with_scalars<<<1, 4>>>(points_alloc, scalars_alloc, result_vec_1, result);
    assert_checks(expect_x, result);
    print_field_tests(result);

    double_and_add_field_vector_with_scalars<<<BLOCKS, THREADS>>>(points_alloc, scalars_alloc, result);
    print_field_tests(result);
}

// Execute kernel with vector of curve elements with scalars
void execute_kernels_curve_vector_with_scalars(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    // Read curve points and scalars
    fr_gpu *scalars = read_scalars<fr_gpu>();
    g1::element *points = read_curve_points<g1::element>();

    // auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(POINTS, "../srs_db");
    // g1::affine_element* points = reference_string->get_monomials();

    // Define pointers to uint64_t type
    g1::element *points_alloc, *result_vec;
    fr_gpu *scalars_alloc;

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&points_alloc, 3 * LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&scalars_alloc, LIMBS * POINTS * sizeof(var));
    cudaMallocManaged(&result_vec, 3 * LIMBS * POINTS * sizeof(var));

    // Load curve elements 
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < LIMBS; j++) {
            points_alloc[i].x.data[j] = points[i].x.data[j];
            points_alloc[i].y.data[j] = points[i].y.data[j];
            points_alloc[i].z.data[j] = points[i].z.data[j];
        }
    }

    // Load scalars
    for (int i = 0; i < POINTS; i++) {
        for (int j = 0; j < LIMBS; j++) {
            scalars_alloc[i].data[j] = scalars[i].data[j];
        }
    }

    // Load expected result for 1024 points
    expect_x[0] = 0xC651AE98201F2ED5;
    expect_x[1] = 0x7FCBB8625702746E;
    expect_x[2] = 0xEA1323D929C8744;
    expect_x[3] = 0x299F0951AE8445B8;

    expect_y[0] = 0x9471B80AA7A54D0B;
    expect_y[1] = 0x985A8A5A3CA5C1EB;
    expect_y[2] = 0x3CF789E4252D1EE8;
    expect_y[3] = 0xECCA07E625BB37D;
    
    expect_z[0] = 0x88FE75761CA8C8DF;
    expect_z[1] = 0x996914266C5DE61A;
    expect_z[2] = 0x53745CEECA7D48FE;
    expect_z[3] = 0x293E5E8A3728B7C6;

    // Kernel invocation 1
    naive_double_and_add_curve_with_scalars<<<1, 4>>>(points_alloc, scalars_alloc, result_vec, res_x, res_y, res_z);
    assert_checks(expect_x, res_x);
    assert_checks(expect_y, res_y);
    assert_checks(expect_z, res_z);
    print_curve_tests(res_x, res_y, res_z);
}

/* -------------------------- Executing Double-And-Add Functions ---------------------------------------------- */

// Execute kernel with curve elements
void execute_double_and_add_single(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    initialize_simple_double_and_add_curve_single<<<BLOCKS, THREADS>>>(a, b, c, d, expect_x, expect_y, expect_z);

    naive_multiplication_single<<<BLOCKS, THREADS>>>(a, b, c, d, res_x, res_y, res_z);
    cudaDeviceSynchronize();
    print_curve_tests(res_x, res_y, res_z);

    g1::element *final_res;
    g1::element *expected_1;
    g1::element *expected_2;
    var *result_1;
    var *result_2;
    cudaMallocManaged(&final_res, 3 * LIMBS * 1 * sizeof(var));
    cudaMallocManaged(&expected_1, 3 * LIMBS * 1 * sizeof(var));
    cudaMallocManaged(&expected_2, 3 * LIMBS * 1 * sizeof(var));
    cudaMallocManaged(&result_1, LIMBS * 1 * sizeof(var));
    cudaMallocManaged(&result_2, LIMBS * 1 * sizeof(var));

    // Convert final result
    final_res[0].x.data[0] = res_x[0];
    final_res[0].x.data[1] = res_x[1];
    final_res[0].x.data[2] = res_x[2];
    final_res[0].x.data[3] = res_x[3];
    final_res[0].y.data[0] = res_y[0];
    final_res[0].y.data[1] = res_y[1];
    final_res[0].y.data[2] = res_y[2];
    final_res[0].y.data[3] = res_y[3];
    final_res[0].z.data[0] = res_z[0];
    final_res[0].z.data[1] = res_z[1];
    final_res[0].z.data[2] = res_z[2];
    final_res[0].z.data[3] = res_z[3];

    // Expected results from naive double-and-add kernel from Barretenberg
    expected_1[0].x.data[0] = 0x9C9320BC891ED9DE;
    expected_1[0].x.data[1] = 0xACE55F06AA29C3F2;
    expected_1[0].x.data[2] = 0x24DB84E75A391315;
    expected_1[0].x.data[3] = 0x595DBA53EFD6FD5B;
    expected_1[0].y.data[0] = 0x9ECB0640B15EC4D0;
    expected_1[0].y.data[1] = 0x7B9C653CA35FA1AE;
    expected_1[0].y.data[2] = 0x5B387CDD03D7F5EA;
    expected_1[0].y.data[3] = 0x4FCFDCA5887EEB8E;
    expected_1[0].z.data[0] = 0xA35A8F1E4C0E6A4F;
    expected_1[0].z.data[1] = 0x6715FFC8177D607C;
    expected_1[0].z.data[2] = 0x71EFC8CAC5C4F073;
    expected_1[0].z.data[3] = 0x2BFA6FF080354472;

    // Expected results from custom MSM kernel
    expected_2[0].x.data[0] = 0xE62BB32C93AA0F8A;
    expected_2[0].x.data[1] = 0x97CEA47D1C9918D8;
    expected_2[0].x.data[2] = 0x6E4F6B103D5CB238;
    expected_2[0].x.data[3] = 0xCC97B85B5D3266B;
    expected_2[0].y.data[0] = 0x384B9977174D6D23;
    expected_2[0].y.data[1] = 0x9EB9C140BF105B1;
    expected_2[0].y.data[2] = 0xEF83689C8B6B86AB;
    expected_2[0].y.data[3] = 0x1DEE77175CFD916E;
    expected_2[0].z.data[0] = 0x51EC9058273498D3;
    expected_2[0].z.data[1] = 0x54FA2618A1043D98;
    expected_2[0].z.data[2] = 0x86005E4D5FEBEB41;
    expected_2[0].z.data[3] = 0x171CBD46E7C215D3;

    comparator_kernel<<<1, 4>>>(final_res, expected_1, result_1);
    comparator_kernel<<<1, 4>>>(final_res, expected_2, result_2);
    cudaDeviceSynchronize();
    print_field_tests(result_1);
    print_field_tests(result_2);
}

// Execute kernel with curve elements
void execute_double_and_add_multiple(
var *a, var *b, var *c, var *d, var *result, var *res_x, var *res_y, var *res_z, var *expect_x, var *expect_y, var *expect_z) {    
    // Allocate unified memory accessible by host and device
    fr_gpu *scalars;
    g1::element *points;
    g1::element *final_result;
    g1::element *expected_1;
    g1::element *expected_2;
    var *result_1;
    var *result_2;
    cudaMallocManaged(&scalars, 2 * LIMBS * sizeof(var));
    cudaMallocManaged(&points, 2 * 3 * LIMBS * sizeof(var));
    cudaMallocManaged(&final_result, 3 * LIMBS * sizeof(var));
    cudaMallocManaged(&expected_1, 3 * LIMBS * sizeof(var));
    cudaMallocManaged(&expected_2, 3 * LIMBS * sizeof(var));
    cudaMallocManaged(&result_1, LIMBS * sizeof(var));
    cudaMallocManaged(&result_2, LIMBS * sizeof(var));

    naive_multiplication_multiple<<<BLOCKS, THREADS>>>(scalars, points, final_result);
    cudaDeviceSynchronize();

    // Expected results from custom MSM kernel
    expected_1[0].x.data[0] = 0x7D08D399C74D6F60;
    expected_1[0].x.data[1] = 0x8B7E6B6EB490841B;
    expected_1[0].x.data[2] = 0xD4D3B85D62A522F0;
    expected_1[0].x.data[3] = 0xA88CC30AEC67D21;
    expected_1[0].y.data[0] = 0x4760C013E773E092;
    expected_1[0].y.data[1] = 0x27AF313043E09A67;
    expected_1[0].y.data[2] = 0x1B1C6848E1C81D14;
    expected_1[0].y.data[3] = 0xD404C6D6230ABCF;
    expected_1[0].z.data[0] = 0x8CEA309B633D24D2;
    expected_1[0].z.data[1] = 0xC02A4EA980DA9C0;
    expected_1[0].z.data[2] = 0x4C73226DC94B0750;
    expected_1[0].z.data[3] = 0x148BF6C82971F1A0;

    // Expected results from naive double-and-add kernel
    expected_2[0].x.data[0] = 0x22D564276729916F;
    expected_2[0].x.data[1] = 0xCE84C2A015B18E50;
    expected_2[0].x.data[2] = 0xB609ED7D21C3346B;
    expected_2[0].x.data[3] = 0x5CBE8AB22BDF0611;
    expected_2[0].y.data[0] = 0x77D41ED256685368;
    expected_2[0].y.data[1] = 0x3AC1367E35FA7D3B;
    expected_2[0].y.data[2] = 0x35173845C2840DD4;
    expected_2[0].y.data[3] = 0x1353139CDE2A848F;
    expected_2[0].z.data[0] = 0xFBA158CF61C2E612;
    expected_2[0].z.data[1] = 0x8F32B98BC4097D0F;
    expected_2[0].z.data[2] = 0x82396ABF83AC1599;
    expected_2[0].z.data[3] = 0x21498DDAAB185B8C;

    comparator_kernel<<<1, 4>>>(final_result, expected_1, result_1);
    comparator_kernel<<<1, 4>>>(final_result, expected_2, result_2);
    cudaDeviceSynchronize();
    print_field_tests(result_1);
    print_field_tests(result_2);
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
    // execute_kernels_finite_fields_vector(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);
    // execute_kernels_curve_vector(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);
    // execute_kernels_finite_fields_vector_with_scalars(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);
    // execute_kernels_curve_vector_with_scalars(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);
    execute_double_and_add_single(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);
    execute_double_and_add_multiple(a, b, c, d, result, res_x, res_y, res_z, expect_x, expect_y, expect_z);

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