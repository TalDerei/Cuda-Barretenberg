#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <chrono>
#include <iostream>

#include "field.cu"
#include "element.cuh"

using namespace std;

namespace gpu_barretenberg {
/* -------------------------- BN254 G1 Parameters ---------------------------------------------- */
// __device__ var one_x = fq_gpu::one();
__device__ __constant__ var one_y[LIMBS] = {
    0xa6ba871b8b1e1b3aUL, 0x14f1d651eb8e167bUL, 
    0xccdd46def0f28c58UL, 0x1c14ef83340fbe5eUL 
};

__device__ __constant__ var a[LIMBS] = { 
    0UL, 0UL, 
    0UL, 0UL
};

__device__ __constant__ var b[LIMBS] { 
    0x7a17caa950ad28d7UL, 0x1f6ac17ae15521b9UL, 
    0x334bea4e696bd284UL, 0x2a1f6744ce179d8eUL 
};

__device__ __constant__ bool USE_ENDOMORPHISM = true;
__device__ __constant__ bool can_hash_to_curve = true;
__device__ __constant__ bool small_elements = true;
__device__ __constant__ bool has_a = false;

/* -------------------------- G1 Elliptic Curve Operations ---------------------------------------------- */
// Group class. Represents an elliptic curve group element
template < typename fq_gpu > 
class group_gpu {
    public:    
        typedef group_elements::element<fq_gpu> element;
        typedef group_elements::affine_element<fq_gpu> affine_element;

        __device__ static void load_affine(affine_element &X, const var *y);

        __device__ static void load_jacobian(element &X, const var *y);

        __device__ static void is_affine(const affine_element &X);

        __device__ static void is_affine_equal(const affine_element &X);

        __device__ static void is_jacobian_equal(const affine_element &X);

        __device__ static void store_affine(const affine_element &X, const var *y);

        __device__ static void store_jacobian(const element &X, const var *y);

        __device__ static void set_zero(const element &X);

        __device__ static void is_zero(const element &X);

        __device__ static void mixed_add(var X, var Y, var Z, var A, var B, var &res_x, var &res_y, var &res_z) noexcept;
        
        // add, mul, dbl, neg
    };
    typedef group_gpu<fq_gpu> g1;
}