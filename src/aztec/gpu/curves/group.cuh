#include "field.cu"
#include "element.cu"
#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <chrono>
#include <iostream>

using namespace std;

namespace gpu_barretenberg {

/* -------------------------- BN-254 G1 Elliptic Curve Parameters ---------------------------------------------- */

/**
 * Coefficients of the elliptic curve equation: y^2 = x^3 + 3
 */
__device__ __constant__ var a_bn_254 [LIMBS] = { 
    0UL, 0UL, 
    0UL, 0UL
};

__device__ __constant__ var b_bn_254 [LIMBS] { 
    0x7a17caa950ad28d7UL, 0x1f6ac17ae15521b9UL, 
    0x334bea4e696bd284UL, 0x2a1f6744ce179d8eUL 
};

/**
 * Generator point = (x, y) = (1, 2)
 */
__device__ __constant__ var one_x_bn_254[LIMBS] = {
    0xd35d438dc58f0d9dUL, 0xa78eb28f5c70b3dUL, 
    0x666ea36f7879462cUL, 0xe0a77c19a07df2fUL
};

__device__ __constant__ var one_y_bn_254 [LIMBS] = {
    0xa6ba871b8b1e1b3aUL, 0x14f1d651eb8e167bUL, 
    0xccdd46def0f28c58UL, 0x1c14ef83340fbe5eUL 
};

__device__ __constant__ bool USE_ENDOMORPHISM_BN_254 = true;
__device__ __constant__ bool can_hash_to_curve_bn_254 = true;
__device__ __constant__ bool small_elements_bn_254 = true;
__device__ __constant__ bool has_a_bn_254 = false;

/* -------------------------- Grumpkin G1 Elliptic Curve Parameters ---------------------------------------------- */

namespace grumpkin {
    typedef gpu_barretenberg::fr_gpu fq;
    typedef gpu_barretenberg::fq_gpu fr;

    __device__ __constant__ var a_grumpkin[LIMBS] = {
        0UL, 0UL, 
        0UL, 0UL
    };

    __device__ __constant__ var b_grumpkin[LIMBS] = {
        0xdd7056026000005a, 0x223fa97acb319311, 
        0xcc388229877910c0, 0x34394632b724eaa
    };

    /**
     * Generator point = (x, y) = (1, sqrt(-15))
     */
    __device__ __constant__ var one_x_grumpkin[LIMBS] = {
        0xac96341c4ffffffbUL, 0x36fc76959f60cd29UL, 
        0x666ea36f7879462eUL, 0xe0a77c19a07df2fUL
    };

    __device__ __constant__ var one_y_grumpkin[LIMBS] = {
        0x11b2dff1448c41d8UL, 0x23d3446f21c77dc3UL, 
        0xaa7b8cf435dfafbbUL, 0x14b34cf69dc25d68UL
    };

    __device__ __constant__ bool USE_ENDOMORPHISM_GRUMPKIN = true;
    __device__ __constant__ bool can_hash_to_curve_grumpkin = true;
    __device__ __constant__ bool small_elements_grumpkin = true;
    __device__ __constant__ bool has_a_grumpkin = false;
}

/* -------------------------- G1 Elliptic Curve Operations ---------------------------------------------- */

/**
 * Group class that represents an elliptic curve group element
 */
template < typename fq_gpu, typename fr_gpu > 
class group_gpu {
    public:    
        typedef group_elements::element<fq_gpu, fr_gpu> element;
        typedef group_elements::affine_element<fq_gpu, fr_gpu> affine_element;
        typedef group_elements::projective_element<fq_gpu, fr_gpu> projective_element;

        /* -------------------------- Affine and Jacobian Coordinate Operations ---------------------------------------------- */
        
        __device__ static void load_affine(affine_element &X, affine_element &result);

        __device__ static void load_jacobian(element &X, element &result);

        __device__ static void add(var X1, var Y1, var Z1, var X2, var Y2, var Z2, var &res_x, var &res_y, var &res_z);

        __device__ static void mixed_add(var X1, var Y1, var Z1, var X2, var Y2, var &res_x, var &res_y, var &res_z);

        __device__ static void doubling(var X, var Y, var Z, var &res_x, var &res_y, var &res_z);

        /* -------------------------- Projective Coordinate Operations ---------------------------------------------- */

        __device__ static void load_projective(projective_element &X, projective_element &result);
        
        projective_element from_affine(const affine_element &other);
        
        __device__ static void add_projective(var X1, var Y1, var Z1, var X2, var Y2, var Z2, var &res_x, var &res_y, var &res_z);
};
typedef group_gpu<fq_gpu, fr_gpu> g1;

}