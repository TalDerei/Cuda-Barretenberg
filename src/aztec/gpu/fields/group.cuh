#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <chrono>
#include <iostream>

#include <fixnum.cu>
#include "field.cuh"

using namespace std;

namespace gpu_barretenberg {
/* -------------------------- BN254 G1 Parameters ---------------------------------------------- */
struct Bn254G1Params {
    __device__ __constant__ bool USE_ENDOMORPHISM = true;
    __device__ __constant__ bool can_hash_to_curve = true;
    __device__ __constant__ bool small_elements = true;
    __device__ __constant__ bool has_a = false;
    __device__ __constant__ fq_gpu one_x = fq_gpu::one();
    __device__ __constant__ fq_gpu one_y{ 0xa6ba871b8b1e1b3aUL, 0x14f1d651eb8e167bUL, 0xccdd46def0f28c58UL, 0x1c14ef83340fbe5eUL };
    __device__ __constant__ fq_gpu a{ 0UL, 0UL, 0UL, 0UL };
    __device__ __constant__ fq_gpu b{ 0x7a17caa950ad28d7UL, 0x1f6ac17ae15521b9UL, 0x334bea4e696bd284UL, 0x2a1f6744ce179d8eUL };
};

/* -------------------------- G1 Elliptic Curve Operations ---------------------------------------------- */
/* @brief group class. Represents an elliptic curve group element.
 * Group is parametrised by coordinate_field and subgroup_field
 *
 * Note: Currently subgroup checks are NOT IMPLEMENTED
 * Our current Plonk implementation uses G1 points that have a cofactor of 1.
 * All G2 points are precomputed (generator [1]_2 and trusted setup point [x]_2).
 * Explicitly assume precomputed points are valid members of the prime-order subgroup for G2.
 */
template < typename fq_gpu, typename GroupParams > 
class group_gpu {
    public:    

    };
    typedef group_gpu<fq_gpu, Bn254G1Params> g1;
}