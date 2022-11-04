#include <chrono>
#include <iostream>
#include <cuda-fixnum/src/fixnum/warp_fixnum.cu>
#include <cuda-fixnum/src/array/fixnum_array.h>
#include <cuda-fixnum/src/functions/modexp.cu>
#include <cuda-fixnum/src/functions/multi_modexp.cu>
#include <cuda-fixnum/src/modnum/modnum_monty_redc.cu>
#include <cuda-fixnum/src/modnum/modnum_monty_cios.cu>

#include "field.cuh"

using namespace std;
using namespace cuFIXNUM;

namespace gpu_barretenberg {
class BN254GPUFrParams {
    public:
        // Base field modulus
        __device__ __constant__
        const var MOD_Q[BIG_WIDTH] = {
            0x3C208C16D87CFD47UL, 0x97816a916871ca8dUL,
            0xb85045b68181585dUL, 0x30644e72e131a029UL
        };

        // -Q^{-1} (mod 2^256)
        static constexpr var Q_NINV_MOD = 0x87d20782e4866389UL;

        // 2^256 mod Q
        __device__ __constant__
        const var X_MOD_Q[BIG_WIDTH] = {
            0xF32CFC5B538AFA89UL, 0xB5E71911D44501FBUL,
            0x47AB1EFF0A417FF6UL, 0x06D89F71CAB8351FUL
        };
    };  

    // typedef fq by instantiating the field with Bn254FqParams
    typedef field_gpu<BN254GPUFrParams> fq_gpu;
}



