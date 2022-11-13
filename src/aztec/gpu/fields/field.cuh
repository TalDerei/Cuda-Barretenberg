#include <cstdint>
#include <stdio.h>
#include <fixnum.cu>

using namespace std;

namespace gpu_barretenberg {
typedef std::uint64_t var;
static constexpr size_t ELT_LIMBS = 4;
static constexpr size_t ELT_BYTES = ELT_LIMBS * sizeof(var);
static constexpr size_t BIG_WIDTH = ELT_LIMBS;

// Base field modulus
__device__
const var MOD_Q[BIG_WIDTH] = {
    0x3C208C16D87CFD47UL, 0x97816a916871ca8dUL,
    0xb85045b68181585dUL, 0x30644e72e131a029UL
};

// 2^256 mod Q
__device__ __constant__
const var X_MOD_Q[BIG_WIDTH] = {
    0xF32CFC5B538AFA89UL, 0xB5E71911D44501FBUL,
    0x47AB1EFF0A417FF6UL, 0x06D89F71CAB8351FUL
};

struct BN254_MOD {
    // -Q^{-1} (mod 2^256)
    static constexpr var Q_NINV_MOD = 0x87d20782e4866389UL;
    __device__ __forceinline__ static int lane() { return fixnum::layout().thread_rank(); }
    __device__ __forceinline__ static var mod() { return MOD_Q[lane()]; }
};

template < typename params > 
class field_gpu {
    public:        
        var data[4];    

        // Constructor 
        __device__
        field_gpu() noexcept {}
        
        __device__ 
        field_gpu(const var a, const var b, const var c, const var d) noexcept;

        __device__
        field_gpu zero() noexcept;

        __device__
        bool is_zero() const noexcept;

        __device__ 
        static void add(const var *a, const var *b, var *res);

        __device__ 
        static void sub(const var *x, const var *y, var *z);

        __device__ 
        static void mul(const var a, const var b, var &res);
    };
    typedef field_gpu<BN254_MOD> fq_gpu;
}