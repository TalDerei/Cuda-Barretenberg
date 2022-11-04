#include <cstdint>
#include <iostream>
#include "fixnum.cuh"

// var is the basic register type that we deal with
typedef std::uint64_t var;

static constexpr size_t ELT_LIMBS = 4;
static constexpr size_t ELT_BYTES = ELT_LIMBS * sizeof(var);
static constexpr size_t BIG_WIDTH = ELT_LIMBS;

namespace gpu_barretenberg {
template < typename params > 
struct field_gpu {
    public:        
        // Constructors 
        __host__ __device__ 
        field_gpu () noexcept {};

        __host__ __device__ 
        field_gpu(const var a, const var b, const var c, const var d) noexcept : data{ a, b, c, d } {};

        var data[4];    

        __host__ __device__ 
        static constexpr field_gpu zero() { 
            return field_gpu(0, 0, 0, 0); 
        }

        struct BN254_MOD {
            __device__ static int lane() {
                return fixnum::layout().thread_rank();
            }

        };
    };
}