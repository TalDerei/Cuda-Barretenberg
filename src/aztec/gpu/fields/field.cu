#include <cstdint>
#include <stdio.h>

#include "field.cuh"
#include <fixnum.cu>

using namespace std;
using namespace gpu_barretenberg;

// Templated with base and scalar fields
template<class params, class _params> 
__device__ field_gpu<params, _params>::field_gpu(var a, var b, var c, var d) noexcept
    : data{ a, b, c, d } {};
    
template<class params, class _params> 
__device__ field_gpu<params, _params> field_gpu<params, _params>::zero() noexcept {
    return field_gpu(0, 0, 0, 0); 
}

// template<class params, class _params> 
// __device__ field_gpu<params, _params> field_gpu<params, _params>::one() noexcept {
//     return field_gpu{ 0x01, 0x00, 0x00, 0x00 }.to_montgomery_form();
// }

template<class params, class _params> 
__device__ bool field_gpu<params, _params>::is_zero() const noexcept {
    return ((data[0] | data[1] | data[2] | data[3]) == 0);
}

template<class params, class _params> 
__device__ int field_gpu<params, _params>::equal(const var x, const var y) { 
    return fixnum::cmp(x, y) == 0; 
}

// Load operation copies data from main memory into a register
template<class params, class _params> 
__device__ void field_gpu<params, _params>::load(field_gpu &x, const var *mem) {
    int id = params::lane();
    x.data = (id < LIMBS) ? mem[id] : 0UL;
}

// Store operation copies data from a register into main memory
template<class params, class _params> 
__device__ void field_gpu<params, _params>::store(var *mem, const field_gpu &x) {
    int id = params::lane();
    if (id < LIMBS) {
        mem[id] = x.data;
    }
}

// Addition operation
template<class params, class _params> 
__device__ var field_gpu<params, _params>::add(const var a, const var b, var &res) {
    int br;
    var x = a, y = b, z, r;
    var mod = params::mod();
    fixnum::add(z, x, y);
    fixnum::sub_br(r, br, z, mod);
    res = br ? z : r;
    return res;
}

// Subtraction operation
template<class params, class _params> 
__device__ var field_gpu<params, _params>::sub(const var x, const var y, var &res) {
    int br;
    var r, mod = params::mod();
    fixnum::sub_br(r, br, x, y);
    if (br)
        fixnum::add(r, r, mod);
    res = r;
    return r;
}

// Square operation -- worth special casing for 1.5 - 2x speed improvement
template<class params, class _params> 
__device__ var field_gpu<params, _params>::square(var x, var &y) {
    field_gpu::mul(x, x, y);
    return y;
}

template<class params, class _params> 
__device__ var field_gpu<params, _params>::to_monty(var x, var &res) {
    var r_sqr_mod = params::monty();
    field_gpu::mul(x, r_sqr_mod, res);
    return res;
}

template<class params, class _params> 
__device__ var field_gpu<params, _params>::from_monty(var x, var &res) {
    var mont;
    mont = fixnum::one();
    mul(x, mont, res);
    return res;
}

// Mongomery multiplication (CIOS) operation
template<class params, class _params> 
__device__ var field_gpu<params, _params>::mul(const var a, const var b, var &res) {
    auto grp = fixnum::layout();
    int L = grp.thread_rank();
    var mod = params::mod();

    var x = a, y = b, z = digit::zero();
    var tmp;
    digit::mul_lo(tmp, x, gpu_barretenberg::r_inv_base);
    digit::mul_lo(tmp, tmp, grp.shfl(y, 0));
    int cy = 0;

    for (int i = 0; i < LIMBS; ++i) {
        var u;
        var xi = grp.shfl(x, i);
        var z0 = grp.shfl(z, 0);
        var tmpi = grp.shfl(tmp, i);

        digit::mad_lo(u, z0, gpu_barretenberg::r_inv_base, tmpi);
        digit::mad_lo_cy(z, cy, mod, u, z);
        digit::mad_lo_cy(z, cy, y, xi, z);

        assert(L || z == 0);  // z[0] must be 0
        z = grp.shfl_down(z, 1); // Shift right one word
        z = (L >= LIMBS - 1) ? 0 : z;

        digit::add_cy(z, cy, z, cy);
        digit::mad_hi_cy(z, cy, mod, u, z);
        digit::mad_hi_cy(z, cy, y, xi, z);
    }
    // Resolve carries
    int msb = grp.shfl(cy, LIMBS - 1);
    cy = grp.shfl_up(cy, 1); // left shift by 1
    cy = (L == 0) ? 0 : cy;

    fixnum::add_cy(z, cy, z, cy);
    msb += cy;
    assert(msb == !!msb); // msb = 0 or 1.

    // br = 0 ==> z >= mod
    var r;
    int br;
    fixnum::sub_br(r, br, z, mod);
    if (msb || br == 0) {
        // If the msb was set, then we must have had to borrow.
        assert(!msb || msb == br);
        z = r;
    }
    res = z;
    return res;
}