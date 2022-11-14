#include <cstdint>
#include <stdio.h>

#include "field.cuh"
#include <fixnum.cu>

using namespace std;
using namespace gpu_barretenberg;

template<class params> 
__device__  field_gpu<params>::field_gpu(var a, var b, var c, var d) noexcept
    : data{ a, b, c, d } {};

template<class params> 
__device__ field_gpu<params> field_gpu<params>::zero() noexcept {
    return field_gpu(0, 0, 0, 0); 
}

template<class params> 
__device__ bool field_gpu<params>::is_zero() const noexcept {
    return ((data[0] | data[1] | data[2] | data[3]) == 0);
}

template<class params> 
__device__ void field_gpu<params>::add(const var *a, const var *b, var *res) {
    int br;
    var x = *a, y = *b, z, r;
    var mod = BN254_MOD::mod();
    fixnum::add(z, x, y);
    if (z < mod) {
        *res = z;
    }
    else {
        fixnum::sub_br(r, br, z, mod);
        *res = r;
    }
}

template<class params> 
__device__ void field_gpu<params>::sub(const var *x, const var *y, var *z) {
    int br;
    var r, mod = BN254_MOD::mod();
    fixnum::sub_br(r, br, *x, *y);
    if (br)
        fixnum::add(r, r, mod);
    *z = r;
}

template<class params> 
__device__ void field_gpu<params>::mul(const var a, const var b, var &res) {
    auto grp = fixnum::layout();
    int L = grp.thread_rank();
    var mod = BN254_MOD::mod();

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
}