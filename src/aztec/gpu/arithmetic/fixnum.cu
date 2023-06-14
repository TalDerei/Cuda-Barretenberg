#pragma once

#include <cooperative_groups.h>
#include "primitives.cu"

/*
 * var is the basic register type that we deal with. The
 * interpretation of (one or more) such registers is determined by the
 * struct used, e.g. digit, fixnum, etc.
 */
typedef std::uint64_t var;

struct digit {
    // BYTES is the number of bytes in a var value
    static constexpr int BYTES = sizeof(var);
    // BITS is the number of bits in a var value
    static constexpr int BITS = BYTES * 8;

    // Add the values of two variables 'a' and 'b' and stores the result in 's'
    __device__ __forceinline__
    static void add(var &s, var a, var b) {
        s = a + b;
    }

    // Add the values of two variables 'a' and b and stores the result in 's' 
    // Store the carry bit in variable 'cy'
    __device__ __forceinline__
    static void add_cy(var &s, int &cy, var a, var b) {
        s = a + b;
        cy = s < a;
    }

    // Subtract the value of one variable 'b' from 'a' and stores the result in 'd'
    __device__ __forceinline__
    static void sub(var &d, var a, var b) {
        d = a - b;
    }

    // Subtract the value of variable 'b' from 'a' and stores the result in 'd'
    // Store the borrow bit in variable 'br'
    __device__ __forceinline__
    static void sub_br(var &d, int &br, var a, var b) {
        d = a - b;
        br = d > a;
    }

    // Return zero value of the var type
    __device__ __forceinline__
    static var zero() { return 0ULL; }

    // Return true if variable 'a' is equal to the maximum value of the var type, and false otherwise
    __device__ __forceinline__
    static int is_max(var a) { return a == ~0ULL; }

    // Return true if variable 'a' is equal to the minimum value of the var type, and false otherwise
    __device__ __forceinline__
    static int is_min(var a) { return a == 0ULL; }

    // Return true if variable 'a' is equal to zero, and false otherwise
    __device__ __forceinline__
    static int is_zero(var a) { return a == zero(); }

    // Mltiply two variables 'a' and 'b' and stores the lower 64 bits of the result in 'lo'
    __device__ __forceinline__
    static void mul_lo(var &lo, var a, var b) {
        lo = a * b;
    }

    // Compute the result of the operation a * b + c and stores the lower 64 bits in 'lo'
    // lo = a * b + c (mod 2^64)
    __device__ __forceinline__
    static void mad_lo(var &lo, var a, var b, var c) {
        internal::mad_lo(lo, a, b, c);
    }

    // Compute the result of the operation a * b + c and stores the lower 64 bits in 'lo'
    // Increment the value of 'cy' by the mad carry
    __device__ __forceinline__
    static void mad_lo_cy(var &lo, int &cy, var a, var b, var c) {
        internal::mad_lo_cc(lo, a, b, c);
        internal::addc(cy, cy, 0);
    }

    // Compute the result of the operation a * b + c and stores the upper 64 bits in 'hi'
    __device__ __forceinline__
    static void mad_hi(var &hi, var a, var b, var c) {
        internal::mad_hi(hi, a, b, c);
    }

    // Compute the result of the operation a * b + c and stores the upper 64 bits in 'hi'
    // Increment the value of 'cy' by the mad carry
    __device__ __forceinline__
    static void mad_hi_cy(var &hi, int &cy, var a, var b, var c) {
        internal::mad_hi_cc(hi, a, b, c);
        internal::addc(cy, cy, 0);
    }
};

struct fixnum {
    static constexpr unsigned WIDTH = 4;

    // Return the layout of the current thread block as a thread_block_tile object with WIDTH threads
    __device__ __forceinline__
    static cooperative_groups::thread_block_tile<WIDTH>
    layout() {
        return cooperative_groups::tiled_partition<WIDTH>(cooperative_groups::this_thread_block());
    }

    // Return zero value of var type
    __device__ __forceinline__
    static var zero() { return digit::zero(); }

    __device__ __forceinline__
    static var one() {
        auto t = layout().thread_rank();
        return (var)(t == 0);
    }

    // Add the values of two variables 'a' and 'b' and stores the result in 'r'
    // Store the carry bit in the variable 'cy_hi'. If the result of the addition overflows, 
    // it is propagated to the 'cy_hi' variable
    __device__ __forceinline__
    static void add_cy(var &r, int &cy_hi, const var &a, const var &b) {
        int cy;
        digit::add_cy(r, cy, a, b);
        // r propagates carries iff r = FIXNUM_MAX
        var r_cy = effective_carries(cy_hi, digit::is_max(r), cy);
        digit::add(r, r, r_cy);
    }

    // Add the values of two variables 'a' and 'b' and stores the result in a third variable 'r' 
    // If the result of the addition overflows, it is propagated to the next higher digit
    __device__ __forceinline__
    static void add(var &r, const var &a, const var &b) {
        int cy_hi;
        add_cy(r, cy_hi, a, b);
    }

    // Ssubtract the value of one variable 'b' from 'a' and stores the result in 'r'
    // Store the borrow bit in the variable 'br_lo'. If the result of the subtraction underflows, 
    // it is propagated to the 'br_lo' variable
    __device__ __forceinline__
    static void sub_br(var &r, int &br_lo, const var &a, const var &b) {
        int br;
        digit::sub_br(r, br, a, b);
        // r propagates borrows iff r = FIXNUM_MIN
        var r_br = effective_carries(br_lo, digit::is_min(r), br);
        digit::sub(r, r, r_br);
    }


    // Subtract the value of one variable 'b' from 'a' and stores the result in 'r'. If the result
    // of the subtraction underflows, it is propagated to the next higher digit
    __device__ __forceinline__
    static void sub(var &r, const var &a, const var &b) {
        int br_lo;
        sub_br(r, br_lo, a, b);
    }

    __device__ __forceinline__ 
    static uint32_t nonzero_mask(var r) {
        return fixnum::layout().ballot( ! digit::is_zero(r));
    }

    __device__ __forceinline__
    static int is_zero(var r) {
        return nonzero_mask(r) == 0U;
    }

    __device__ __forceinline__
    static int most_sig_dig(var x) {
        enum { UINT32_BITS = 8 * sizeof(uint32_t) };

        uint32_t a = nonzero_mask(x);
        return UINT32_BITS - (internal::clz(a) + 1);
    }

    // Compare equality of two var arrays
    __device__ __forceinline__
    static int cmp(var x, var y) {
        var r;
        int br;
        sub_br(r, br, x, y);
        // r != 0 iff x != y. If x != y, then br != 0 => x < y.
        return nonzero_mask(r) ? (br ? -1 : 1) : 0;
    }

    // Helper function 
    __device__ __forceinline__
    static var effective_carries(int &cy_hi, int propagate, int cy) {
        uint32_t allcarries, p, g;
        auto grp = fixnum::layout();

        g = grp.ballot(cy);                       // carry generate
        p = grp.ballot(propagate);                // carry propagate
        allcarries = (p | g) + g;                 // propagate all carries
        cy_hi = (allcarries >> grp.size()) & 1;   // detect hi overflow
        allcarries = (allcarries ^ p) | (g << 1); // get effective carries
        return (allcarries >> grp.thread_rank()) & 1;
    }
};