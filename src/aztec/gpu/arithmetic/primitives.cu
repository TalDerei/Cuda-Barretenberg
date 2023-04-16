#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>

// Perform operations on fixed-precision integers using 
// Instructions in Nvidia's inline PTX Assembly Language
namespace internal {
    typedef std::uint32_t u32;
    typedef std::uint64_t u64;

    // Add two 32-bit signed integers and set carry flag on overflow ('s' > INT_MAX).
    // Overflow involves exceeding the the maximum value for a 32-bit signed integer, 
    // and 's' is set to INT_MIN (minimum value for a 32-bit signed integer),
    // and carry flag is set
    __device__ __forceinline__
    void
    addc(int &s, int a, int b) {
        asm ("addc.s32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }

    // Add two 32-bit unsigned integers and set carry flag on overflow ('s' > INT_MAX)
    __device__ __forceinline__
    void
    addc(u32 &s, u32 a, u32 b) {
        asm ("addc.u32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }

    // Same as above, except it updates the carry flag in the processor's flag register
    // With carry out result of the addition. On overflow, the value of carry flag 'cc' 
    // Register is set to 1, but it does not modify the value of 's'. The result of the 
    // addition is still stored in s
    __device__ __forceinline__
    void
    add_cc(u32 &s, u32 a, u32 b) {
        asm ("add.cc.u32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }

    __device__ __forceinline__
    void
    addc_cc(u32 &s, u32 a, u32 b) {
        asm ("addc.cc.u32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }

    __device__ __forceinline__
    void
    addc(u64 &s, u64 a, u64 b) {
        asm ("addc.u64 %0, %1, %2;"
             : "=l"(s)
             : "l"(a), "l" (b));
    }

    __device__ __forceinline__
    void
    add_cc(u64 &s, u64 a, u64 b) {
        asm ("add.cc.u64 %0, %1, %2;"
             : "=l"(s)
             : "l"(a), "l" (b));
    }

    __device__ __forceinline__
    void
    addc_cc(u64 &s, u64 a, u64 b) {
        asm ("addc.cc.u64 %0, %1, %2;"
             : "=l"(s)
             : "l"(a), "l" (b));
    }

    /*
     * hi * 2^n + lo = a * b
     */
    __device__ __forceinline__
    void
    mul_hi(u32 &hi, u32 a, u32 b) {
        asm ("mul.hi.u32 %0, %1, %2;"
             : "=r"(hi)
             : "r"(a), "r"(b));
    }

    __device__ __forceinline__
    void
    mul_hi(u64 &hi, u64 a, u64 b) {
        asm ("mul.hi.u64 %0, %1, %2;"
             : "=l"(hi)
             : "l"(a), "l"(b));
    }

    /*
     * hi * 2^n + lo = a * b
     */
    __device__ __forceinline__
    void
    mul_wide(u32 &hi, u32 &lo, u32 a, u32 b) {
        asm ("{\n\t"
             " .reg .u64 tmp;\n\t"
             " mul.wide.u32 tmp, %2, %3;\n\t"
             " mov.b64 { %1, %0 }, tmp;\n\t"
             "}"
             : "=r"(hi), "=r"(lo)
             : "r"(a), "r"(b));
    }

    __device__ __forceinline__
    void
    mul_wide(u64 &hi, u64 &lo, u64 a, u64 b) {
        asm ("mul.hi.u64 %0, %2, %3;\n\t"
             "mul.lo.u64 %1, %2, %3;"
             : "=l"(hi), "=l"(lo)
             : "l"(a), "l"(b));
    }

    /*
     * (hi, lo) = a * b + c
     */
    __device__ __forceinline__
    void
    mad_wide(u32 &hi, u32 &lo, u32 a, u32 b, u32 c) {
        asm ("{\n\t"
             " .reg .u64 tmp;\n\t"
             " mad.wide.u32 tmp, %2, %3, %4;\n\t"
             " mov.b64 { %1, %0 }, tmp;\n\t"
             "}"
             : "=r"(hi), "=r"(lo)
             : "r"(a), "r"(b), "r"(c));
    }

    __device__ __forceinline__
    void
    mad_wide(u64 &hi, u64 &lo, u64 a, u64 b, u64 c) {
        asm ("mad.lo.cc.u64 %1, %2, %3, %4;\n\t"
             "madc.hi.u64 %0, %2, %3, 0;"
             : "=l"(hi), "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }

    // lo = a * b + c (mod 2^n)
    __device__ __forceinline__
    void
    mad_lo(u32 &lo, u32 a, u32 b, u32 c) {
        asm ("mad.lo.u32 %0, %1, %2, %3;"
             : "=r"(lo)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void
    mad_lo(u64 &lo, u64 a, u64 b, u64 c) {
        asm ("mad.lo.u64 %0, %1, %2, %3;"
             : "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }


    // as above but with carry in cy
    __device__ __forceinline__
    void
    mad_lo_cc(u32 &lo, u32 a, u32 b, u32 c) {
        asm ("mad.lo.cc.u32 %0, %1, %2, %3;"
             : "=r"(lo)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void
    mad_lo_cc(u64 &lo, u64 a, u64 b, u64 c) {
        asm ("mad.lo.cc.u64 %0, %1, %2, %3;"
             : "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }

    __device__ __forceinline__
    void
    madc_lo_cc(u32 &lo, u32 a, u32 b, u32 c) {
        asm ("madc.lo.cc.u32 %0, %1, %2, %3;"
             : "=r"(lo)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void
    madc_lo_cc(u64 &lo, u64 a, u64 b, u64 c) {
        asm ("madc.lo.cc.u64 %0, %1, %2, %3;"
             : "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }

    __device__ __forceinline__
    void
    mad_hi(u32 &hi, u32 a, u32 b, u32 c) {
        asm ("mad.hi.u32 %0, %1, %2, %3;"
             : "=r"(hi)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void
    mad_hi(u64 &hi, u64 a, u64 b, u64 c) {
        asm ("mad.hi.u64 %0, %1, %2, %3;"
             : "=l"(hi)
             : "l"(a), "l" (b), "l"(c));
    }

    __device__ __forceinline__
    void
    mad_hi_cc(u32 &hi, u32 a, u32 b, u32 c) {
        asm ("mad.hi.cc.u32 %0, %1, %2, %3;"
             : "=r"(hi)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void
    mad_hi_cc(u64 &hi, u64 a, u64 b, u64 c) {
        asm ("mad.hi.cc.u64 %0, %1, %2, %3;"
             : "=l"(hi)
             : "l"(a), "l" (b), "l"(c));
    }

    __device__ __forceinline__
    void
    madc_hi_cc(u32 &hi, u32 a, u32 b, u32 c) {
        asm ("madc.hi.cc.u32 %0, %1, %2, %3;"
             : "=r"(hi)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void
    madc_hi_cc(u64 &hi, u64 a, u64 b, u64 c) {
        asm ("madc.hi.cc.u64 %0, %1, %2, %3;\n\t"
             : "=l"(hi)
             : "l"(a), "l" (b), "l"(c));
    }

    __device__ __forceinline__
    int
    clz(u32 x) {
        int n;
        asm ("clz.b32 %0, %1;" : "=r"(n) : "r"(x));
        return n;
    }

    __device__ __forceinline__
    int
    clz(u64 x) {
        int n;
        asm ("clz.b64 %0, %1;" : "=r"(n) : "l"(x));
        return n;
    }

    /*
     * Count Trailing Zeroes in x.
     */
    __device__ __forceinline__
    int
    ctz(u32 x) {
        int n;
        asm ("{\n\t"
             " .reg .u32 tmp;\n\t"
             " brev.b32 tmp, %1;\n\t"
             " clz.b32 %0, tmp;\n\t"
             "}"
             : "=r"(n) : "r"(x));
        return n;
    }

    __device__ __forceinline__
    int
    ctz(u64 x) {
        int n;
        asm ("{\n\t"
             " .reg .u64 tmp;\n\t"
             " brev.b64 tmp, %1;\n\t"
             " clz.b64 %0, tmp;\n\t"
             "}"
             : "=r"(n) : "l"(x));
        return n;
    }

    __device__ __forceinline__
    void
    min(u32 &m, u32 a, u32 b) {
        asm ("min.u32 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(b));
    }

    __device__ __forceinline__
    void
    min(u64 &m, u64 a, u64 b) {
        asm ("min.u64 %0, %1, %2;" : "=l"(m) : "l"(a), "l"(b));
    }

    __device__ __forceinline__
    void
    max(u32 &m, u32 a, u32 b) {
        asm ("max.u32 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(b));
    }

    __device__ __forceinline__
    void
    max(u64 &m, u64 a, u64 b) {
        asm ("max.u64 %0, %1, %2;" : "=l"(m) : "l"(a), "l"(b));
    }
} // End namespace internal