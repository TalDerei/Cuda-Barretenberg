#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>

/*
 * Perform operations on fixed-precision integers using
 * Nvidia's inline PTX Assembly Language
 */
namespace internal {
    typedef std::uint32_t u32;
    typedef std::uint64_t u64;

    // Add two 32-bit signed integers and set carry flag on overflow ('s' > INT_MAX)
    __device__ __forceinline__
    void addc(int &s, int a, int b) {
        asm ("addc.s32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }

    // Add two 32-bit unsigned integers and set carry flag on overflow ('s' > INT_MAX)
    __device__ __forceinline__
    void addc(u32 &s, u32 a, u32 b) {
        asm ("addc.u32 %0, %1, %2;"
             : "=r"(s)
             : "r"(a), "r" (b));
    }

    // Add two 64-bit unsigned integers and set carry flag on overflow ('s' > INT_MAX)
    __device__ __forceinline__
    void addc(u64 &s, u64 a, u64 b) {
        asm ("addc.u64 %0, %1, %2;"
             : "=l"(s)
             : "l"(a), "l" (b));
    }

    /*
     * hi * 2^n + lo = a * b
     */
    __device__ __forceinline__
    void mul_wide(u32 &hi, u32 &lo, u32 a, u32 b) {
        asm ("{\n\t"
             " .reg .u64 tmp;\n\t"
             " mul.wide.u32 tmp, %2, %3;\n\t"
             " mov.b64 { %1, %0 }, tmp;\n\t"
             "}"
             : "=r"(hi), "=r"(lo)
             : "r"(a), "r"(b));
    }

    __device__ __forceinline__
    void mul_wide(u64 &hi, u64 &lo, u64 a, u64 b) {
        asm ("mul.hi.u64 %0, %2, %3;\n\t"
             "mul.lo.u64 %1, %2, %3;"
             : "=l"(hi), "=l"(lo)
             : "l"(a), "l"(b));
    }

    // lo = a * b + c (mod 2^n)
    __device__ __forceinline__
    void mad_lo(u32 &lo, u32 a, u32 b, u32 c) {
        asm ("mad.lo.u32 %0, %1, %2, %3;"
             : "=r"(lo)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void mad_lo(u64 &lo, u64 a, u64 b, u64 c) {
        asm ("mad.lo.u64 %0, %1, %2, %3;"
             : "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }


    // as above but with carry in cy
    __device__ __forceinline__
    void mad_lo_cc(u32 &lo, u32 a, u32 b, u32 c) {
        asm ("mad.lo.cc.u32 %0, %1, %2, %3;"
             : "=r"(lo)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void mad_lo_cc(u64 &lo, u64 a, u64 b, u64 c) {
        asm ("mad.lo.cc.u64 %0, %1, %2, %3;"
             : "=l"(lo)
             : "l"(a), "l" (b), "l"(c));
    }

    __device__ __forceinline__
    void mad_hi(u32 &hi, u32 a, u32 b, u32 c) {
        asm ("mad.hi.u32 %0, %1, %2, %3;"
             : "=r"(hi)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void mad_hi(u64 &hi, u64 a, u64 b, u64 c) {
        asm ("mad.hi.u64 %0, %1, %2, %3;"
             : "=l"(hi)
             : "l"(a), "l" (b), "l"(c));
    }

    __device__ __forceinline__
    void mad_hi_cc(u32 &hi, u32 a, u32 b, u32 c) {
        asm ("mad.hi.cc.u32 %0, %1, %2, %3;"
             : "=r"(hi)
             : "r"(a), "r" (b), "r"(c));
    }

    __device__ __forceinline__
    void mad_hi_cc(u64 &hi, u64 a, u64 b, u64 c) {
        asm ("mad.hi.cc.u64 %0, %1, %2, %3;"
             : "=l"(hi)
             : "l"(a), "l" (b), "l"(c));
    }
    
} 