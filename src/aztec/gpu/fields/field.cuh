#include <cstdint>
#include <stdio.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <chrono>
#include <iostream>
#include <fixnum.cu>

using namespace std;

namespace gpu_barretenberg {
typedef std::uint64_t var;
static constexpr size_t LIMBS = 4;
static constexpr size_t BYTES_PER_ELEM = LIMBS * sizeof(var);

/* -------------------------- Base Field Modulus Fq ---------------------------------------------- */

/**
 * BN-254 defined by Y^2 = X^3 + 3 over the base field F_q, 
 * q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
 */
__device__ __constant__ 
var MOD_Q_BASE[LIMBS] = {
    0x3C208C16D87CFD47UL, 0x97816a916871ca8dUL,
    0xb85045b68181585dUL, 0x30644e72e131a029UL
};

/**
 * R^2 where R = 2^256 mod Q
 */
__device__ __constant__ 
var R_SQUARED_BASE[LIMBS] = {
    0xF32CFC5B538AFA89UL, 0xB5E71911D44501FBUL,
    0x47AB1EFF0A417FF6UL, 0x06D89F71CAB8351FUL
};

__device__ __constant__ 
var CUBE_ROOT_BASE[LIMBS] = {
    0x71930c11d782e155UL, 0xa6bb947cffbe3323UL,
    0xaa303344d4741444UL, 0x2c3b3f0d26594943UL
};

__device__ __constant__ 
var PRIMTIVE_ROOTS_UNITY_BASE[LIMBS] = {
    0UL, 0UL,
    0UL, 0UL
};

__device__ __constant__  
var COSET_GENERATORS_BASE_0[8]{
    0x7a17caa950ad28d7ULL, 0x4d750e37163c3674ULL, 0x20d251c4dbcb4411ULL, 0xf42f9552a15a51aeULL,
    0x4f4bc0b2b5ef64bdULL, 0x22a904407b7e725aULL, 0xf60647ce410d7ff7ULL, 0xc9638b5c069c8d94ULL,
};

__device__ __constant__  
var COSET_GENERATORS_BASE_1[8]{
    0x1f6ac17ae15521b9ULL, 0x29e3aca3d71c2cf7ULL, 0x345c97cccce33835ULL, 0x3ed582f5c2aa4372ULL,
    0x1a4b98fbe78db996ULL, 0x24c48424dd54c4d4ULL, 0x2f3d6f4dd31bd011ULL, 0x39b65a76c8e2db4fULL,
};

__device__ __constant__ 
var COSET_GENERATORS_BASE_2[8]{
    0x334bea4e696bd284ULL, 0x99ba8dbde1e518b0ULL, 0x29312d5a5e5edcULL,   0x6697d49cd2d7a508ULL,
    0x5c65ec9f484e3a79ULL, 0xc2d4900ec0c780a5ULL, 0x2943337e3940c6d1ULL, 0x8fb1d6edb1ba0cfdULL,
};

__device__ __constant__  
var COSET_GENERATORS_BASE_3[8]{
    0x2a1f6744ce179d8eULL, 0x3829df06681f7cbdULL, 0x463456c802275bedULL, 0x543ece899c2f3b1cULL,
    0x180a96573d3d9f8ULL,  0xf8b21270ddbb927ULL,  0x1d9598e8a7e39857ULL, 0x2ba010aa41eb7786ULL,
};

__device__ __constant__ var ONE_MONT[LIMBS] = {
    0xd35d438dc58f0d9d, 0xa78eb28f5c70b3d, 0x666ea36f7879462c, 0xe0a77c19a07df2f
};

/**
 * -Q^{-1} (mod 2^256)
 */ 
__device__ __constant__ var r_inv_base = 0x87d20782e4866389UL;

__device__ __constant__ var endo_g1_lo_base = 0x7a7bd9d4391eb18d;
__device__ __constant__ var endo_g1_mid_base = 0x4ccef014a773d2cfUL;
__device__ __constant__ var endo_g1_hi_base = 0x0000000000000002UL;
__device__ __constant__ var endo_g2_lo_base = 0xd91d232ec7e0b3d2UL;
__device__ __constant__ var endo_g2_mid_base = 0x0000000000000002UL;
__device__ __constant__ var endo_minus_b1_lo_base = 0x8211bbeb7d4f1129UL;
__device__ __constant__ var endo_minus_b1_mid_base = 0x6f4d8248eeb859fcUL;
__device__ __constant__ var endo_b2_lo_base = 0x89d3256894d213e2UL;
__device__ __constant__ var endo_b2_mid_base = 0UL;
__device__ __constant__ var b = 3;

/* -------------------------- Scalar Field Modulus Fr ---------------------------------------------- */

/**
 * Scalar field F_r has curve order r,
 * r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
 */
__device__ __constant__
var MOD_Q_SCALAR[LIMBS] = {
    0x43E1F593F0000001UL, 0x2833E84879B97091UL,
    0xB85045B68181585DUL, 0x30644E72E131A029UL
};

/**
 * R^2 where R = 2^256 mod Q
 */
__device__ __constant__
var R_SQUARED_SCALAR[LIMBS] = {
    0x1BB8E645AE216DA7UL, 0x53FE3AB1E35C59E3UL,
    0x8C49833D53BB8085UL, 0x216D0B17F4E44A5UL
};

__device__ __constant__ 
var CUBE_ROOT_SCALAR[LIMBS] = {
    0x93e7cede4a0329b3UL, 0x7d4fdca77a96c167UL,
    0x8be4ba08b19a750aUL, 0x1cbd5653a5661c25UL
};

/**
 * Primitive roots of unity
 */
__device__ __constant__ 
var PRIMTIVE_ROOTS_UNITY_SCALAR[LIMBS] = {
    0x636e735580d13d9cUL, 0xa22bf3742445ffd6UL,
    0x56452ac01eb203d8UL, 0x1860ef942963f9e7UL
};

__device__ __constant__
var COST_GENERATORS_SCALAR_O[8]{
    0x5eef048d8fffffe7ULL, 0xb8538a9dfffffe2ULL,  0x3057819e4fffffdbULL, 0xdcedb5ba9fffffd6ULL,
    0x8983e9d6efffffd1ULL, 0x361a1df33fffffccULL, 0xe2b0520f8fffffc7ULL, 0x8f46862bdfffffc2ULL,
};

__device__ __constant__
var COST_GENERATORS_SCALAR_1[8]{
    0x12ee50ec1ce401d0ULL, 0x49eac781bc44cefaULL, 0x307f6d866832bb01ULL, 0x677be41c0793882aULL,
    0x9e785ab1a6f45554ULL, 0xd574d1474655227eULL, 0xc7147dce5b5efa7ULL,  0x436dbe728516bcd1ULL,
};

__device__ __constant__ 
var COST_GENERATORS_SCALAR_2[8]{
    0x29312d5a5e5ee7ULL,   0x6697d49cd2d7a515ULL, 0x5c65ec9f484e3a89ULL, 0xc2d4900ec0c780b7ULL,
    0x2943337e3940c6e5ULL, 0x8fb1d6edb1ba0d13ULL, 0xf6207a5d2a335342ULL, 0x5c8f1dcca2ac9970ULL,
};

__device__ __constant__ 
var COST_GENERATORS_SCALAR_3[8]{
    0x463456c802275bedULL, 0x543ece899c2f3b1cULL, 0x180a96573d3d9f8ULL,  0xf8b21270ddbb927ULL,
    0x1d9598e8a7e39857ULL, 0x2ba010aa41eb7786ULL, 0x39aa886bdbf356b5ULL, 0x47b5002d75fb35e5ULL,
};

__device__ __constant__ var endo_g1_lo_scalar = 0x7a7bd9d4391eb18dUL;
__device__ __constant__ var endo_g1_mid_scalar = 0x4ccef014a773d2cfUL;
__device__ __constant__ var endo_g1_hi_scalar = 0x0000000000000002UL;
__device__ __constant__ var endo_g2_lo_scalar = 0xd91d232ec7e0b3d7UL;
__device__ __constant__ var endo_g2_mid_scalar = 0x0000000000000002UL;
__device__ __constant__ var endo_minus_b1_lo_scalar = 0x8211bbeb7d4f1128UL;
__device__ __constant__ var endo_minus_b1_mid_scalar = 0x6f4d8248eeb859fcUL;
__device__ __constant__ var endo_b2_lo_scalar = 0x89d3256894d213e3UL;
__device__ __constant__ var endo_b2_mid_scalar = 0UL;

/**
 * -Q^{-1} (mod 2^256)
 */
__device__ __constant__ var r_inv_scalar = 0xc2e1f593efffffffUL;

struct BN254_MOD_BASE {
    __device__ __forceinline__ static int lane() { return fixnum::layout().thread_rank(); }
    __device__ __forceinline__ static var mod() { return MOD_Q_BASE[lane()]; }
    __device__ __forceinline__ static var monty() { return R_SQUARED_BASE[lane()]; }
    __device__ __forceinline__ static var cube() { return CUBE_ROOT_BASE[lane()]; }
    __device__ __forceinline__ static var root() { return PRIMTIVE_ROOTS_UNITY_BASE[lane()]; }
    __device__ __forceinline__ static var one_mont() { return ONE_MONT[lane()]; }
};

struct BN254_MOD_SCALAR {
    __device__ __forceinline__ static int lane() { return fixnum::layout().thread_rank(); }
    __device__ __forceinline__ static var mod() { return MOD_Q_SCALAR[lane()]; }
    __device__ __forceinline__ static var monty() { return R_SQUARED_SCALAR[lane()]; }
    __device__ __forceinline__ static var cube() { return CUBE_ROOT_SCALAR[lane()]; }
    __device__ __forceinline__ static var root() { return PRIMTIVE_ROOTS_UNITY_SCALAR[lane()]; }
};

/* -------------------------- Finite Field Arithmetic for G1 ---------------------------------------------- */

template < typename params > 
class field_gpu {
    public:    
        var data[4];    
    
        __device__ __forceinline__ field_gpu() noexcept {}
        
        __device__ __forceinline__ field_gpu(const var a, const var b, const var c, const var d) noexcept;

        __device__ __forceinline__ static field_gpu zero();
        
        __device__ __forceinline__ static field_gpu one();
        
        __device__ __forceinline__ static bool is_zero(const var &x);

        __device__ __forceinline__ static var equal(const var x, const var y);

        __device__ __forceinline__ static var load(var x, var &res);

        __device__ __forceinline__ static void store(var *mem, const var &x);

        __device__ __forceinline__ static var add(const var a, const var b, var &res);

        __device__ __forceinline__ static var sub(const var x, const var y, var &z);

        __device__ __forceinline__ static var square(var x, var &y);   

        __device__ __forceinline__ static var mul(const var a, const var b, var &res);

        __device__ __forceinline__ static var to_monty(var x, var &res);
        
        __device__ __forceinline__ static var from_monty(var x, var &res);

        __device__ __forceinline__ static var neg(var &x, var &res);
};
typedef field_gpu<BN254_MOD_BASE> fq_gpu;
typedef field_gpu<BN254_MOD_SCALAR> fr_gpu;

}