#ifndef MONT_MULT_DEF
#define MONT_MULT_DEF

#include <stdio.h>
#include <cstdint>
#include <chrono>
#include <inttypes.h>
#include <stdint.h>
#include <assert.h>
#include <numeric/random/engine.hpp>
#include <numeric/uint128/uint128.hpp>
#include <numeric/uint256/uint256.hpp>

namespace barretenberg_test {
template <class Params> 
struct field {
  public:
    field() noexcept {}

    field (const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d) noexcept : data{ a, b, c, d } {};

    static constexpr uint256_t modulus = uint256_t{ Params::modulus_0, Params::modulus_1, Params::modulus_2, Params::modulus_3 };

    uint64_t data[4];

    static constexpr uint256_t not_modulus = -modulus;

    // Montgomery Multiplication
    field montgomery_mul(const field& other) const noexcept;
    static std::pair<uint64_t, uint64_t> mul_wide(const uint64_t a, const uint64_t b) noexcept;
    static uint64_t mac(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t carry_in, uint64_t& carry_out) noexcept;
    static void mac(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t carry_in, uint64_t& out, uint64_t& carry_out) noexcept;
    static uint64_t mac_mini(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t& out) noexcept;
    static void mac_mini(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t& out, uint64_t& carry_out) noexcept;
    static uint64_t mac_discard_lo(const uint64_t a, const uint64_t b, const uint64_t c) noexcept;
    uint64_t addc(const uint64_t a, const uint64_t b, const uint64_t carry_in, uint64_t& carry_out) noexcept;

    // Operator overloading
    field operator*(const field& other) const noexcept;
    field& operator=(const field& other) = default;

    // To and from montogomery form
    field to_montgomery_form() noexcept;
    field self_to_montgomery_form() noexcept;
    field from_montgomery_form() noexcept;
    void self_from_montgomery_form() noexcept;
    field reduce_once() noexcept;
    void self_reduce_once() noexcept;
    field reduce() noexcept;
};

class Bn254FqParamsTest {
  public:
    // Base field modulus
    static constexpr uint64_t modulus_0 = 0x3C208C16D87CFD47UL;
    static constexpr uint64_t modulus_1 = 0x97816a916871ca8dUL;
    static constexpr uint64_t modulus_2 = 0xb85045b68181585dUL;
    static constexpr uint64_t modulus_3 = 0x30644e72e131a029UL;

    // R^2 where R = 2^256 mod p
    static constexpr uint64_t r_squared_0 = 0xF32CFC5B538AFA89UL;
    static constexpr uint64_t r_squared_1 = 0xB5E71911D44501FBUL;
    static constexpr uint64_t r_squared_2 = 0x47AB1EFF0A417FF6UL;
    static constexpr uint64_t r_squared_3 = 0x06D89F71CAB8351FUL;

    // -Q^{-1} (mod 2^64)
    static constexpr uint64_t r_inv = 0x87d20782e4866389UL;
};
typedef field<Bn254FqParamsTest> fq;
}

#endif