#include "montgomery.h"

using namespace std;
using namespace std::chrono;

using namespace barretenberg_test;

template <class T> 
field<T> field<T>::operator*(const field& other) const noexcept {
    return montgomery_mul(other);
}

template <class T>
std::pair<uint64_t, uint64_t> field<T>::mul_wide(const uint64_t a, const uint64_t b) noexcept {
    // mul_wide multiplies in two uint64_t variables, and explicitly casts them to uint128_t to avoid overflows
    const uint128_t res = ((uint128_t)a * (uint128_t)b);
    
    // Need to explicitely cast result back to uint64_t and right bit-shfit by 8 bytes
    return { (uint64_t)(res), (uint64_t)(res >> 64) };
}

template <class T>
uint64_t field<T>::mac_discard_lo(const uint64_t a, const uint64_t b, const uint64_t c) noexcept {
    const uint128_t res = (uint128_t)a + ((uint128_t)b * (uint128_t)c);
    return (uint64_t)(res >> 64);
}

template <class T>
uint64_t field<T>::mac_mini(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t& carry_out) noexcept {
    const uint128_t res = (uint128_t)a + ((uint128_t)b * (uint128_t)c);
    carry_out = (uint64_t)(res >> 64);
    return (uint64_t)(res);
}

template <class T>
void field<T>::mac_mini(const uint64_t a, const uint64_t b, const uint64_t c, uint64_t& out, uint64_t& carry_out) noexcept {
    const uint128_t res = (uint128_t)a + ((uint128_t)b * (uint128_t)c);
    out = (uint64_t)(res);
    carry_out = (uint64_t)(res >> 64);
}

template <class T>
void field<T>::mac(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t carry_in, uint64_t& out, uint64_t& carry_out) noexcept {
    const uint128_t res = (uint128_t)a + ((uint128_t)b * (uint128_t)c) + (uint128_t)carry_in;
    out = (uint64_t)(res);
    carry_out = (uint64_t)(res >> 64);
}

template<class T> 
uint64_t field<T>::addc(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t &carry_out) noexcept {
    uint128_t res = (uint128_t)a + (uint128_t)b + (uint128_t)carry_in;
    carry_out = (uint64_t)(res >> 64);
    return (uint64_t)res;
}

template<class T> 
field<T> field<T>::to_montgomery_form() noexcept {
    field r_squared{ T::r_squared_0, T::r_squared_1, T::r_squared_2, T::r_squared_3 };

    field result = *this;
    result.reduce_once();
    result.reduce_once();
    result.reduce_once();
    return (result * r_squared).reduce_once();
}

template<class T> 
field<T> field<T>::self_to_montgomery_form() noexcept {
    field r_squared{ T::r_squared_0, T::r_squared_1, T::r_squared_2, T::r_squared_3 };

    self_reduce_once();
    self_reduce_once();
    self_reduce_once();
    *this *= r_squared;
    self_reduce_once();
}

template<class T> 
field<T> field<T>::from_montgomery_form() noexcept {
    field one_raw{ 1, 0, 0, 0 };
    return operator*(one_raw).reduce_once();
}

template<class T> 
void field<T>::self_from_montgomery_form() noexcept {
    constexpr field one_raw{ 1, 0, 0, 0 };
    *this *= one_raw;
    self_reduce_once();
}

template<class T> 
field<T> field<T>::reduce_once() noexcept {
    return reduce();
}

template<class T> 
void field<T>::self_reduce_once() noexcept {
    *this = reduce();
}

template<class T> 
field<T> field<T>::reduce() noexcept {
    if constexpr (modulus.data[3] >= 0x4000000000000000ULL) {
        uint256_t val{ data[0], data[1], data[2], data[3] };
        if (val >= modulus) {
            val -= modulus;
        }
        return { val.data[0], val.data[1], val.data[2], val.data[3] };
    }
    uint64_t t0 = data[0] + not_modulus.data[0];
    uint64_t c = t0 < data[0];
    auto t1 = addc(data[1], not_modulus.data[1], c, c);
    auto t2 = addc(data[2], not_modulus.data[2], c, c);
    auto t3 = addc(data[3], not_modulus.data[3], c, c);
    const uint64_t selection_mask = 0ULL - c; // 0xffff... if we have overflowed.
    const uint64_t selection_mask_inverse = ~selection_mask;
    // if we overflow, we want to swap
    return {
        (data[0] & selection_mask_inverse) | (t0 & selection_mask),
        (data[1] & selection_mask_inverse) | (t1 & selection_mask),
        (data[2] & selection_mask_inverse) | (t2 & selection_mask),
        (data[3] & selection_mask_inverse) | (t3 & selection_mask),
    };
}

// Montgomery multiplication (Alg. 1/2 CIOS METHOD): https://eprint.iacr.org/2012/140.pdf
template <class T> 
field<T> field<T>::montgomery_mul(const field& other) const noexcept {
    auto [t0, c] = mul_wide(data[0], other.data[0]);
    uint64_t k = t0 * T::r_inv;
    uint64_t a = mac_discard_lo(t0, k, modulus.data[0]);

    uint64_t t1 = mac_mini(a, data[0], other.data[1], a);
    mac(t1, k, modulus.data[1], c, t0, c);
    uint64_t t2 = mac_mini(a, data[0], other.data[2], a);
    mac(t2, k, modulus.data[2], c, t1, c);
    uint64_t t3 = mac_mini(a, data[0], other.data[3], a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;

    mac_mini(t0, data[1], other.data[0], t0, a);
    k = t0 * T::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, data[1], other.data[1], a, t1, a);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, data[1], other.data[2], a, t2, a);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, data[1], other.data[3], a, t3, a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;

    mac_mini(t0, data[2], other.data[0], t0, a);
    k = t0 * T::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, data[2], other.data[1], a, t1, a);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, data[2], other.data[2], a, t2, a);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, data[2], other.data[3], a, t3, a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;

    mac_mini(t0, data[3], other.data[0], t0, a);
    k = t0 * T::r_inv;
    c = mac_discard_lo(t0, k, modulus.data[0]);
    mac(t1, data[3], other.data[1], a, t1, a);
    mac(t1, k, modulus.data[1], c, t0, c);
    mac(t2, data[3], other.data[2], a, t2, a);
    mac(t2, k, modulus.data[2], c, t1, c);
    mac(t3, data[3], other.data[3], a, t3, a);
    mac(t3, k, modulus.data[3], c, t2, c);
    t3 = c + a;
    return { t0, t1, t2, t3 };
}

int main() {
    // Start timer
    auto start = high_resolution_clock::now();
    
    // Field elements
    fq a{ 0x2523b6fa3956f038, 0x158aa08ecdd9ec1d, 0xf48216a4c74738d4, 0x2514cc93d6f0a1bf };
    fq b{ 0xb68aee5e4c8fc17c, 0xc5193de7f401d5e8, 0xb8777d4dde671db3, 0xe513e75c087b0bb };
    fq const_expected{ 0x7ed4174114b521c4, 0x58f5bd1d4279fdc2, 0x6a73ac09ee843d41, 0x687a76ae9b3425c };
    fq const_result = a * b;

    printf("%" PRIu64 "\n", const_result.data[0]);
    printf("%" PRIu64 "\n", const_result.data[1]);
    printf("%" PRIu64 "\n", const_result.data[2]);
    printf("%" PRIu64 "\n", const_result.data[3]);

    // Benchmarking large executions 
    for (int i = 0; i < 10240000; i++) {
        const_result = a * b; 
    }

    // End timer
    auto stop = high_resolution_clock::now();

    // Calculate duraion of execution time 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds\n" << endl; 
  
    return 0;
}