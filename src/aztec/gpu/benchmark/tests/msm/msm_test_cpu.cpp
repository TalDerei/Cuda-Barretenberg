#include <chrono>
#include <common/assert.hpp>
#include <cstdlib>
#include <ecc/curves/bn254/scalar_multiplication/scalar_multiplication.hpp>
#include <plonk/reference_string/file_reference_string.hpp>
#include <polynomials/polynomial_arithmetic.hpp>
#include <iostream>
#include <fstream>
using namespace barretenberg;
using namespace std;

constexpr size_t NUM_POINTS = 2;
std::vector<fr> scalars;
auto reference_string = std::make_shared<waffle::FileReferenceString>(NUM_POINTS, "../srs_db");

void pippenger() {
    // Generate points and scalars
    g1::affine_element* points = reference_string->get_monomials();
    fr* scalars_new = (fr*)aligned_alloc(32, sizeof(fr) * NUM_POINTS);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        scalars_new[i] = fr::random_element();
    }

    // Execute Double-And-Add algorithm
    g1::element final_result;
    final_result.self_set_infinity();
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        g1::element temp = points[i] * scalars_new[i];
        final_result += temp;
    }
    
    // Execute Pippenger's Bucket Method
    scalar_multiplication::generate_pippenger_point_table(points, points, NUM_POINTS);
    scalar_multiplication::pippenger_runtime_state state(NUM_POINTS);
    g1::element result1 = scalar_multiplication::pippenger(scalars_new, points, NUM_POINTS, state);

    // Equality check
    assert(final_result == result1);
    cout << "TESTS PASSED!" << endl;
}

int main()
{
    pippenger();
    return 0;
}