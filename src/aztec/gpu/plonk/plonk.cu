#include <ecc/curves/bn254/fr.hpp>
#include <numeric/bitop/get_msb.hpp>
#include <plonk/composer/standard_composer.hpp>
#include <plonk/proof_system/prover/prover.hpp>
#include <plonk/proof_system/verifier/verifier.hpp>
#include <stdlib/primitives/field/field.hpp>
#include <iostream>

#include "composer_wrapper.cu"
#include "kzg_wrapper.cu"

using namespace std;
using namespace composer_gpu_wrapper;
using namespace waffle;

constexpr size_t MAX_GATES = 1 << 10;

Prover prover;
Verifier verifier;
plonk_proof proof;

void generate_test_plonk_circuit(StandardComposer& composer, size_t num_gates) {
    plonk::stdlib::field_t a(plonk::stdlib::witness_t(&composer, barretenberg::fr::random_element()));
    plonk::stdlib::field_t b(plonk::stdlib::witness_t(&composer, barretenberg::fr::random_element()));
    plonk::stdlib::field_t c(&composer);
    for (size_t i = 0; i < (num_gates / 4) - 4; ++i) {
        c = a + b;
        c = a * c;
        a = b * b;
        b = c * c;
    }
}

int main(int, char**) {
    cout << "Entered Plonk on GPU!\n" << endl;

    // Initialize composer wrapper object 
    composer_gpu_wrapper::composer *composer = new composer_gpu_wrapper::composer;

    // Generate test plonk circuit
    generate_test_plonk_circuit(composer->standard_composer, static_cast<size_t>(MAX_GATES));

    cout << "Constructed prover instance!" << endl; 
    prover = composer->create_prover();

    // cout << "Constructed verifier instance!" << endl; 
    // verifier = composer->create_verifier();

    // cout << "Generated proof!" << endl; 
    // proof = prover.construct_proof();

    // cout << "Verified proof!" << endl; 
    // verifier.verify_proof(proof);

    cout << "Successfully generated and verified proof for circuit of size: " << MAX_GATES << endl;
}