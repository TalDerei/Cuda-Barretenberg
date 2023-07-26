#include <ecc/curves/bn254/fr.hpp>
#include <numeric/bitop/get_msb.hpp>
#include <plonk/composer/standard_composer.hpp>
#include <plonk/proof_system/prover/prover.hpp>
#include <plonk/proof_system/verifier/verifier.hpp>
#include <stdlib/primitives/field/field.hpp>
#include <iostream>

#include "composer_wrapper.cuh"

using namespace waffle;

constexpr size_t MAX_GATES = 1 << 5;

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
    // Initialize composer and prover wrapper objects
    composer_gpu_wrapper::ComposerWrapper *composer = new composer_gpu_wrapper::ComposerWrapper;
    StandardComposer *composer_wrapper = &(*composer);
    Prover *prover = new prover_gpu_wrapper::ProverWrapper;
    
    // Generate test plonk circuit
    generate_test_plonk_circuit(composer->composer_wrapper, static_cast<size_t>(MAX_GATES));

    // Construct prover and verifier instances
    *prover = composer_wrapper->create_prover();
    Verifier verifier = composer_wrapper->create_verifier();

    // Generate and verify proof
    plonk_proof proof = prover->construct_proof();
    verifier.verify_proof(proof);
}