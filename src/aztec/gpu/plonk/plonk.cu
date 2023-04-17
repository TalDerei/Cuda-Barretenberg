#include <ecc/curves/bn254/fr.hpp>
#include <numeric/bitop/get_msb.hpp>
#include <plonk/composer/standard_composer.hpp>
#include <plonk/proof_system/prover/prover.hpp>
#include <plonk/proof_system/verifier/verifier.hpp>
#include <stdlib/primitives/field/field.hpp>
#include <iostream>

#include "plonk.cuh"
#include "kzg_wrapper.cuh"

using namespace std;

constexpr size_t MAX_GATES = 1 << 10;

waffle::Prover provers;
waffle::Verifier verifiers;
waffle::plonk_proof proofs;

void generate_test_plonk_circuit(waffle::StandardComposer& composer, size_t num_gates) {
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

    // Initialize composer and generate test plonk circuit
    waffle::StandardComposer composer = waffle::StandardComposer();
    generate_test_plonk_circuit(composer, static_cast<size_t>(MAX_GATES));
    
    cout << "Constructed prover instance!" << endl; 
    provers = composer.create_prover();

    cout << "Constructed verifier instance!" << endl; 
    verifiers = composer.create_verifier();

    cout << "Generated proof!" << endl; 
    proofs = provers.construct_proof();

    cout << "Verified proof!" << endl; 
    verifiers.verify_proof(proofs);

    cout << "Successfully generated and verified proof for circuit of size: " << MAX_GATES << endl;
}

