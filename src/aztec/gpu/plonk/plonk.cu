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

    cout << "------------Constructing witness---------------------" << endl; 
    waffle::StandardComposer composer = waffle::StandardComposer();
    generate_test_plonk_circuit(composer, static_cast<size_t>(MAX_GATES));
    composer.compute_witness();
    
    cout << "------------Constructing prover and proving key---------------------" << endl; 
    composer.compute_proving_key();    
    provers = composer.create_prover();

    cout << "------------Generatring proof---------------------" << endl; 
    proofs = provers.construct_proof();

    cout << "------------Constructing verifier instance and verify proof---------------------" << endl; 
    composer.preprocess();
    verifiers = composer.create_verifier();
    verifiers.verify_proof(proofs);
}

