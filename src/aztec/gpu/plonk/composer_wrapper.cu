#include "composer_wrapper.cuh"
#include "kzg_wrapper.cu"

// Create verifier: compute verification key,
//  * initialize verifier with it and an initial manifest and initialize commitment_scheme.

/**
 * Create prover.
 *  1. Compute the starting polynomials (q_l, etc, sigma, witness polynomials).
 *  2. Initialize Prover with them.
 *  3. Add Permutation and arithmetic widgets to the prover.
 *  4. Add KateCommitmentScheme to the prover.
 *
 * @return Initialized prover.
 * */
waffle::Prover composer_gpu_wrapper::composer::create_prover() {
    cout << "Entered create_prover (overloaded)!" << endl;

    // Compute the polynomials q_l, q_r, etc. and sigma polynomials
    compute_proving_key();

    // Compute witness polynomials
    compute_witness();
    Prover output_state(circuit_proving_key, witness, create_manifest(public_inputs.size()));

    std::unique_ptr<ProverPermutationWidget<3, false>> permutation_widget =
        std::make_unique<ProverPermutationWidget<3, false>>(circuit_proving_key.get(), witness.get());
    std::unique_ptr<ProverArithmeticWidget<standard_settings>> arithmetic_widget =
        std::make_unique<ProverArithmeticWidget<standard_settings>>(circuit_proving_key.get(), witness.get());

    output_state.random_widgets.emplace_back(std::move(permutation_widget));
    output_state.transition_widgets.emplace_back(std::move(arithmetic_widget));

    std::unique_ptr<KateCommitmentScheme<standard_settings>> kate_commitment_scheme =
        std::make_unique<KateCommitmentScheme<standard_settings>>();

    output_state.commitment_scheme = std::move(kate_commitment_scheme);
    
    return output_state;
}

