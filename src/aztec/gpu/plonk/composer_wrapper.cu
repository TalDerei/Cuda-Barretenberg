#include "composer_wrapper.cuh"

/**
 * Create prover
 * 
 * @return Initialized prover
 * */
waffle::Prover composer_gpu_wrapper::ComposerWrapper::create_prover() {
    cout << "Entered virtual create_prover()" << endl;

    // Construct proving key
    compute_proving_key();

    // Compute the starting polynomials (q_l, etc, sigma, witness polynomials)
    compute_witness();
    Prover output_state(circuit_proving_key, waffle::ComposerBase::witness, create_manifest(public_inputs.size()));

    // Add Permutation and arithmetic widgets to the prover
    std::unique_ptr<ProverPermutationWidget<3, false>> permutation_widget =
        std::make_unique<ProverPermutationWidget<3, false>>(circuit_proving_key.get(), waffle::ComposerBase::witness.get());
    std::unique_ptr<ProverArithmeticWidget<standard_settings>> arithmetic_widget =
        std::make_unique<ProverArithmeticWidget<standard_settings>>(circuit_proving_key.get(), waffle::ComposerBase::witness.get());

    output_state.random_widgets.emplace_back(std::move(permutation_widget));
    output_state.transition_widgets.emplace_back(std::move(arithmetic_widget));

    // Add KateCommitmentScheme to the prover
    std::unique_ptr<KateCommitmentScheme<standard_settings>> kate_commitment_scheme =
        std::make_unique<KateCommitmentScheme<standard_settings>>();

    output_state.commitment_scheme = std::move(kate_commitment_scheme);
    
    return output_state;
}
