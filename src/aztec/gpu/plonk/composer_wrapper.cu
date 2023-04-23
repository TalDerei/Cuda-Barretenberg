#include "composer_wrapper.cuh"

using namespace composer_gpu_wrapper;

/**
 * Create prover.
 *  1. Compute the starting polynomials (selector, witness, and sigma permutation polynomials).
 *  2. Initialize Prover with them.
 *  3. Add Permutation and arithmetic widgets to the prover.
 *  4. Add KateCommitmentScheme to the prover.
 *
 * @return Initialized prover.
 * */
waffle::Prover ComposerWrapper::create_prover() {
    cout << "Entered virtual create_prover()" << endl;

    compute_proving_key();
    compute_witness();

    Prover output_state(circuit_proving_key, waffle::ComposerBase::witness, create_manifest(public_inputs.size()));

    std::unique_ptr<ProverPermutationWidget<3, false>> permutation_widget =
        std::make_unique<ProverPermutationWidget<3, false>>(circuit_proving_key.get(), waffle::ComposerBase::witness.get());
    std::unique_ptr<ProverArithmeticWidget<standard_settings>> arithmetic_widget =
        std::make_unique<ProverArithmeticWidget<standard_settings>>(circuit_proving_key.get(), waffle::ComposerBase::witness.get());

    output_state.random_widgets.emplace_back(std::move(permutation_widget));
    output_state.transition_widgets.emplace_back(std::move(arithmetic_widget));

    // Create object and pass it to a smart pointer
    std::unique_ptr<KateCommitmentScheme<standard_settings>> kzg(new kzg_gpu_wrapper::KzgWrapper());
    kzg = std::make_unique<kzg_gpu_wrapper::KzgWrapper>();
    output_state.commitment_scheme = std::move(kzg);  

    return output_state;
}