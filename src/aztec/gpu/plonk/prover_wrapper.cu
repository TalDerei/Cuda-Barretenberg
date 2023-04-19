#include "prover_wrapper.cuh"

/**
 * Construct proof by running prover rounds
 *
 * @return proof
 * */
waffle::plonk_proof &prover_wrapper::ProverWrapper::construct_proof() {
    cout << "Entered virtual construct_proof()" << endl;
    
    // Execute init round. Randomize witness polynomials
    execute_preamble_round();
    queue.process_queue();

    // Compute wire precommitments and sometimes random widget round commitments
    execute_first_round();
    queue.process_queue();

    // Fiat-Shamir eta + execute random widgets
    execute_second_round();
    queue.process_queue();

    // Fiat-Shamir beta, execute random widgets (Permutation 
    // widget is executed here) and fft the witnesses
    execute_third_round();
    queue.process_queue();

    execute_fourth_round();
    queue.process_queue();

    execute_fifth_round();
    execute_sixth_round();
    queue.process_queue();

    return export_proof();
}