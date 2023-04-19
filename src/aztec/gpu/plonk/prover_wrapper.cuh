#include <plonk/proof_system/prover/prover.hpp>
#include <plonk/proof_system/public_inputs/public_inputs.hpp>
#include <plonk/proof_system/utils/linearizer.hpp>
#include <chrono>
#include <ecc/curves/bn254/scalar_multiplication/scalar_multiplication.hpp>
#include <polynomials/iterate_over_domain.hpp>
#include <polynomials/polynomial_arithmetic.hpp>
#include <iostream>

using namespace std;
using namespace waffle;

namespace prover_wrapper {

/**
 * Polymorphic 'prover_wrapper' class that represents the top-level prover functions and 
 * derives the 'ProverBase' base class.
 */
class Prover_Wrapper : public ProverBase<standard_settings> {
    public:    
        virtual plonk_proof& construct_proof() override;
};

}