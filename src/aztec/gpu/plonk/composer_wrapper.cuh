#include <plonk/composer/standard_composer.hpp>
#include <ecc/curves/bn254/scalar_multiplication/scalar_multiplication.hpp>
#include <numeric/bitop/get_msb.hpp>
#include <plonk/composer/standard/compute_verification_key.hpp>
#include <plonk/proof_system/widgets/transition_widgets/arithmetic_widget.hpp>
#include <plonk/proof_system/widgets/random_widgets/permutation_widget.hpp>
#include <plonk/proof_system/types/polynomial_manifest.hpp>
#include <plonk/proof_system/commitment_scheme/kate_commitment_scheme.hpp>
#include <unordered_set>
#include <unordered_map>
#include <iostream> 

using namespace std;
using namespace waffle;

namespace composer_gpu_wrapper {

/**
 * Polymorphic 'composer_gpu' class that represents the top-level prover functions and 
 * derives the 'StandardComposer' base class, which derives the 'ComposerBase' base class.
 */
class composer : public StandardComposer {
    public:    
        // Inline constructor 
        composer () : StandardComposer() {}
        StandardComposer standard_composer;

        Prover create_prover();    
};

}