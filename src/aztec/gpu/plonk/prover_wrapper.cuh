#include "kzg_wrapper.cu"
#include "queue_wrapper.cu"

using namespace std;
using namespace waffle;

namespace prover_wrapper {

/**
 * Polymorphic 'prover_wrapper' class that represents the top-level prover functions and 
 * derives the 'ProverBase' base class.
 */
class ProverWrapper : public ProverBase<standard_settings> {
    public:    
        virtual plonk_proof& construct_proof() override;
};

}