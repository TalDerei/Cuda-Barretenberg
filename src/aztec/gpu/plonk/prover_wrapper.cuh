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