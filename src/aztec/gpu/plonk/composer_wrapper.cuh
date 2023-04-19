#include "prover_wrapper.cu"

using namespace std;
using namespace waffle;
using namespace prover_wrapper;

namespace composer_gpu_wrapper {

/**
 * Polymorphic 'composer_gpu' class that represents the top-level prover functions and 
 * derives the 'StandardComposer' base class, which derives the 'ComposerBase' base class.
 */

class ComposerWrapper : public StandardComposer {
    public:    
        // Inline constructor 
        ComposerWrapper () : StandardComposer() {}
        
        StandardComposer composer_wrapper;
        
        virtual Prover create_prover() override; 
};

}