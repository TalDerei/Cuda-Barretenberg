#include "prover_wrapper.cu"

using namespace std;
using namespace waffle;
using namespace prover_gpu_wrapper;

namespace composer_gpu_wrapper {

/**
 * Polymorphic 'ComposerWrapper' class that represents the top-level prover and verifier functions 
 * and derives the 'StandardComposer' base class, which derives the 'ComposerBase' base class.
 *
 * 'StandardComposer' contains an enum for selectors (QL, QR, QM, QO, QC), and contains
 * circuit-specific functions for constructing gates. Composer uses a Manifest (
 * ie. transcript) to define the structure of the protocol. The prover progresses 
 * in five rounds, and produces a message at the end of each. After each round, the 
 * message is hashed into the Transcript. The proof π consists of all the round messages. 
 * 
 * 'StandardComposer' inherents from the 'ComposerBase' base class, which defines functions
 * for computing the selector (gate), witness (wire), and sigma permutation polynomials. 
 */
class ComposerWrapper : public StandardComposer {
    public:    
        // Inline constructor 
        ComposerWrapper () : StandardComposer() {}

        StandardComposer composer_wrapper;
        
        virtual Prover create_prover() override; 
};

}