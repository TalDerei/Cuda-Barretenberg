#include <plonk/proof_system/commitment_scheme/kate_commitment_scheme.hpp>

using namespace std;
using namespace waffle;

namespace kzg_wrapper {

/**
 * Polymorphic 'kzg_wrapper' class that represents the top-level commitment functions and 
 * derives the 'KateCommitmentScheme' base class, which derives the 'CommitmentScheme' base class.
 */
class KzgWrapper : public KateCommitmentScheme<standard_settings> {
    public:    
        // These virtual functions include cuda memory allocation strategies
        
        // Include kate commitment scheme and process_queue functions
};

}