#include <plonk/proof_system/commitment_scheme/kate_commitment_scheme.hpp>

using namespace std;
using namespace waffle;

namespace kzg_gpu_wrapper {

/**
 * Polymorphic 'KzgWrapper' class that represents the top-level commitment functions and 
 * derives the 'KateCommitmentScheme' base class, which derives the 'CommitmentScheme' base class.
 * The virtual functions use cuda memory allocation strategies.
 */
class KzgWrapper : public KateCommitmentScheme<standard_settings> {
    public:    
        virtual void commit(fr* coefficients, std::string tag, fr item_constant, std::unique_ptr<work_queue>& queue) override;
};

}