#include "queue_wrapper.cu"

// using namespace std;
using namespace waffle;

namespace prover_gpu_wrapper {

/**
 * Polymorphic 'ProverWrapper' class that represents the top-level prover functions and 
 * derives the 'ProverBase' base class.
 */
class ProverWrapper : public ProverBase<standard_settings> {
    public:    
        virtual ProverBase<standard_settings>& operator=(ProverBase<standard_settings>&& other) override;
        virtual ~ProverWrapper() {};
};

}