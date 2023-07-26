#include <plonk/proof_system/prover/work_queue.hpp>

using namespace waffle;

namespace queue_gpu_wrapper {

/**
 * Polymorphic 'QueueWrapper' class that represents the top-level queue operations.
 */
class QueueWrapper : public work_queue {
    public:    
        QueueWrapper(proving_key* prover_key = nullptr, program_witness* program_witness = nullptr,
                    transcript::StandardTranscript* prover_transcript = nullptr) 
                    : work_queue(prover_key, program_witness, prover_transcript) {}

        virtual void process_queue() override;
        virtual ~QueueWrapper() {};
};

}