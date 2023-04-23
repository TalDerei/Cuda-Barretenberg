#include "prover_wrapper.cuh"

using namespace prover_gpu_wrapper;

/**
 * Operator overloading work_queue 
 */
ProverBase<standard_settings>& prover_gpu_wrapper::ProverWrapper::operator=(ProverBase<standard_settings>&& other) {
    cout << "Entered virtual operator=()" << endl;
    
    n = other.n;
    random_widgets.resize(0);
    transition_widgets.resize(0);
    for (size_t i = 0; i < other.random_widgets.size(); ++i) {
        random_widgets.emplace_back(std::move(other.random_widgets[i]));
    }
    for (size_t i = 0; i < other.transition_widgets.size(); ++i) {
        transition_widgets.emplace_back(std::move(other.transition_widgets[i]));
    }
    transcript = other.transcript;
    key = std::move(other.key);
    witness = std::move(other.witness);
    commitment_scheme = std::move(other.commitment_scheme);

    // Create 'QueueWrapper' object and pass it to a smart pointer
    std::unique_ptr<work_queue> queue_(new queue_gpu_wrapper::QueueWrapper(key.get(), witness.get(), &transcript));
    queue_ = std::make_unique<queue_gpu_wrapper::QueueWrapper>(key.get(), witness.get(), &transcript);
    queue = std::move(queue_);  

    return *this;
}