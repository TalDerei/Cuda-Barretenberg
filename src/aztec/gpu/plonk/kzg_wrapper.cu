#include "kzg_wrapper.cuh"

using namespace kzg_gpu_wrapper;

void kzg_gpu_wrapper::KzgWrapper::commit(fr *coefficients, std::string tag, fr item_constant, work_queue &queue) {
    cout << "Entered virtual commit()" << endl;
    
    queue.add_to_queue({
        work_queue::WorkType::SCALAR_MULTIPLICATION,
        coefficients,
        tag,
        item_constant,
        0,
    });
}
