#include "pippenger.cu"

using namespace std;
using namespace pippenger_common;
using namespace waffle;

int main(int, char**) {
    // Initialize dynamic 'msm_t' object 
    msm_t<point_t, scalar_t> *msm = new msm_t<point_t, scalar_t>();
    
    // Construct elliptic curve points from SRS
    auto reference_string = std::make_shared<waffle::FileReferenceString>(NUM_POINTS, "../srs_db/ignition");
    g1::affine_element* points = reference_string->get_monomials();

    // Construct random scalars -- move to test file 
    std::vector<fr> scalars;
    fr element = fr::random_element();
    fr accumulator = element;
    scalars.reserve(NUM_POINTS);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        accumulator *= element;
        scalars.emplace_back(accumulator);
    }

    // Initialize dynamic pippenger 'context' object
    Context<point_t, scalar_t> *context = msm->pippenger_initialize(points,  &scalars[0]);

    // Execute "Double-And-Add" reference kernel
    g1_gpu::element *result_1 = msm->naive_double_and_add(context, NUM_POINTS, points, &scalars[0]);

    // Execute "Pippenger's Bucket Method" kernel
    g1_gpu::element *result_2 = msm->msm_bucket_method(context, NUM_POINTS, points, &scalars[0]);

    // Print results 
    context->pipp.print_result(result_1, result_2);

    // Verify the final results match
    context->pipp.verify_result(result_1, result_2);
}

/**
 * TODO: Add "inlining" to C++/cuda functions
 * TODO: benchmark Ingonyama "Icicle" MSM
 * TODO: add multiple stream support
 * TODO: change unified memory to pinned host memory
 * TODO: look into asynchronous transfers 
 * TODO: change to memcpy instead of reading from files
 * TODO: incorperate cooperative groups in accmulation 
 * TODO: choose block sizes based on occupancy in terms of active blocks
 * TODO: free memory
 * TODO: look into shared memory optimizations instead of global memory accesses (100x latency lower than global memory)
 * TODO: remove extraneous loops
 * TODO: adjust kernel parameters to reduce overhead
 * TODO: look into loop unrolling with pragma
 * TODO: more efficient sum reduction kernel
 * TODO: remove synchronization primtives
 * TODO: remove print statements
 * TODO: change size of windows
 * TODO: Address depaul notes on SOL, occupancy achieved, etc.
 * TODO: look into reducing registers and pipelining loads (e.g. __launch_bounds__)
 * TODO: change scalar size from c = 10 to c = 16
 * TODO: change the indexing for the other sum reduction kernel
 * TODO: change indexing of threads from tid to threadrank. maybe it's better need to look into it
 * TODO: loop unroll here -- and account for unused threads after first iteration
 * TODO: clean up comments in kernels
 * TODO: switch jacobian to projective coordinates to eliminate infinity and zero checks 
 * TODO: are conditional checks are degrading performance?
 * TODO: Look into 'Staged concurrent copy and execute' over 'Sequential copy and execute'
 */