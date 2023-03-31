#include "pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    // Dynamically initialize new 'msm_t' object
    msm_t<g1::affine_element, scalar_t, point_t> *msm = new msm_t<g1::affine_element, fr_gpu, point_t>();
    
    // Read curve points
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();

    // Initialize MSM
    Context<bucket_t, point_t, scalar_t, affine_t> *context = msm->pippenger_initialize(points);

    // MSM test
    msm->pippenger_test(context, NUM_POINTS, points);

    // MSM 1
    // msm->pippenger_execute(context, NUM_POINTS, points);

    // MSM 2
    // msm->naive_msm(context, NUM_POINTS, points);

    // MSM 3 -- "Bucket Method"
    // msm->msm_bucket_method(context, NUM_POINTS, points);
}

/**
 * TODO: C++/cuda inline of functions
 * TODO: benchmark Ingonyama "Icicle" MSM
 * TODO: add multiple stream support
 * TODO: look into cuda graph support 
 * TODO: change unified memory to pinned host memory
 * TODO: look into asynchronous transfers 
 * TODO: read affine points instead of jacobian
 * TODO: change to memcpy instead of reading from files
 * TODO: modify number of scalars
 * TODO: incorperate cooperative groups in accmulation 
 * TODO: choose block sizes based on occupancy in terms of active blocks
 * TODO: free memory
 * TODO: look into shared memory optimizations
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
 */