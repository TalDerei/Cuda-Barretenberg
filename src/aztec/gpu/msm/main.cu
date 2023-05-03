#include "pippenger.cu"

using namespace std;
using namespace pippenger_common;
using namespace waffle;

int main(int, char**) {
    // Dynamically initialize new 'msm_t' object
    msm_t<g1_gpu::affine_element, scalar_t, point_t> *msm = new msm_t<g1_gpu::affine_element, fr_gpu, point_t>();
    
    // Generate sample curve points SRS 
    auto reference_string = std::make_shared<waffle::FileReferenceString>(NUM_POINTS, "../srs_db/ignition");
    g1::affine_element* points = reference_string->get_monomials();

    // Generate sample scalars
    std::vector<fr> scalars;
    fr element = fr::random_element();
    fr accumulator = element;
    scalars.reserve(NUM_POINTS);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        accumulator *= element;
        scalars.emplace_back(accumulator);
    }

    // Initialize pippenger 'context' object
    Context<bucket_t, point_t, scalar_t, affine_t> *context = msm->pippenger_initialize(points);

    // Execute "Double-And-Add" reference kernel
    g1_gpu::element *final_result_1 = msm->naive_double_and_add(context, NUM_POINTS, points, &scalars[0]);

    // Execute "Pippenger's Bucket Method" kernel
    g1_gpu::element *final_result_2 = msm->msm_bucket_method(context, NUM_POINTS, points, &scalars[0]);

    cout << "final_result_1 is: " << final_result_1[0].x.data[0] << endl;
    cout << "final_result_1 is: " << final_result_1[0].x.data[1] << endl;
    cout << "final_result_1 is: " << final_result_1[0].x.data[2] << endl;
    cout << "final_result_1 is: " << final_result_1[0].x.data[3] << endl;

    cout << "final_result_1 is: " << final_result_1[0].y.data[0] << endl;
    cout << "final_result_1 is: " << final_result_1[0].y.data[1] << endl;
    cout << "final_result_1 is: " << final_result_1[0].y.data[2] << endl;
    cout << "final_result_1 is: " << final_result_1[0].y.data[3] << endl;

    cout << "final_result_1 is: " << final_result_1[0].z.data[0] << endl;
    cout << "final_result_1 is: " << final_result_1[0].z.data[1] << endl;
    cout << "final_result_1 is: " << final_result_1[0].z.data[2] << endl;
    cout << "final_result_1 is: " << final_result_1[0].z.data[3] << endl;

    cout << endl;

    cout << "final_result_2 is: " << final_result_2[0].x.data[0] << endl;
    cout << "final_result_2 is: " << final_result_2[0].x.data[1] << endl;
    cout << "final_result_2 is: " << final_result_2[0].x.data[2] << endl;
    cout << "final_result_2 is: " << final_result_2[0].x.data[3] << endl;

    cout << "final_result_2 is: " << final_result_2[0].y.data[0] << endl;
    cout << "final_result_2 is: " << final_result_2[0].y.data[1] << endl;
    cout << "final_result_2 is: " << final_result_2[0].y.data[2] << endl;
    cout << "final_result_2 is: " << final_result_2[0].y.data[3] << endl;

    cout << "final_result_2 is: " << final_result_2[0].z.data[0] << endl;
    cout << "final_result_2 is: " << final_result_2[0].z.data[1] << endl;
    cout << "final_result_2 is: " << final_result_2[0].z.data[2] << endl;
    cout << "final_result_2 is: " << final_result_2[0].z.data[3] << endl;

    // Verify the final results match
    msm->verify_result(final_result_1, final_result_2);
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
 * since conditional checks are degrading performance
 */

/**
 * Exploring performance bottlenecks:
 * 
 * After setting up a simple bench, the bottleneck is not curve addition or the conditional doubling....a single for loop
 * with 4 threads and 1 block can execute 2^15 additions in 4000 ms. I need to benchmark on A10 to figure out what's going on...
 * Usage of too many registers might be the problem here. Additionally, I need to figure out why there are still correctness errors
 * (ie. some tests pass and some fail on the same run)!
 * 
 * The issue is not with initialize buckets or split scalars kernels. Print statements add some time to the timings, but not much 
 * in this case....accumulate buckets kernel is also not the problem since again, additions are fast. The bucket sum reduction kernel
 * is slow for a couple reasons...1. it's a serial operation, 2. there are 2 additions per iteration, 3. the parallelism amount is low
 * (launch parameters include 26 blocks and 4 threads ONLY). Need to look more into this kernel. We'll also remove the synchronization 
 * primitive cudaDeviceSynchronize since kernel launches are asynchrous but execute serially in the same stream. CudaDeviceSynchroniza
 * and swithching from unified memory to cudaMalloc didn't make any change really. I still haven't pinned point the main performance 
 * bottleneck...which I suspect is the the number of registers not sure. Okay...now the conditional double adds 2x execution time in these
 * kernels, but not in the baseline benches for some reason...need to reconcile that difference as well.  
 * 
 * I'm curious if since the maximum registers per thread are the same, will the performance be similiar between A10 and P100
 * if the bottleneck is the number of registers?
*/