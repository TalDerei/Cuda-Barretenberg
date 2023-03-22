#include "pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    // Dynamically initialize new 'msm_t' object
    msm_t<g1::affine_element, scalar_t, point_t> *msm = new msm_t<g1::affine_element, fr_gpu, point_t>();
    
    // Read curve points
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();

    // Initlize MSM
    Context<bucket_t, point_t, scalar_t, affine_t> *context = msm->pippenger_initialize(points);

    // Read scalars
    // Switch order
    // fr_gpu *scalars;
    // cudaMallocManaged(&scalars, POINTS * LIMBS * sizeof(uint64_t));
    // context->pipp.read_scalars(scalars);

    // Execute MSM
    // msm->pippenger_execute(context, NUM_POINTS, points, scalars);

    // Naive MSM
    // msm->naive_msm(context, NUM_POINTS, points);

    // MSM "Bucket Method"
    msm->msm_bucket_method(context, NUM_POINTS, points);
}

/**
 * TODO: C++/cuda inline of functions
 * TODO: Benchmark Ingonyama "Icicle" MSM
 * TODO: Add multiple stream support
 * TODO: Look into cuda graph support 
 * TODO: change unified memory to pinned host memory
 * TODO: look into asynchronous transfers 
 * TODO: seperate kernel code
 * TODO: read affine points instead of jacobian
 * TODO: change to memcpy instead of reading from files
 * TODO: modify number of scalars
 * TODO: incorperate cooperative groups in accmulation 
 * TODO: choose block sizes based on occupancy in terms of active blocks
 * TODO: free memory
 * TODO: add timer
 * TODO: look into shared memory optimizations
 * TODO: remove extraneous loops
 * TODO: adjust kernel parameters to reduce overhead
 * TODO: check jacobian addition by 0, and P == Q checks
 * TODO: look into loop unrolling
 */


/*
Ways to speed up:
- remove synchronizatin pritives
- remove print statements
- change unified memory to pinned host memory
- change size of windows
*/

/*
Open Questions:

1. We're not handling all the bucket modules for some reason....
It's either a occupancy limit problem or data initialization probelem. It doesn't seem to be a data initialization problem,
because shifting the blockId by a constant factor revealed the missing data. It has to be an occupancy limit. The naive way 
to deal with this is moving to a more powerful GPU, i.e. A10. I need to figure out another standard way to deal with this.

2. How are these buckets logically seperated into bucket modules? where's the logical seperation happening in the code?
Using offsets

3. And why are some buckets empty when calling the accumulate_buckets_kernel, and why do we index on single_bucket_indices
instead of point_indices?
We have N buckets to compute, each with a variable size M. N * M = total num buckets. We're sorting based on single_bucket_indices,
so not every sequential indice correpsonds to a filled bucket. So buckets will be empty by design. The more threads you have, the more 
densely populated these buckets will be. 
*/

/*
Steps:
    1. Initialize buckets num_modules << c
    2. Split b-bit scalats into c-bit scalars and assign each of them in a "bucket_index". One sub-scalar per index.
    3. Then group the similiar sub-scalars together into buckets (single_bucket_indices and bucket_sizes) for each bucket. 
        This is just a logical mapping. The total number of unique buckets will be smaller. So to recape, e.g. we have 16216 unique buckets,
        those are split into single_bucket_indices up to 26k, and each unique bucket has a non-zero size. 
    4. Then launch the bucket accumulation to add the points together in each bucket. This is done for each bucket module. Note
        that some buckets are empty since they weren't filled. See #3 above.
    5. Apply a sum reduction to reduce all the buckets into a single value in each bucket module
    6. Final accumulation step to sum up the partial sums into a final output.
*/