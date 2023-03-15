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
    fr_gpu *d_scalars;
    cudaMallocManaged(&d_scalars, POINTS * LIMBS * sizeof(uint64_t));
    fr_gpu *scalars = context->pipp.read_scalars(d_scalars);

    // Execute MSM
    msm->pippenger_execute(context, NUM_POINTS, points, scalars);

    // Naive MSM
    msm->naive_msm(context, NUM_POINTS, points, scalars);

    // MSM "Bucket Method"
    msm->msm_bucket_method(context, NUM_POINTS, points, scalars);
}

/**
 * TODO: C++/cuda inline of functions
 * TODO: Benchmark Ingonyama "Icicle" MSM
 */
