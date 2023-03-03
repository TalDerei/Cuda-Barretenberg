#include "pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    // Dynamically initialize new 'msm_t' object
    msm_t<g1::affine_element, fr_gpu> *msm = new msm_t<g1::affine_element, fr_gpu>();
    
    // Read curve points
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();

    // Read scalars
    fr_gpu *scalars = msm->read_scalars();

    // Initlize MSM
    Context<bucket_t, point_t, scalar_t, affine_t> *context = msm->pippenger_initialize(points);

    // Execute MSM
    msm->pippenger_execute(context, NUM_POINTS, points, scalars);
}

// TODO:
// Implement simple double and add algorithm in gpu kernel
// Test kernel on smaller instance size (i.e. grid and blocks)
// Understand these kernel functions better + add comments
// Need to figure out how to take FF elements, instead of elements of limbs