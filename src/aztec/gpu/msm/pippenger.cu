#include "./init_pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    // Dynamically initialize new 'msm_t' object
    msm_t<g1::affine_element, fr_gpu> *msm = new msm_t<g1::affine_element, fr_gpu>();

    // Read elliptic curve points
    g1::affine_element *points = msm->read_curve_points();

    // Read scalars
    fr_gpu *scalars = msm->read_scalars();
    
    // Initlize MSM
    Context<bucket_t, point_t, scalar_t, affine_t> *context = msm->pippenger_initialize(points);

    // Perform MSM
    msm->pippenger_execute(context, NUM_POINTS, points, scalars);
}