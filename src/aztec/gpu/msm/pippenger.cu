#include "./init_pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    // Dynamically initialize new 'msm_t' object
     msm_t<g1::affine_element> *msm = new msm_t<g1::affine_element>();

    // Read points and scalars
    g1::affine_element* points = msm->read_points_scalars();
    
    // Initlize MSM
    Context<bucket_t, point_t, scalar_t, affine_t> *context = msm->pippenger_initialize(points);

    // Perform MSM
    // TODO: pass in scalars 
    msm->pippenger_execute(context, NUM_POINTS, points);
}