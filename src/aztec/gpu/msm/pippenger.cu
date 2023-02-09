#include "./init_pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    // Dynamically initialize new msm_t object
     msm_t<g1::affine_element *> *msm = new msm_t<g1::affine_element *>();

    // Read points and scalars
    g1::affine_element* points = msm->read_points_scalars();
    
    // Initlize MSM
    msm->pippenger_init(points);

    // Perform MSM
    msm->pippenger_execute(points);
}