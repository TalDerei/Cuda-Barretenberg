#include "./init_pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    g1::affine_element* points = read_points_scalars();
    
    pippenger_init(points);
}