#include "./init_pippenger.cu"

using namespace std;
using namespace pippenger_common;

int main(int, char**) {
    read_points_scalars();
    
    pippenger_init();
}