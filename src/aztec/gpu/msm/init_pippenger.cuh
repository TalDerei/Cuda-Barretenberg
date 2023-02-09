#include "./common.cu"
#include <iostream>
#include <memory>

using namespace std;

namespace pippenger_common {

/**
 * Pippenger top-level function prototypes 
 */
template < typename T >
class msm_t {
    public: 
        msm_t() {}

        T read_points_scalars();

        void pippenger_init(T points);
        
        void pippenger_execute(T points);
};

}