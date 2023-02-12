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

        T* read_points_scalars();

        Context<bucket_t, point_t, scalar_t, affine_t>* pippenger_initialize(T* points);
        
        void pippenger_execute(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, T* points);
};

}