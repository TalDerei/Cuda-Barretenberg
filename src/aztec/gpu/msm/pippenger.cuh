#include "common.cu"
#include <iostream>
#include <memory>

using namespace std;

namespace pippenger_common {

/**
 * Pippenger top-level function prototypes 
 */
template < typename A, typename S, typename J >
class msm_t {
    public: 
        msm_t() {}
        
        Context<bucket_t, point_t, scalar_t, affine_t>* pippenger_initialize(A* points);
        
        void pippenger_test(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, A* points);

        void pippenger_execute(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, A* points);
        
        void naive_msm(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, A* points);

        void msm_bucket_method(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, A* points);
};

}