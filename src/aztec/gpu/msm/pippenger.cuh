#include "common.cu"
#include <iostream>
#include <memory>

namespace pippenger_common {

/**
 * Pippenger top-level function prototypes 
 */
template < typename A, typename S, typename J >
class msm_t {
    public: 
        msm_t() {}
        Context<bucket_t, point_t, scalar_t>* pippenger_initialize(g1::affine_element* points, fr *scalars);

        g1_gpu::element* naive_double_and_add(Context<bucket_t, point_t, scalar_t> *context, size_t npoints, g1::affine_element* points, fr *scalars);

        g1_gpu::element* msm_bucket_method(Context<bucket_t, point_t, scalar_t> *context, size_t npoints, g1::affine_element* points, fr *scalars);
                
        void verify_result(J *result_1, J *result_2);
};

}