#include "common.cu"
#include <iostream>
#include <memory>

namespace pippenger_common {

/**
 * Pippenger top-level function prototypes 
 */
template < typename P, typename S >
class msm_t {
    public: 
        msm_t() {}
        
        Context<point_t, scalar_t>* pippenger_initialize(g1::affine_element* points, fr *scalars, int rounds, size_t npoints);

        g1_gpu::element* msm_double_and_add(
            Context<point_t, scalar_t> *context, size_t npoints, g1::affine_element* points, fr *scalars
        );

        g1_gpu::element** msm_bucket_method(
            Context<point_t, scalar_t> *context, g1::affine_element* points, fr *scalars, int num_streams
        );
};

}