#include "common.cu"
#include <iostream>
#include <memory>

using namespace std;

/**
 * Kernel launch parameters
 */
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 4;
static constexpr size_t POINTS = 1 << 10;

namespace pippenger_common {

/**
 * Pippenger top-level function prototypes 
 */
template < typename A, typename S, typename J >
class msm_t {
    public: 
        msm_t() {}
        
        Context<bucket_t, point_t, scalar_t, affine_t>* pippenger_initialize(A* points);
        
        void pippenger_execute(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, A* points, S* scalars);
        
        void naive_msm(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, A* points);

        void msm_bucket_method(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, A* points);
};

}