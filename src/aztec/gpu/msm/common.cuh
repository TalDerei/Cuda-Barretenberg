#include "reference_string.cu"
#include <cuda.h>

namespace pippenger_common {

/**
 * Global variables
*/
#define WARP 32
#define NTHREADS 256
#define NBITS 253
#define WBITS 17
#define NWINS ((NBITS + WBITS - 1) / WBITS)   

static const size_t NUM_BATCH_THREADS = 2;

/**
 * Typedef points, scalars, and buckets 
*/
typedef element<fq_gpu> point_t;
typedef fq_gpu scalar_t;
typedef affine_element<fq_gpu> bucket_t;

/**
 * Pippenger MSM class
*/
template <typename bucket_t, typename point_t, typename scalar_t> 
class pippenger_t {
    private:
        size_t sm_count;
    public: 
        size_t npoints;        
        size_t N;
        size_t n;        
    
        pippenger_t initialize_msm(size_t npoints);
};
typedef pippenger_t<bucket_t, point_t, scalar_t> pipp_t;

/**
 * MSM context used to store persistent state
*/
template <typename bucket_t, typename point_t, typename scalar_t> 
struct Context {
    public: 
        pipp_t pipp;

        size_t ffi_affine_sz; 
        size_t d_points_idx; 
        size_t d_buckets_idx; 
        size_t d_scalar_idx[NUM_BATCH_THREADS];  
        scalar_t *h_scalars;
    };
}