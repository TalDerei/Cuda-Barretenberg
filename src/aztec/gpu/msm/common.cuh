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

size_t NUM_POINTS = 1 << 15;
static const size_t NUM_BATCH_THREADS = 2;

/**
 * Typedef points, scalars, and buckets 
 */
typedef element<fq_gpu> point_t;
typedef fr_gpu scalar_t;
typedef affine_element<fq_gpu> bucket_t;
typedef affine_element<fq_gpu> affine_t;

/**
 * Allocate device storage and buffers
 */
template < typename T >
class device_ptr {
    public:
        vector<T*> d_ptrs;

        device_ptr() {}
        
        ~device_ptr() {}

        size_t allocate(size_t bytes);

        size_t size();

        T* operator[](size_t i);
};

/**
 * Allocate cuda streams
 */
template < typename T >
class stream_t {
    public:
        cudaStream_t stream;

        stream_t(int device);

        ~stream_t() {}
};

/**
 * Store results
 */
template < typename T >
class result_t {
    T ret[NWINS][NTHREADS][2];
    public:
        result_t() {}
};

/**
 * Initialize pippenger's bucket method for MSM algorithm
 */
template < typename bucket_t, typename point_t, typename scalar_t, typename affine_t > 
class pippenger_t {
    private:
        size_t sm_count;
        device_ptr<affine_t> device_base_ptrs;
        device_ptr<scalar_t> device_scalar_ptrs;
        device_ptr<bucket_t> device_bucket_ptrs;
    public: 
        typedef vector<result_t<bucket_t>> result_container_t;
        size_t npoints;        
        size_t N;
        size_t n;        
    
        pippenger_t initialize_msm(size_t npoints);
        
        size_t get_size_bases(pippenger_t &config);

        size_t get_size_scalars(pippenger_t &config);

        size_t get_size_buckets(pippenger_t &config);

        size_t allocate_bases(pippenger_t &config);

        size_t allocate_scalars(pippenger_t &config);
        
        size_t allocate_buckets(pippenger_t &config);

        size_t num_base_ptrs();

        size_t num_scalar_ptrs();

        size_t num_bucket_ptrs();

        void transfer_bases_to_device(pippenger_t &config, size_t d_points_idx, const affine_t points[], size_t ffi_affine_sz);

        void transfer_scalars_to_device(pippenger_t &config, size_t d_scalars_idx, const scalar_t scalars[], cudaStream_t s);
        
        result_container_t result_container(pippenger_t &config);
};
typedef pippenger_t<bucket_t, point_t, scalar_t, affine_t> pipp_t;

/**
 * Context used to store persistent state
 */
template < typename bucket_t, typename point_t, typename scalar_t, typename affine_t > 
struct Context {
    public: 
        pipp_t pipp;

        size_t ffi_affine_sz; 
        size_t d_points_idx; 
        size_t d_buckets_idx; 
        size_t d_scalar_idx[NUM_BATCH_THREADS];  
        scalar_t *h_scalars;

        typename pipp_t::result_container_t result0;
        typename pipp_t::result_container_t result1;
};

}