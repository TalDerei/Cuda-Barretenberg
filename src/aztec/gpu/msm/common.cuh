#include "reference_string.cu"
#include "util/thread_pool.hpp"
#include "error.cuh"
#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

namespace pippenger_common {

/**
 * Global variables
*/
#define WARP 64                               // Warp size 
#define NTHREADS 256                          // Thread count      
#define NBITS 255                             // Field elements size
#define WBITS 17                              // Scalar size
#define NWINS ((NBITS + WBITS - 1) / WBITS)   // Windowing size 

// change NUM_POINTS
size_t NUM_POINTS = 1 << 11;
static const size_t NUM_BATCH_THREADS = 1;
static thread_pool_t batch_pool(NUM_BATCH_THREADS);

/**
 * Typedef points, scalars, and buckets 
 */
typedef element<fq_gpu, fr_gpu> point_t;
typedef fr_gpu scalar_t;
typedef affine_element<fq_gpu, fr_gpu> bucket_t;
typedef affine_element<fq_gpu, fr_gpu> affine_t;

/**
 * Allocate device storage and buffers
 */
template < typename T >
class device_ptr {
    public:
        // Vector of initialized pointers to memory locations
        vector<T*> d_ptrs;

        device_ptr() {}
        
        ~device_ptr() {}

        size_t allocate(size_t bytes);

        size_t size();

        T* operator[](size_t i);
};

/**
 * Create and destroy cuda streams
 */
class stream_t {
    public:
        cudaStream_t stream;

        stream_t(int device) {
            // Set GPU device
            CUDA_WRAPPER(cudaSetDevice(device));

            // Creates a new asynchronous stream
            CUDA_WRAPPER(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }

        inline operator decltype(stream)() { 
            return stream; 
        }
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
        device_ptr<affine_t> device_base_ptrs;
        device_ptr<scalar_t> device_scalar_ptrs;
        device_ptr<bucket_t> device_bucket_ptrs;
    public: 
        typedef vector<result_t<bucket_t>> result_container_t;
        stream_t default_stream;
        int device;
        size_t sm_count;
        size_t npoints;        
        size_t n;
        size_t N;

        // Constructor method
        pippenger_t() : default_stream(0) {
            device = 0;
        }        
    
        pippenger_t initialize_msm(pippenger_t &config, size_t npoints);
        
        size_t get_size_bases(pippenger_t &config);

        size_t get_size_scalars(pippenger_t &config);

        size_t get_size_buckets(pippenger_t &config);

        size_t allocate_bases(pippenger_t &config);

        size_t allocate_scalars(pippenger_t &config);
        
        size_t allocate_buckets(pippenger_t &config);

        size_t num_base_ptrs();

        size_t num_scalar_ptrs();

        size_t num_bucket_ptrs();

        void transfer_bases_to_device(pippenger_t &config, size_t d_points_idx, const affine_t points[]);

        void transfer_scalars_to_device(pippenger_t &config, size_t d_scalars_idx, const scalar_t scalars[], cudaStream_t aux_stream);
        
        result_container_t result_container(pippenger_t &config);

        void launch_kernel(pippenger_t &config, size_t d_bases_idx, size_t d_scalar_idx, size_t d_buckets_idx);

        template <typename... Types>
        void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim, cudaStream_t stream, Types... args);

        void synchronize_stream(pippenger_t &config);

        void print_result(point_t *result);

        point_t* execute_bucket_method(scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints);
};
typedef pippenger_t<bucket_t, point_t, scalar_t, affine_t> pipp_t;

/**
 * Context used to store persistent state
 */
template < typename bucket_t, typename point_t, typename scalar_t, typename affine_t > 
struct Context {
    public: 
        pipp_t pipp;

        // Indices for device_ptr
        size_t d_points_idx;  // or this should change
        size_t d_buckets_idx; 
        size_t d_scalar_idx; // this should change
        scalar_t *h_scalars;

        typename pipp_t::result_container_t result0;
        typename pipp_t::result_container_t result1;
};

}