#include "reference_string.cu"
#include "util/thread_pool.hpp"
#include "error.cuh"
#include <cuda.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <ecc/curves/bn254/fr.hpp>
#include <plonk/proof_system/types/program_settings.hpp>
#include <plonk/reference_string/file_reference_string.hpp>

using namespace std;
using namespace barretenberg;

namespace pippenger_common {

/**
 * Global variables
*/
#define BITSIZE 254
#define C 10
#define MODULES ((BITSIZE / C) + 1)
size_t NUM_POINTS = 1 << 10;

/**
 * Typedef points, scalars, and buckets 
 */
typedef element<fq_gpu, fr_gpu> point_t;
typedef fr_gpu scalar_t;
typedef element<fq_gpu, fr_gpu> bucket_t;

/**
 * Allocate device storage and buffers. The primary
 * data structure is a vector of initialized pointers
 * to memory locations. 
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
 * Initialize pippenger's bucket method for MSM algorithm
 */
template < typename bucket_t, typename point_t, typename scalar_t > 
class pippenger_t {
    private:
        device_ptr<point_t> device_base_ptrs;
        device_ptr<scalar_t> device_scalar_ptrs;
        device_ptr<bucket_t> device_bucket_ptrs;
    public: 
        stream_t default_stream;
        size_t npoints;        
        size_t n;
        int device;

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

        void transfer_bases_to_device(pippenger_t &config, size_t d_points_idx, const point_t points[]);

        void transfer_scalars_to_device(pippenger_t &config, scalar_t *d_scalars_idx, fr *scalars, cudaStream_t aux_stream);
        
        void launch_kernel(pippenger_t &config, size_t d_bases_idx, size_t d_scalar_idx, size_t d_buckets_idx);

        void synchronize_stream(pippenger_t &config);

        void print_result(point_t *result);

        point_t* execute_bucket_method(pippenger_t &config, scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints);
};
typedef pippenger_t<bucket_t, point_t, scalar_t> pipp_t;

/**
 * Context used to store persistent state
 */
template < typename bucket_t, typename point_t, typename scalar_t > 
struct Context {
    public: 
        pipp_t pipp;

        // Indices for device_ptr
        size_t d_points_idx; 
        size_t d_buckets_idx; 
        size_t d_scalar_idx; 
        scalar_t *h_scalars; // will need to make some conversion between scalar_t and fr...
};

}