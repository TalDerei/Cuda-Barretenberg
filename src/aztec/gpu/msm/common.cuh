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

#define BITSIZE 254
#define C 10
size_t NUM_POINTS = 1 << 11;

typedef element<fq_gpu, fr_gpu> point_t;
typedef fr_gpu scalar_t;

/**
 * Allocate device storage and buffers 
 */
template < typename T >
class device_ptr {
    public:
        vector<T*> d_ptrs;

        device_ptr() {}
        
        ~device_ptr() {}

        void allocate(size_t bytes);

        size_t size();

        T* operator[](size_t i);
};

/**
 * Parameters for CUB routines
 */
class cub_routines {
    public:
        unsigned *bucket_offsets;
        unsigned *bucket_sizes;
        unsigned *single_bucket_indices;
        unsigned *bucket_indices;
        unsigned *point_indices;
        unsigned *sort_indices_temp_storage;
        unsigned *nof_buckets_to_compute;
        unsigned *encode_temp_storage;
        unsigned *offsets_temp_storage;
};

/**
 * Initialize pippenger's bucket method for MSM algorithm
 */
template < typename point_t, typename scalar_t > 
class pippenger_t {        
    public:
        pippenger_t() { cudaSetDevice(0); }
            
        size_t get_size_bases(pippenger_t &config);

        size_t get_size_scalars(pippenger_t &config);
                
        void allocate_bases(pippenger_t &config);
        
        void allocate_scalars(pippenger_t &config);
        
        void transfer_bases_to_device(pippenger_t &config, point_t *device_bases_ptrs, const point_t *points, cudaStream_t stream);
        
        void transfer_scalars_to_device(pippenger_t &config, scalar_t *device_scalar_ptrs, fr *scalars, cudaStream_t stream);
                
        void print_result(g1_gpu::element *result_1, g1_gpu::element **result_2);
        
        void verify_result(point_t *result_1, point_t **result_2);
        
        point_t* execute_bucket_method(
            pippenger_t &config, scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints, cudaStream_t stream
        );

        void execute_cub_routines(pippenger_t &config, cub_routines *params, cudaStream_t stream);

        void calculate_windows(pippenger_t &config, size_t npoints);

        device_ptr<point_t> device_base_ptrs;
        device_ptr<scalar_t> device_scalar_ptrs;
        cudaStream_t *streams;
        cub_routines *params;
        int num_streams;
        int device;
        size_t npoints;
        size_t num_buckets;
        int windows;
};
typedef pippenger_t<point_t, scalar_t> pipp_t;

/**
 * Context used to store persistent state
 */
template < typename point_t, typename scalar_t > 
struct Context {
    public: 
        pipp_t pipp;
};

}