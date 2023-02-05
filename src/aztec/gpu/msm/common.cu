#include "./common.cuh"

using namespace std;

namespace pippenger_common {

/**
 * Initialize parameters for MSM
*/
template <class bucket_t, class point_t, class scalar_t>
pippenger_t<bucket_t, point_t, scalar_t> pippenger_t<bucket_t, point_t, scalar_t>::initialize_msm(size_t npoints) {
    // Set cuda device parameters    
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    sm_count = prop.multiProcessorCount;

    // Set MSM parameters
    pippenger_t config;
    config.npoints = npoints;
    config.n = (npoints + WARP - 1) & ((size_t)0 - WARP);
    config.N = (sm_count * 256) / (NTHREADS * NWINS);

    return config;
}

}