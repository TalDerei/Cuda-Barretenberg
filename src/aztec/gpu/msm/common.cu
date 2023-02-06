#include "./common.cuh"

using namespace std;

namespace pippenger_common {

/***************************************** Function declerations for 'pippenger_t' class  *****************************************/

/**
 * Initialize parameters for MSM
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
pippenger_t<bucket_t, point_t, scalar_t, affine_t> pippenger_t<bucket_t, point_t, scalar_t, affine_t>::initialize_msm(size_t npoints) {
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

/**
 * Calculate the amount of storage neccessary to store bases 
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::get_size_bases(pippenger_t &config) {
    return config.n * sizeof(point_t);
}

/**
 * Calculate the amount of storage neccessary to store scalars 
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::get_size_scalars(pippenger_t &config) {
    return config.n * sizeof(scalar_t);
}

/**
 * Calculate the amount of storage neccessary to store buckets 
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::get_size_buckets(pippenger_t &config) {
    return config.N * sizeof(bucket_t) * NWINS * (1 << WBITS);
}

/**
 * Allocate device storage for bases
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::allocate_bases(pippenger_t &config) {
    return device_base_ptrs.allocate(get_size_bases(config));
}

/**
 * Allocate device storage for scalars
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::allocate_scalars(pippenger_t &config) {
    return device_scalar_ptrs.allocate(get_size_scalars(config));
}

/**
 * Allocate device storage for buckets
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::allocate_buckets(pippenger_t &config) {
    return device_bucket_ptrs.allocate(get_size_buckets(config));
}

/**
 * Return size of base pointers
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::num_base_ptrs() {
    return device_base_ptrs.size();
}

/**
 * Return size of scalar pointers
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::num_scalar_ptrs() {
    return device_scalar_ptrs.size();
}

/**
 * Return size of bucket pointers
*/
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::num_bucket_ptrs() {
    return device_bucket_ptrs.size();
}

template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::transfer_bases_to_device(
pippenger_t &config, size_t d_points_idx, const affine_t points[], size_t ffi_affine_sz, cudaStream_t s) {
    cudaSetDevice(0);
    cudaStream_t stream = 0; // default stream
    affine_t *d_points = device_base_ptrs[d_points_idx];

    if (ffi_affine_sz != sizeof(*d_points)) {
        cudaError_t status = cudaMemcpy2DAsync(
            d_points, sizeof(*d_points), points, ffi_affine_sz,ffi_affine_sz, config.npoints,cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
            printf("Error copying bases to device\n");
        }
    }
    else {
        cudaError_t status = cudaMemcpy2DAsync(d_points, points, config.npoints*sizeof(*d_points), cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
            printf("Error copying bases to device\n");
        } 
    }
}

template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::transfer_scalars_to_device(
pippenger_t &config, size_t d_scalars_idx, const scalar_t scalars[], cudaStream_t s) {
    cudaSetDevice(0);
    cudaStream_t stream = 0; // default stream
    scalar_t *d_scalars = device_scalar_ptrs[d_scalars_idx];

    cudaError_t status = cudaMemcpy2DAsync(d_scalars, scalars, config.npoints*sizeof(*d_scalars), cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        printf("Error copying bases to device\n");
    }
}

/***************************************** Function declerations for 'device_ptr' class  *****************************************/

template <class T>
size_t device_ptr<T>::allocate(size_t bytes) {
    T* d_ptr;
    cudaMalloc(&d_ptr, bytes);
    d_ptrs.push_back(d_ptr);
    return d_ptrs.size() - 1;
}

template <class T>
size_t device_ptr<T>::size() {
    return d_ptrs.size();
}

}