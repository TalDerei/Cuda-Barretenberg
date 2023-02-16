#include "./common.cuh"

using namespace std;

namespace pippenger_common {
/***************************************** Function declerations for 'pippenger_t' class  *****************************************/

/**
 * Initialize cuda device and MSM parameters
 */
template <>
pipp_t pipp_t::initialize_msm(size_t npoints) {
    cudaError_t status = cudaSetDevice(device);
    if (status != cudaSuccess) {
        cout << "CUDA error from 'cudaSetDevice': " << cudaGetErrorString(status) << endl;
        throw cudaGetErrorString(status);
    }

    cudaDeviceProp prop;
    cudaError_t status_prop = cudaGetDeviceProperties(&prop, device);
    if (status_prop != cudaSuccess) {
        cout << "CUDA error from 'cudaGetDeviceProperties': " << cudaGetErrorString(status_prop) << endl;
        throw cudaGetErrorString(status_prop);
    }

    // Set streaming multiprocessor count, where each SM contains N cuda cores
    sm_count = prop.multiProcessorCount;

    // Set MSM parameters 
    pipp_t config;
    config.npoints = npoints;
    config.n = (npoints + WARP - 1) & ((size_t)0 - WARP);
    config.N = (sm_count * 256) / (NTHREADS * NWINS);

    return config;
}

/**
 * Calculate the amount of device storage required to store bases 
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::get_size_bases(pippenger_t &config) {
    return config.n * sizeof(point_t);
}

/**
 * Calculate the amount of device storage required to store scalars 
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
size_t pippenger_t<bucket_t, point_t, scalar_t, affine_t>::get_size_scalars(pippenger_t &config) {
    return config.n * sizeof(scalar_t);
}

/**
 * Calculate the amount of device storage required to store buckets 
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

/**
 * Transfer base points to GPU device
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::transfer_bases_to_device(
pippenger_t &config, size_t d_points_idx, const affine_t points[]) {    
    // Set cuda device and default stream
    cudaSetDevice(device);
    cudaStream_t stream = 0; 
    
    affine_t *d_points = device_base_ptrs[d_points_idx];
    
    // cudaMemcpyAsync() is non-blocking and asynchronous variant of cudaMemcpy() that requires pinned memory.
    // Asynchronous transfers enable overalapping data transfers with kernel execution.
    cudaError_t status = cudaMemcpyAsync(d_points, points, config.npoints * sizeof(*d_points), cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(status) << endl;
        throw cudaGetErrorString(status);
    } 
}

/**
 * Transfer scalars to GPU device
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::transfer_scalars_to_device(
pippenger_t &config, size_t d_scalars_idx, const scalar_t scalars[], cudaStream_t aux_stream) {
    // Set cuda device and auxilary stream
    cudaSetDevice(device);
    cudaStream_t stream = aux_stream;

    scalar_t *d_scalars = device_scalar_ptrs[d_scalars_idx];

    cudaError_t status = cudaMemcpyAsync(d_scalars, scalars, config.npoints * sizeof(*d_scalars), cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(status) << endl;
        throw cudaGetErrorString(status);
    }
}

/**
 * Result container
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
pippenger_t<bucket_t, point_t, scalar_t, affine_t>::result_container_t 
pippenger_t<bucket_t, point_t, scalar_t, affine_t>::result_container(pippenger_t &config) {
    result_container_t res(config.N);
    return res;
}

/***************************************** Function declerations for 'device_ptr' class  *****************************************/

/**
 * Allocate memory using cudaMalloc
 */
template <class T>
size_t device_ptr<T>::allocate(size_t bytes) {
    T* d_ptr;
    cudaError_t error = cudaMalloc(&d_ptr, bytes);
    if (error != cudaSuccess) {
        cout << "CUDA error from 'cudaMalloc': " << cudaGetErrorString(error) << endl;
        throw cudaErrorMemoryAllocation; 
    }
    d_ptrs.push_back(d_ptr);
    return d_ptrs.size() - 1;
}

/**
 * Get size of d_ptrs vector
 */
template <class T>
size_t device_ptr<T>::size() {
    return d_ptrs.size();
}

/**
 * Operator overloading for device_ptr
 */
template <class T>
T* device_ptr<T>::operator[](size_t i) {
    if (i > d_ptrs.size() - 1) {
        cout << "CUDA error from 'cudaGetDeviceProperties': " << cudaGetErrorString(cudaErrorInvalidDevicePointer) << endl;
        throw cudaGetErrorString(cudaErrorInvalidDevicePointer);
    }
    return d_ptrs[i];
}

}