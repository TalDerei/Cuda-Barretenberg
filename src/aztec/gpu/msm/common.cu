#include "kernel.cu"

using namespace std;

namespace pippenger_common {
/***************************************** Function declerations for 'pippenger_t' class  *****************************************/

/**
 * Initialize cuda device and MSM parameters
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
pippenger_t<bucket_t, point_t, scalar_t, affine_t> 
pippenger_t<bucket_t, point_t, scalar_t, affine_t>::initialize_msm(pippenger_t &config, size_t npoints) {
    CUDA_WRAPPER(cudaSetDevice(config.device));

    cudaDeviceProp prop;
    CUDA_WRAPPER(cudaGetDeviceProperties(&prop, 0));

    // Set streaming multiprocessor count, where each SM contains N cuda cores
    sm_count = prop.multiProcessorCount;
    
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
    CUDA_WRAPPER(cudaSetDevice(config.device));

    cudaStream_t stream = config.default_stream;

    // change to affine_t, along with device_base_ptrs
    point_t *d_points = device_base_ptrs[d_points_idx];

    // cudaMemcpyAsync() is non-blocking and asynchronous variant of cudaMemcpy() that requires pinned memory.
    // Asynchronous transfers enable overalapping data transfers with kernel execution.
    CUDA_WRAPPER(cudaMemcpyAsync(d_points, points, config.npoints * sizeof(*d_points), cudaMemcpyHostToDevice, default_stream));
}

/**
 * Transfer scalars to GPU device
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::transfer_scalars_to_device(
pippenger_t &config, size_t d_scalars_idx, const scalar_t scalars[], cudaStream_t aux_stream = nullptr) {
    // Set cuda device and auxilary stream
    cudaSetDevice(config.device);
    cudaStream_t stream = aux_stream;

    scalar_t *d_scalars = device_scalar_ptrs[d_scalars_idx];
    
    CUDA_WRAPPER(cudaMemcpyAsync(d_scalars, scalars, config.npoints * sizeof(*d_scalars), cudaMemcpyHostToDevice, stream));
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

/**
 * Synchronize stream
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::synchronize_stream(pippenger_t &config) {
    CUDA_WRAPPER(cudaSetDevice(config.device));
    CUDA_WRAPPER(cudaStreamSynchronize(config.default_stream));
}

/**
 * Helper function  
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
template<typename... Types>
inline void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::launch_coop(
void(*f)(Types...), dim3 gridDim, dim3 blockDim, cudaStream_t stream, Types... args) {
    void* va_args[sizeof...(args)] = { &args... };

    CUDA_WRAPPER(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim, va_args, 0, stream));
}

template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::launch_kernel(
pippenger_t &config, size_t d_bases_idx, size_t d_scalar_idx, size_t d_buckets_idx) {
    // Set default stream
    cudaStream_t stream = config.default_stream;

    // Pointers to malloced memory locations
    point_t *d_points = device_base_ptrs[d_bases_idx];
    scalar_t *d_scalars = device_scalar_ptrs[d_scalar_idx];

    // Two-dimensional array of pointers to 'bucket_t' values with NWINS slices, each slice containing 1<<WBITS bucket_t pointers
    bucket_t (*d_buckets)[NWINS][1<<WBITS] = reinterpret_cast<decltype(d_buckets)>(device_bucket_ptrs[d_buckets_idx]);
    bucket_t (*d_none)[NWINS][NTHREADS][2] = nullptr;
    
    CUDA_WRAPPER(cudaSetDevice(config.device));

    // Helper function that triggers the kernel launch
    launch_coop(
        pippenger, dim3(NWINS, config.N), NTHREADS, stream, (const point_t*)d_points, config.npoints, 
        (const scalar_t*)d_scalars, d_buckets, d_none
    );
}

/***************************************** Function declerations for 'device_ptr' class  *****************************************/

/**
 * Allocate memory using cudaMalloc
 */
template <class T>
size_t device_ptr<T>::allocate(size_t bytes) {
    T* d_ptr;
    CUDA_WRAPPER(cudaMalloc(&d_ptr, bytes));

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
        cout << "Indexing error!" << endl;
        throw;
    }
    return d_ptrs[i];
}

}