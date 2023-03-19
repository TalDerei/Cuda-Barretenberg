#include "kernel.cu"
#include <cuda.h>
// #include <cub/device/device_radix_sort.cuh>
// #include <cub/device/device_run_length_encode.cuh>
// #include <cub/device/device_scan.cuh>
#include <cub/cub.cuh>

#include <iostream>
#include <vector>

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
    return config.n * sizeof(affine_t);
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
    affine_t *d_points = device_base_ptrs[d_points_idx];

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

/**
 * Launch kernel
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::launch_kernel(
pippenger_t &config, size_t d_bases_idx, size_t d_scalar_idx, size_t d_buckets_idx) {
    // Set default stream
    cudaStream_t stream = config.default_stream;

    // Pointers to malloced memory locations
    affine_t *d_points = device_base_ptrs[d_bases_idx];
    scalar_t *d_scalars = device_scalar_ptrs[d_scalar_idx];

    // Two-dimensional array of pointers to 'bucket_t' values with NWINS slices, each slice containing 1<<WBITS bucket_t pointers
    bucket_t (*d_buckets)[NWINS][1<<WBITS] = reinterpret_cast<decltype(d_buckets)>(device_bucket_ptrs[d_buckets_idx]);
    bucket_t (*d_none)[NWINS][NTHREADS][2] = nullptr;
    
    CUDA_WRAPPER(cudaSetDevice(config.device));

    // Helper function that triggers the kernel launch
    launch_coop(
        pippenger, dim3(NWINS, config.N), NTHREADS, stream, (affine_t*)d_points, config.npoints, 
        (const scalar_t*)d_scalars, d_buckets, d_none
    );
}

/**
 * Read affine elliptic curve points from SRS
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
affine_t* pippenger_t<bucket_t, point_t, scalar_t, affine_t>::read_affine_curve_points() {
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();

    return points;
}

/**
 * Read jacobian elliptic curve points from file
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
point_t* pippenger_t<bucket_t, point_t, scalar_t, affine_t>::read_jacobian_curve_points(point_t *points) {
    std::ifstream myfile ("../src/aztec/gpu/benchmark/tests/msm/points/curve_points.txt"); 

    if ( myfile.is_open() ) {   
        for (size_t i = 0; i < NUM_POINTS; i++) {
            for (size_t j = 0; j < 4; j++) {
                myfile >> points[i].x.data[j];
            }
            for (size_t y = 0; y < 4; y++) {
                myfile >> points[i].y.data[y];
            }
            for (size_t z = 0; z < 4; z++) {
                myfile >> points[i].z.data[z];
            }
        }   
    }
    myfile.close();
    
    return points;
} 

/**
 * Read scalars from scalar field
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
scalar_t* pippenger_t<bucket_t, point_t, scalar_t, affine_t>::read_scalars(scalar_t *scalars) {
    ifstream stream;
    stream.open("../src/aztec/gpu/msm/points/scalars.txt", ios::in);

    if ( stream.is_open() ) {   
        for (size_t i = 0; i < NUM_POINTS; i++) {
            for (size_t j = 0; j < 4; j++) {
                stream >> scalars[i].data[j];
            }
        }   
    }
    stream.close();
        
    return scalars;
}

/**
 * Print results
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::print_result(point_t *result) {
    for (int i = 0; i < LIMBS; i++) {
        printf("result is: %zu\n", result[0].x.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result is: %zu\n", result[0].y.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result is: %zu\n", result[0].z.data[i]);
    }
    printf("\n");
}

/**
 * Initialze buckets
 */
template <class bucket_t, class point_t, class scalar_t, class affine_t>
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::initialize_buckets(scalar_t *scalars, unsigned bitsize, unsigned c, size_t npoints) {
    unsigned num_bucket_modules = bitsize / c; 
    if (bitsize / c) {  
        num_bucket_modules++;
    }

    size_t num_buckets = num_bucket_modules << c;

    point_t *buckets;
    cudaMallocManaged(&buckets, num_buckets * sizeof(point_t));

    // Launch bucket initialization kernel
    unsigned NUM_THREADS = 1 << 10;
    unsigned NUM_BLOCKS = (num_buckets + NUM_THREADS - 1) / NUM_THREADS;

    initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets);
    cudaDeviceSynchronize();

    cout << "b-bit scalar is: " << bitsize << endl;
    cout << "c-bit scalar is: " << c << endl;
    cout << "number of bucket modules is: " << num_bucket_modules << endl;
    cout << "number of bucket is: " << num_buckets << endl;
    cout << "number of blocks is: " << NUM_BLOCKS << endl;
    cout << "number of threads is: " << NUM_THREADS << endl;

    // cout << "bucket 0 is: " << buckets[0].x.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[0].x.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[0].x.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[0].x.data[3] << endl;

    // cout << "bucket 0 is: " << buckets[0].y.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[0].y.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[0].y.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[0].y.data[3] << endl;

    // cout << "bucket 0 is: " << buckets[0].z.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[0].z.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[0].z.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[0].z.data[3] << endl;

    // Allocate memory for bucket and point indices
    unsigned *bucket_indices;
    unsigned *point_indices;
    cudaMallocManaged(&bucket_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1));
    cudaMallocManaged(&point_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1));

    // Split scalars into digits
    unsigned NUM_BLOCKS_2 = ((npoints * (num_bucket_modules + 1)) + NUM_THREADS - 1) / NUM_THREADS;
    
    // cout << "NUM_BLOCKS is: " << NUM_BLOCKS_2 << endl;

    // NUM_THREADS * NUM_BLOCKS = NUM_BUCKETS --> each thread splits a single scalar into num_modules digits, each of size c. 
    // I'm splitting the work between 4 threads, and need the window size to be an even multiple of 4?

    cout << "scalar 0 is: " << scalars[0].data[0] << endl;
    cout << "scalar 0 is: " << scalars[0].data[1] << endl;
    cout << "scalar 0 is: " << scalars[0].data[2] << endl;
    cout << "scalar 0 is: " << scalars[0].data[3] << endl;

    split_scalars_kernel<<<1, 1>>>(bucket_indices + npoints, point_indices + npoints, scalars, npoints, num_bucket_modules, c);
    cudaDeviceSynchronize();

    // integrating CUB routines for things like offset calculations
    // sort indices from smallest to largest in order to group points that belong to same bucket
    // unsigned *sort_indices_temp_storage{};
    // size_t sort_indices_temp_storage_bytes;

    // determine the amount of temporary storage
    // cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + npoints, bucket_indices,
    //                              point_indices + npoints, point_indices, npoints);

    // cudaMalloc(&sort_indices_temp_storage, sort_indices_temp_storage_bytes);

    // // perform the radix sort -- total number of sorts is num_bucket_modules. sorting arrays of bucket_indices and point_indices
    // for (unsigned i = 0; i < num_bucket_modules; i++) {
    //     unsigned offset_out = i * npoints;
    //     unsigned offset_in = offset_out + npoints;
    //     cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
    //                                 bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, npoints);
    // }
    // cudaFree(sort_indices_temp_storage);
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