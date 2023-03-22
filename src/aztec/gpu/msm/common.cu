#include "kernel.cu"
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

    cout << "NWINS is: " << NWINS << endl;
    cout << "config.N is: " << config.N << endl;

    // launch_coop(
    //     pippenger, 1, 10, stream, (affine_t*)d_points, config.npoints, 
    //     (const scalar_t*)d_scalars, d_buckets, d_none
    // );

    launch_coop(
        pippenger, dim3(NWINS, config.N), NTHREADS, stream, (affine_t*)d_points, config.npoints, 
        (const scalar_t*)d_scalars, d_buckets, d_none
    );
    cudaDeviceSynchronize();
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
void pippenger_t<bucket_t, point_t, scalar_t, affine_t>::initialize_buckets(
scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints) {
    // Calculate the number of windows 
    unsigned num_bucket_modules = bitsize / c; 
    if (bitsize % c) {  
        num_bucket_modules++;
    }

    point_t *buckets;
    size_t num_buckets = num_bucket_modules << c; // 1024 * 26
    // change memory size?
    cudaMallocManaged(&buckets, num_buckets * 3 * 4 * sizeof(uint64_t));

    // Launch bucket initialization kernel
    unsigned NUM_THREADS = 1 << 10;
    unsigned NUM_BLOCKS = (num_buckets + NUM_THREADS - 1) / NUM_THREADS;

    // Need to see if initializing the buckets is even neccessary? Let's do it at the beggining anyways
    initialize_buckets_kernel<<<NUM_BLOCKS * 4, NUM_THREADS>>>(buckets); // 1024 * 26 operating on a single limb.
    cudaDeviceSynchronize();

    cout << "b-bit scalar is: " << bitsize << endl;
    cout << "c-bit scalar is: " << c << endl;
    cout << "number of bucket modules is: " << num_bucket_modules << endl;
    cout << "number of bucket is: " << num_buckets << endl;
    cout << "number of blocks is: " << NUM_BLOCKS << endl;
    cout << "number of threads is: " << NUM_THREADS << endl;

    // cout << "bucket 0 is: " << buckets[26623].x.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[26623].x.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[26623].x.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[26623].x.data[3] << endl;

    // cout << "bucket 0 is: " << buckets[26623].y.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[26623].y.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[26623].y.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[26623].y.data[3] << endl;

    // cout << "bucket 0 is: " << buckets[26623].z.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[26623].z.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[26623].z.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[26623].z.data[3] << endl;

    // Allocate memory for bucket and point indices
    unsigned *bucket_indices;
    unsigned *point_indices;
    cudaMallocManaged(&bucket_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1));
    cudaMallocManaged(&point_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1));

    // Split scalars into digits
    // NUM_THREADS * NUM_BLOCKS = NUM_BUCKETS --> each thread splits a single scalar into num_modules digits, each of size c. 
    unsigned NUM_BLOCKS_2 = ((npoints * (num_bucket_modules + 1)) + NUM_THREADS - 1) / NUM_THREADS;

    cout << "NUM_BLOCKS_2 is: " << NUM_BLOCKS_2 << endl;

    cout << "scalar 0 is: " << scalars[0].data[0] << endl;
    cout << "scalar 0 is: " << scalars[0].data[1] << endl;
    cout << "scalar 0 is: " << scalars[0].data[2] << endl;
    cout << "scalar 0 is: " << scalars[0].data[3] << endl;
    cout << "points is: " << points[0].x.data[0] << endl;
    cout << "points is: " << points[0].x.data[1] << endl;
    cout << "points is: " << points[0].x.data[2] << endl;
    cout << "points is: " << points[0].x.data[3] << endl;

    // Split sub-scalars into their own seperate buckets
    // not sure why we're passing pointers at an offset of npoints
    // Would be able to do this 4 coooperaitve groups if c = 16...look into it
    // Why do we have 27 blocks here?? 27 * 1024 = 27648 -- maybe because of the offset?
    // i think I can reduce this to 26 blocks
    split_scalars_kernel<<<NUM_BLOCKS_2, NUM_THREADS>>>(bucket_indices + npoints, point_indices + npoints, scalars, npoints, num_bucket_modules, c);
    cudaDeviceSynchronize();

    // integrating CUB routines for things like offset calculations
    // sort indices from smallest to largest in order to group points that belong to same bucket
    unsigned *sort_indices_temp_storage{};
    size_t sort_indices_temp_storage_bytes; // need to figure out why this is 1, maybe a return value?

    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + npoints, bucket_indices,
                                 point_indices + npoints, point_indices, npoints);

    // cout << "sort_indices_temp_storage_bytes is: " << sort_indices_temp_storage_bytes << endl;

    cudaMallocManaged(&sort_indices_temp_storage, sort_indices_temp_storage_bytes);

    // perform the radix sort operation -- total number of sorts is num_bucket_modules. sorting arrays of bucket_indices and point_indices
    for (unsigned i = 0; i < num_bucket_modules; i++) {
        unsigned offset_out = i * npoints;
        unsigned offset_in = offset_out + npoints;
        // pffsets ensure each iteration operates on different set of points in input / output array
        // compressing inputs + offset --> offset in this method
        cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
                                    bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, npoints);
    }
    cudaFree(sort_indices_temp_storage);

    // we've eliminated the offset here
    // cout << "SORTED: " << endl;
    // for (int i = 0; i < 1024; i++) {
    //     cout << "bucket_indices: " << bucket_indices[i] << endl;
    // }
    // exit(0);

    // Next find the size of the buckets based on bucket_indices
    unsigned *single_bucket_indices;
    unsigned *bucket_sizes;
    unsigned *nof_buckets_to_compute;
    // change this from unifiedc memory to cudaMalloc
    cudaMallocManaged(&single_bucket_indices, sizeof(unsigned) * num_buckets);
    cudaMallocManaged(&bucket_sizes, sizeof(unsigned) * num_buckets);
    cudaMallocManaged(&nof_buckets_to_compute, sizeof(unsigned));
   
    void *encode_temp_storage = NULL;
    size_t encode_temp_storage_bytes = 0;

    // run length encoding computes a simple compressed representation of a sequence of input element
    // this returns the unique bucket #, number of buckets in each, and the total number of unique buckets
    cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                            nof_buckets_to_compute, num_bucket_modules * npoints);
    cudaMallocManaged(&encode_temp_storage, encode_temp_storage_bytes);
    cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                            nof_buckets_to_compute, num_bucket_modules * npoints);
    cudaFree(encode_temp_storage);

    cout << "nof_buckets_to_compute is: " << nof_buckets_to_compute[0] << endl;

    int sum = 0;
    for (int i = 0; i < nof_buckets_to_compute[0]; i++) {
        sum = sum + bucket_sizes[i];
    }

    cout << "sum is: " << sum << endl;

    // this is good
    // int temp = 0;
    // for (int i = 0; i < num_buckets; i++) {
    //     cout << "count is: " << temp << endl;
    //     cout << "single_bucket_indices: " << single_bucket_indices[i] << endl;
    //     temp++;
    // }
    // exit(0);

    // for (int i = 0; i < num_buckets; i++) {
    //     cout << "index is: " << i << endl;
    //     cout << "bucket_sizes: " << bucket_sizes[i] << endl;
    // }
    // exit(0);

    //get offsets - where does each new bucket begin
    unsigned* bucket_offsets;
    cudaMallocManaged(&bucket_offsets, sizeof(unsigned) * num_buckets);
    unsigned* offsets_temp_storage{};
    size_t offsets_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets);
    cudaMallocManaged(&offsets_temp_storage, offsets_temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets);
    cudaFree(offsets_temp_storage);

    cout << "num_buckets is: " << num_buckets << endl;

    // cout << "bucekt size is: " << bucket_sizes[5] << endl;

    // This is correct
    // for (int i = 0; i < num_buckets; i++) {
    //     cout << "index is: " << i << endl;
    //     cout << "bucket size is: " << bucket_sizes[i] << endl;
    // }

    // exit(0);

    // Launch accumulation kernel
    // 512 and 208 respectively
    //Need to adjusge kernel para,meters to reduce overhead
    unsigned NUM_THREADS_3 = 1 << 8;
    unsigned NUM_BLOCKS_3 = ((num_buckets + NUM_THREADS_3 - 1) / NUM_THREADS_3);

    cout << "NUM_THREADS_3 is: " << NUM_THREADS_3 << endl;
    cout << "NUM_BLOCKS_3 is: " << NUM_BLOCKS_3 << endl;

    // The kernel launch parameters need to be changed!
    // What’s the primary reason why the max occupancy (threads + blocks) of a kernel is much lower than the 
    // theoretical occupancy (# SMs * max threads / SM)? I have a complex kernel function that exceeds all the 
    // registers past 512 threads / block. I have to lower the thread count per block and launch more blocks as a 
    // result. But i’m hitting an upper-block limit for this kernel. Is the only solution here splitting up the problem
    // between multiple seperate kernel launches? And if so, what’s the best approach? The naive solution is moving from
    // single P100 to A10.
    accumulate_buckets_kernel<<<50, 128>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, points, num_buckets);
    cudaDeviceSynchronize();

    int count = 0;
    for (int i = 0; i < num_buckets; i++) {
        if (buckets[i].x.data[0] != 0) {
            count++;
        }
        // cout << "bucket index is: " << i << endl;
        // cout << "x: " << "{ " << buckets[i].x.data[0] << "," << buckets[i].x.data[1] << "," << buckets[i].x.data[2] << "," << buckets[i].x.data[3] << " }; "  
        // << " y: " << "{ " << buckets[i].y.data[0] << "," << buckets[i].y.data[1] << "," << buckets[i].y.data[2] << "," << buckets[i].y.data[3] << " }; "  
        // << " z: " << "{ " << buckets[i].z.data[0] << "," << buckets[i].z.data[1] << "," << buckets[i].z.data[2] << "," << buckets[i].z.data[3] << " }; " << endl;
    }
    cout << "count is: " << count << endl;

    // The result is we'll have num_bucket modules each with num_bucket << c buckets of a single result
    cout << "final_result is: " << buckets[0].x.data[0] << endl;
    cout << "final_result is: " << buckets[0].x.data[1] << endl;
    cout << "final_result is: " << buckets[0].x.data[2] << endl;
    cout << "final_result is: " << buckets[0].x.data[3] << endl;

    cudaSetDevice(0);
    size_t free_device_mem = 0;
    size_t total_device_mem = 0;
    cudaMemGetInfo(&free_device_mem, &total_device_mem);
    printf("Currently available amount of device memory: %zu bytes\n", free_device_mem);
    printf("Total amount of device memory: %zu bytes\n", total_device_mem);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    

    // At this point we have n buckets and m bucket modules. Need to first sum up the n buckets per bucket module, and then
    // perform a final accumulation of the bucket modules. Launch 4 threads per bucket module.

    // point_t *final_result;
    // cudaMallocManaged(&final_result, num_bucket_modules * 3 * 4 * sizeof(uint64_t));
    // bucket_module_sum_reduction_kernel<<<1, num_bucket_modules * 4>>>(buckets, final_result, num_buckets, c);
    // cudaDeviceSynchronize();

    // cout << "\nfinal_result is: " << final_result[0].x.data[0] << endl;
    // cout << "final_result is: " << final_result[0].x.data[1] << endl;
    // cout << "final_result is: " << final_result[0].x.data[2] << endl;
    // cout << "final_result is: " << final_result[0].x.data[3] << endl;

    // // Final accumulation kernel
    // point_t *res;
    // cudaMallocManaged(&res, 3 * 4 * sizeof(uint64_t));
    // final_accumulation_kernel<<<1,4>>>(final_result, res, num_bucket_modules, c);
    // cudaDeviceSynchronize();

    // cout << "\n res is: " << res[0].x.data[0] << endl;
    // cout << "res is: " << res[0].x.data[1] << endl;
    // cout << "res is: " << res[0].x.data[2] << endl;
    // cout << "res is: " << res[0].x.data[3] << endl;

    // cudamemcy
    // free memory
}

// number of registers available on the multiprocessor is being exceeded. Reduce the number of threads per block to solve the problem

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