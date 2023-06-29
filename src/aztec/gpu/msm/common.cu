#include "kernel.cu"
#include <iostream>
#include <vector>

namespace pippenger_common {

/**
 * Execute bucket method
 */ 
template <class point_t, class scalar_t>
point_t* pippenger_t<point_t, scalar_t>::execute_bucket_method(
pippenger_t &config, scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints, cudaStream_t stream) {
    unsigned num_bucket_modules = bitsize / c; 
    if (bitsize % c) {  
        num_bucket_modules++;
    }
    size_t num_buckets = num_bucket_modules << c; 

    // Bucket initialization kernel
    point_t *buckets;
    unsigned NUM_THREADS = 1 << 10; 
    unsigned NUM_BLOCKS = (num_buckets + NUM_THREADS - 1) / NUM_THREADS;
    CUDA_WRAPPER(cudaMallocAsync(&buckets, num_buckets * 3 * 4 * sizeof(uint64_t), stream));
    initialize_buckets_kernel<<<NUM_BLOCKS * 4, NUM_THREADS, 0, stream>>>(buckets); 

    // Split scalars kernel
    unsigned *bucket_indices;
    unsigned *point_indices;
    CUDA_WRAPPER(cudaMallocAsync(&bucket_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1), stream));
    CUDA_WRAPPER(cudaMallocAsync(&point_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1), stream));
    split_scalars_kernel<<<NUM_POINTS / NUM_THREADS, NUM_THREADS, 0, stream>>>
        (bucket_indices + npoints, point_indices + npoints, scalars, npoints, num_bucket_modules, c);

    // CUB routines for sorting indices using Radix sort algorithm
    unsigned *sort_indices_temp_storage{};
    size_t sort_indices_temp_storage_bytes; 
    cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + npoints, bucket_indices,
                                    point_indices + npoints, point_indices, npoints, 0, sizeof(unsigned) * 8, stream);
    CUDA_WRAPPER(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
    for (unsigned i = 0; i < num_bucket_modules; i++) {
        unsigned offset_out = i * npoints;
        unsigned offset_in = offset_out + npoints;
        cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
                                        bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, npoints, 
                                        0, sizeof(unsigned) * 8, stream);
    }
    // Determine size of buckets based on bucket indices
    unsigned *single_bucket_indices;
    unsigned *bucket_sizes;
    unsigned *nof_buckets_to_compute;
    CUDA_WRAPPER(cudaMallocAsync(&single_bucket_indices, sizeof(unsigned) * num_buckets, stream));
    CUDA_WRAPPER(cudaMallocAsync(&bucket_sizes, sizeof(unsigned) * num_buckets, stream));
    CUDA_WRAPPER(cudaMallocAsync(&nof_buckets_to_compute, sizeof(unsigned), stream));
   
    // Perform length encoding
    void *encode_temp_storage = NULL;
    size_t encode_temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                       nof_buckets_to_compute, num_bucket_modules * npoints, stream);
    CUDA_WRAPPER(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream));
    cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, 
                                       bucket_sizes, nof_buckets_to_compute, num_bucket_modules * npoints, stream);

    // Calculate offsets for buckets
    unsigned* bucket_offsets;
    CUDA_WRAPPER(cudaMallocAsync(&bucket_offsets, sizeof(unsigned) * num_buckets, stream));
    void *offsets_temp_storage = NULL;
    size_t offsets_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets, stream);
    CUDA_WRAPPER(cudaMallocAsync(&offsets_temp_storage, offsets_temp_storage_bytes, stream));
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets, stream);
    
    // Bucket accumulation kernel
    unsigned NUM_THREADS_3 = 1 << 7;
    unsigned NUM_BLOCKS_3 = ((num_buckets + NUM_THREADS_3 - 1) / NUM_THREADS_3) * 4;
    accumulate_buckets_kernel<<<NUM_BLOCKS_3, NUM_THREADS_3, 0, stream>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, points, num_buckets);

    // Sum reduction kernel
    point_t *final_sum;
    CUDA_WRAPPER(cudaMallocAsync(&final_sum, num_bucket_modules * 3 * 4 * sizeof(uint64_t), stream));
    bucket_module_sum_reduction_lernel_0<<<26, 4, 0, stream>>>(buckets, final_sum, c);

    // Final accumulation kernel
    point_t *res;
    CUDA_WRAPPER(cudaMallocManaged(&res, 3 * 4 * sizeof(uint64_t)));
    final_accumulation_kernel<<<1, 4, 0, stream>>>(final_sum, res, num_bucket_modules, c);
    
    // Synchronize stream
    cudaStreamSynchronize(stream);

    // Free memory
    cudaFreeAsync(points, stream);
    cudaFreeAsync(scalars, stream);
    cudaFreeAsync(buckets, stream);
    cudaFreeAsync(bucket_indices, stream);
    cudaFreeAsync(point_indices, stream);
    cudaFreeAsync(sort_indices_temp_storage, stream);
    cudaFreeAsync(single_bucket_indices, stream);
    cudaFreeAsync(bucket_sizes, stream);
    cudaFreeAsync(nof_buckets_to_compute, stream);
    cudaFreeAsync(encode_temp_storage, stream);
    cudaFreeAsync(bucket_offsets, stream);
    cudaFreeAsync(offsets_temp_storage, stream);
    cudaFreeAsync(final_sum, stream);

    return res;
}

/**
 * Calculate the amount of device storage required to store bases 
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::get_size_bases(pippenger_t &config) {
    return NUM_POINTS * sizeof(point_t);
}

/**
 * Calculate the amount of device storage required to store scalars 
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::get_size_scalars(pippenger_t &config) {
    // return config.n * sizeof(scalar_t);
    return NUM_POINTS * sizeof(scalar_t);
}

/**
 * Allocate device storage for bases
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::allocate_bases(pippenger_t &config) {
    device_base_ptrs.allocate(get_size_bases(config));
}

/**
 * Allocate device storage for scalars
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::allocate_scalars(pippenger_t &config) {
    device_scalar_ptrs.allocate(get_size_scalars(config));
}

/**
 * Transfer base points to GPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_bases_to_device(
pippenger_t &config, point_t *device_bases_ptrs, const point_t *points, cudaStream_t stream) {    
    CUDA_WRAPPER(cudaMemcpyAsync(device_bases_ptrs, points, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
}

/**
 * Transfer scalars to GPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_scalars_to_device(
pippenger_t &config, scalar_t *device_scalar_ptrs, fr *scalars, cudaStream_t stream) {
    CUDA_WRAPPER(cudaMemcpyAsync(device_scalar_ptrs, scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
}

/**
 * Print results
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::print_result(g1_gpu::element *result_1, g1_gpu::element **result_2) {
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_1[0].x.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_1[0].y.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_1[0].z.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_2[0][0].x.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_2[0][0].y.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_2[0][0].z.data[i]);
    }
}

/**
 * Allocate pinned memory using cudaMallocHost
 */
template <class T>
void device_ptr<T>::allocate(size_t bytes) {
    T* d_ptr;
    CUDA_WRAPPER(cudaMallocHost(&d_ptr, bytes));
    d_ptrs.push_back(d_ptr);
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

/**
 * Verify double-and-add and pippenger's bucket method results
 * move to common.cuh file
 */ 
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::verify_result(point_t *result_1, point_t **result_2) {
    var *result;
    // change to cudaMallocAsync()
    CUDA_WRAPPER(cudaMallocManaged(&result, LIMBS * sizeof(uint64_t)));
    comparator_kernel<<<1, 4>>>(result_1, result_2[0], result);
    cudaDeviceSynchronize();

    assert (result[0] == 1);
    assert (result[1] == 1);
    assert (result[2] == 1);
    assert (result[3] == 1);

    cout << "MSM Result Verified!" << endl;
}

}