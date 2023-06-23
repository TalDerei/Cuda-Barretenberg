#include "kernel.cu"
#include <iostream>
#include <vector>

namespace pippenger_common {

/**
 * Execute bucket method
 */ 
template <class point_t, class scalar_t>
point_t* pippenger_t<point_t, scalar_t>::execute_bucket_method(
pippenger_t &config, scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints) {
    unsigned num_bucket_modules = bitsize / c; 
    if (bitsize % c) {  
        num_bucket_modules++;
    }
    size_t num_buckets = num_bucket_modules << c; 

    point_t *buckets;
    CUDA_WRAPPER(cudaMallocManaged(&buckets, num_buckets * 3 * 4 * sizeof(uint64_t)));

    // Launch bucket initialization kernel
    unsigned NUM_THREADS = 1 << 10; // max number of threads
    unsigned NUM_BLOCKS = (num_buckets + NUM_THREADS - 1) / NUM_THREADS; // max number of blocks
    initialize_buckets_kernel<<<NUM_BLOCKS * 4, NUM_THREADS>>>(buckets); 

    cout << "b-bit scalar is: " << bitsize << endl;
    cout << "c-bit scalar is: " << c << endl;
    cout << "number of bucket modules is: " << num_bucket_modules << endl;
    cout << "number of bucket is: " << num_buckets << endl;
    cout << "number of blocks is: " << NUM_BLOCKS << endl;
    cout << "number of threads is: " << NUM_THREADS << endl;

    // Allocate memory for bucket and point indices
    unsigned *bucket_indices;
    unsigned *point_indices;
    CUDA_WRAPPER(cudaMallocManaged(&bucket_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1)));
    CUDA_WRAPPER(cudaMallocManaged(&point_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1)));

    // Split scalars into digits
    unsigned NUM_BLOCKS_2 = NUM_POINTS / NUM_THREADS;

    cout << "npoints is: " << npoints << endl;
    cout << "NUM_THREADS is: " << NUM_THREADS << endl;
    cout << "NUM_BLOCKS_2 is: " << NUM_BLOCKS_2 << endl;

    split_scalars_kernel<<<NUM_BLOCKS_2, NUM_THREADS>>>(bucket_indices + npoints, point_indices + npoints, scalars, npoints, num_bucket_modules, c);
    cudaDeviceSynchronize();

    // CUB routines for Radix Sort
    unsigned *sort_indices_temp_storage{};
    size_t sort_indices_temp_storage_bytes; 

    // Determine temporary device storage requirements
    cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + npoints, bucket_indices,
                                    point_indices + npoints, point_indices, npoints);

    CUDA_WRAPPER(cudaMallocManaged(&sort_indices_temp_storage, sort_indices_temp_storage_bytes));

    for (unsigned i = 0; i < num_bucket_modules; i++) {
        unsigned offset_out = i * npoints;
        unsigned offset_in = offset_out + npoints;
        cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
                                        bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, npoints);
    }

    // Next find the size of the buckets based on bucket_indices
    unsigned *single_bucket_indices;
    unsigned *bucket_sizes;
    unsigned *nof_buckets_to_compute;
    CUDA_WRAPPER(cudaMallocManaged(&single_bucket_indices, sizeof(unsigned) * num_buckets));
    CUDA_WRAPPER(cudaMallocManaged(&bucket_sizes, sizeof(unsigned) * num_buckets));
    CUDA_WRAPPER(cudaMallocManaged(&nof_buckets_to_compute, sizeof(unsigned)));
   
    void *encode_temp_storage = NULL;
    size_t encode_temp_storage_bytes = 0;

    // Perform length encoding
    cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, num_bucket_modules * npoints);
    CUDA_WRAPPER(cudaMallocManaged(&encode_temp_storage, encode_temp_storage_bytes));
    cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, num_bucket_modules * npoints);


    cout << "nof_buckets_to_compute is: " << nof_buckets_to_compute[0] << endl;

    // Calculate offsets for buckets
    unsigned* bucket_offsets;
    CUDA_WRAPPER(cudaMallocManaged(&bucket_offsets, sizeof(unsigned) * num_buckets));
    void *offsets_temp_storage = NULL;
    size_t offsets_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets);
    CUDA_WRAPPER(cudaMallocManaged(&offsets_temp_storage, offsets_temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets);
    CUDA_WRAPPER(cudaFree(offsets_temp_storage));
    
    // Launch accumulation kernel -- Need to adjusge kernel para,meters to reduce overhead
    unsigned NUM_THREADS_3 = 1 << 7;
    unsigned NUM_BLOCKS_3 = ((num_buckets + NUM_THREADS_3 - 1) / NUM_THREADS_3) * 4;

    cout << "NUM_THREADS_3 is: " << NUM_THREADS_3 << endl;
    cout << "NUM_BLOCKS_3 is: " << NUM_BLOCKS_3 << endl;

    accumulate_buckets_kernel<<<NUM_BLOCKS_3, NUM_THREADS_3>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, points, num_buckets);
    cudaDeviceSynchronize();

    point_t *final_sum;
    CUDA_WRAPPER(cudaMallocManaged(&final_sum, num_bucket_modules * 3 * 4 * sizeof(uint64_t)));
    bucket_module_sum_reduction_lernel_0<<<26, 4>>>(buckets, final_sum, c);
    cudaDeviceSynchronize();

    point_t *res;
    CUDA_WRAPPER(cudaMallocManaged(&res, 3 * 4 * sizeof(uint64_t)));
    final_accumulation_kernel<<<1,4>>>(final_sum, res, num_bucket_modules, c);
    cudaDeviceSynchronize();
    
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
void pippenger_t<point_t, scalar_t>::print_result(g1_gpu::element *result_naive_msm, g1_gpu::element *result_bucket_method_msm) {
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_naive_msm[0].x.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_naive_msm[0].y.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_naive_msm is: %zu\n", result_naive_msm[0].z.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_bucket_method_msm[0].x.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_bucket_method_msm[0].y.data[i]);
    }
    printf("\n");
    for (int i = 0; i < LIMBS; i++) {
        printf("result_bucket_method_msm is: %zu\n", result_bucket_method_msm[0].z.data[i]);
    }
}

/**
 * Allocate memory using cudaMallocHost
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
void pippenger_t<point_t, scalar_t>::verify_result(point_t *result_1, point_t *result_2) {
    var *result;
    CUDA_WRAPPER(cudaMallocManaged(&result, LIMBS * sizeof(uint64_t)));
    comparator_kernel<<<1, 4>>>(result_1, result_2, result);
    cudaDeviceSynchronize();

    assert (result[0] == 1);
    assert (result[1] == 1);
    assert (result[2] == 1);
    assert (result[3] == 1);

    cout << "MSM Result Verified!" << endl;
}

}