#include "kernel.cu"
#include <iostream>
#include <vector>

namespace pippenger_common {
/***************************************** Function declerations for 'pippenger_t' class  *****************************************/

/**
 * Initialize cuda device and MSM parameters
 */
template <class point_t, class scalar_t>
pippenger_t<point_t, scalar_t> 
pippenger_t<point_t, scalar_t>::initialize_msm(pippenger_t &config, size_t npoints) {
    CUDA_WRAPPER(cudaSetDevice(config.device));
    config.n = npoints;

    return config;
}

/**
 * Calculate the amount of device storage required to store bases 
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::get_size_bases(pippenger_t &config) {
    return config.n * sizeof(point_t);
}

/**
 * Calculate the amount of device storage required to store scalars 
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::get_size_scalars(pippenger_t &config) {
    // return config.n * sizeof(scalar_t);
    return config.n * LIMBS * sizeof(uint64_t);
}

/**
 * Calculate the amount of device storage required to store buckets 
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::get_size_buckets(pippenger_t &config) { 
    return (MODULES << C) * 3 * 4 * sizeof(uint64_t);
}

/**
 * Allocate device storage for bases
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::allocate_bases(pippenger_t &config) {
    return device_base_ptrs.allocate(get_size_bases(config));
}

/**
 * Allocate device storage for scalars
 */
template <class point_t, class scalar_t>
size_t pippenger_t<point_t, scalar_t>::allocate_scalars(pippenger_t &config) {
    return device_scalar_ptrs.allocate(get_size_scalars(config));
}

/**
 * Transfer base points to GPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_bases_to_device(
pippenger_t &config, point_t *device_bases_ptrs, const point_t *points, cudaStream_t aux_stream = nullptr) {    
    // Set cuda device and auxilary stream
    cudaStream_t stream = (aux_stream == nullptr) ? default_stream : aux_stream;
    cudaSetDevice(config.device);    

    // cudaMemcpyAsync() is non-blocking and asynchronous variant of cudaMemcpy() that requires pinned memory.
    cudaMemcpyAsync(device_bases_ptrs, points, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
}

/**
 * Transfer scalars to GPU device
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::transfer_scalars_to_device(
pippenger_t &config, scalar_t *device_scalar_ptrs, fr *scalars, cudaStream_t aux_stream = nullptr) {
    // Set cuda device and auxilary stream
    cudaStream_t stream = (aux_stream == nullptr) ? default_stream : aux_stream;
    cudaSetDevice(config.device);    

    cudaMemcpyAsync(device_scalar_ptrs, scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
}

/**
 * Synchronize stream
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::synchronize_stream(pippenger_t &config) {
    CUDA_WRAPPER(cudaSetDevice(config.device));
    CUDA_WRAPPER(cudaStreamSynchronize(config.default_stream));
}

/**
 * Launch kernel
 */
// template <class point_t, class scalar_t>
// void pippenger_t<point_t, scalar_t>::launch_kernel(
// pippenger_t &config, size_t d_bases_idx, size_t d_scalar_idx, size_t d_buckets_idx) {
//     // Set default stream
//     cudaStream_t stream = config.default_stream;

//     // Pointers to malloced memory locations
//     point_t *d_points = device_base_ptrs[d_bases_idx];
//     scalar_t *d_scalars = device_scalar_ptrs[d_scalar_idx];
    
//     CUDA_WRAPPER(cudaSetDevice(config.device));

//     g1_gpu::element *final_result;
//     cudaMallocManaged(&final_result, NUM_POINTS * LIMBS * sizeof(uint64_t));    

//     cudaDeviceSynchronize();

//     cout << "\nfinal_result is: " << final_result[0].x.data[0] << endl;
//     cout << "final_result is: " << final_result[0].x.data[1] << endl;
//     cout << "final_result is: " << final_result[0].x.data[2] << endl;
//     cout << "final_result is: " << final_result[0].x.data[3] << endl;
// }

/**
 * Print results
 */
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::print_result(g1_gpu::element *result_naive_msm, g1_gpu::element *result_bucket_method_msm) {
    for (int i = 0; i < 3; i++) {
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
    for (int i = 0; i < 3; i++) {
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
 * Verify double-and-add and pippenger's bucket method results
 * move to common.cuh file
 */ 
template <class point_t, class scalar_t>
void pippenger_t<point_t, scalar_t>::verify_result(point_t *result_1, point_t *result_2) {
    var *result;
    cudaMallocManaged(&result, LIMBS * sizeof(uint64_t));
    
    comparator_kernel<<<1, 4>>>(result_1, result_2, result);
    cudaDeviceSynchronize();

    assert (result[0] == 1);
    assert (result[1] == 1);
    assert (result[2] == 1);
    assert (result[3] == 1);

    cout << "MSM Result Verified!" << endl;
}

// /**
//  * Execute bucket method 
//  */
template <class point_t, class scalar_t>
point_t* pippenger_t<point_t, scalar_t>::execute_bucket_method(
pippenger_t &config, scalar_t *scalars, point_t *points, unsigned bitsize, unsigned c, size_t npoints) {
    unsigned num_bucket_modules = bitsize / c; 
    if (bitsize % c) {  
        num_bucket_modules++;
    }
    size_t num_buckets = num_bucket_modules << c; // 1024 * 26 or 65536 * 16 = 1M buckets


    point_t *buckets;
    // change memory size?
    cudaMallocManaged(&buckets, num_buckets * 3 * 4 * sizeof(uint64_t));

    // Launch bucket initialization kernel
    unsigned NUM_THREADS = 1 << 10; // max number of threads
    unsigned NUM_BLOCKS = (num_buckets + NUM_THREADS - 1) / NUM_THREADS; // max number of blocks

    // where blocks * threads = total numbrt of buckets

    // Need to see if initializing the buckets is even neccessary? Let's do it at the beggining anyways for now...
    // and later incorperate it into the kernel with "pipelining loads and compute" as depaul put it.
    // Each thread operates on a single scalar of size num_bucket_modules windows,
    // But there's a problem here. There's a mismatch between the number of threads being launched to represent the 
    // number of buckets, and the total number of scalars that need to be partitioned into these buckets. I think the
    // buckets array needs to be a 2D array to account for this? Will change after...
    // It will need to support # scalars * 16, which may exceed the bucket count. Therefore the total number of buckets will
    // stay the same, but the indexing will change...2D instead.
    initialize_buckets_kernel<<<NUM_BLOCKS * 4, NUM_THREADS>>>(buckets); 
    // cudaDeviceSynchronize();

    // cudaMallocManaged(&buckets, num_buckets * 3 * 4 * sizeof(uint64_t));
    // // convert affine to jacobian coordinates
    // af<<<NUM_BLOCKS * 4, NUM_THREADS>>>(points, points); 

    cout << "b-bit scalar is: " << bitsize << endl;
    cout << "c-bit scalar is: " << c << endl;
    cout << "number of bucket modules is: " << num_bucket_modules << endl;
    cout << "number of bucket is: " << num_buckets << endl;
    cout << "number of blocks is: " << NUM_BLOCKS << endl;
    cout << "number of threads is: " << NUM_THREADS << endl;
    
    // cout << "bucket 0 is: " << buckets[1048575].y.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].x.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].x.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].x.data[3] << endl;

    // cout << "bucket 0 is: " << buckets[1048575].y.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].y.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].y.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].y.data[3] << endl;

    // cout << "bucket 0 is: " << buckets[1048575].z.data[0] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].z.data[1] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].z.data[2] << endl;
    // cout << "bucket 0 is: " << buckets[1048575].z.data[3] << endl;

    // After here, we have 2^16 windows, with each scalar represented by 16 bits. Each window will have 2^16 buckets,
    // and we'll have 2^16 * 16 total buckets. 

    // Allocate memory for bucket and point indices
    // need to understand these launch parameters
    unsigned *bucket_indices;
    unsigned *point_indices;
    cudaMallocManaged(&bucket_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1));
    cudaMallocManaged(&point_indices, sizeof(unsigned) * npoints * (num_bucket_modules + 1));

    // Split scalars into digits
    // NUM_THREADS * NUM_BLOCKS = NUM_BUCKETS --> each thread splits a single scalar into num_modules digits, each of size c. 
    // unsigned NUM_BLOCKS_2 = ((npoints * (num_bucket_modules + 1)) + NUM_THREADS - 1) / NUM_THREADS;
    // unsigned NUM_BLOCKS_2 = ((num_buckets + NUM_THREADS - 1) / NUM_THREADS);
    unsigned NUM_BLOCKS_2 = NUM_POINTS / NUM_THREADS;
    

    cout << "npoints is: " << npoints << endl;
    cout << "NUM_THREADS is: " << NUM_THREADS << endl;
    cout << "NUM_BLOCKS_2 is: " << NUM_BLOCKS_2 << endl;

    // cout << "scalar 0 is: " << scalars[0].data[0] << endl;
    // cout << "scalar 0 is: " << scalars[0].data[1] << endl;
    // cout << "scalar 0 is: " << scalars[0].data[2] << endl;
    // cout << "scalar 0 is: " << scalars[0].data[3] << endl;
    // cout << "points is: " << points[0].x.data[0] << endl;
    // cout << "points is: " << points[0].x.data[1] << endl;
    // cout << "points is: " << points[0].x.data[2] << endl;
    // cout << "points is: " << points[0].x.data[3] << endl;

    // Split sub-scalars into their own seperate buckets
    // not sure why we're passing pointers at an offset of npoints
    // Would be able to do this 4 coooperaitve groups if c = 16...look into it
    // Why do we have 27 blocks here?? 27 * 1024 = 27648 -- maybe because of the offset?
    // i think I can reduce this to 26 blocks

    // 1024 * 17 = 17408 threads to split scalars, but have more than 17408 scalars?
    // What if you have more scalars than buckets, which is the case here. Previously, there were less
    // scalars (1024) than the total number of buckets (27648).
    
    // Each thread will handle splitting it's own scalar into sub-scalars, and placing them into buckets. 
    // split_scalars_kernel<<<NUM_BLOCKS_2, NUM_THREADS>>>(bucket_indices + npoints, point_indices + npoints, scalars, npoints, num_bucket_modules, c);
    // ********* This scalar value for launch parameters will also need to be changed instead of being hardcoded! ******************
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
    // cudaFree(sort_indices_temp_storage);

    // we've eliminated the offset here
    // cout << "SORTED: " << endl;
    // for (int i = 0; i < 1024; i++) {
    //     cout << "index is: " << i << endl;
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

    // int sum = 0;
    // for (int i = 0; i < nof_buckets_to_compute[0]; i++) {
    //     sum = sum + bucket_sizes[i];
    // }

    // cout << "sum is: " << sum << endl;

    // int temp = 0;
    // for (int i = 0; i < nof_buckets_to_compute[0]; i++) {
    //     cout << "count is: " << temp << endl;
    //     cout << "single_bucket_indices: " << single_bucket_indices[i] << endl;
    //     temp++;
    // }

    // int temp = 0;
    // for (int i = 0; i < num_buckets; i++) {
    //     if (single_bucket_indices[i] != 0) {
    //         temp++;
    //     }
    // }
    // cout << "temp is: " << temp << endl;

    // for (int i = 0; i < num_buckets; i++) {
    //     if (bucket_sizes[i] == 2) {
    //         cout << "!!!!!!!!!\n" << endl;
    //     } 
    // }

    //get offsets - where does each new bucket begin
    unsigned* bucket_offsets;
    cudaMallocManaged(&bucket_offsets, sizeof(unsigned) * num_buckets);
    // unsigned* offsets_temp_storage{};
    void *offsets_temp_storage = NULL;
    size_t offsets_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets);
    cudaMallocManaged(&offsets_temp_storage, offsets_temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, num_buckets);
    cudaFree(offsets_temp_storage);

    // cout << "num_buckets is: " << num_buckets << endl;

    // cout << "bucekt size is: " << bucket_sizes[5] << endl;

    // This is correct
    // for (int i = 0; i < num_buckets; i++) {
    //     if (bucket_sizes[i] != 0) {
    //         cout << "index is: " << i << endl;
    //         cout << "bucket size is: " << bucket_sizes[i] << endl;
    //     }
    // }

    // exit(0);

    // Launch accumulation kernel
    // 512 and 208 respectively
    //Need to adjusge kernel para,meters to reduce overhead
    unsigned NUM_THREADS_3 = 1 << 7;
    unsigned NUM_BLOCKS_3 = ((num_buckets + NUM_THREADS_3 - 1) / NUM_THREADS_3) * 4;

    cout << "NUM_THREADS_3 is: " << NUM_THREADS_3 << endl;
    cout << "NUM_BLOCKS_3 is: " << NUM_BLOCKS_3 << endl;
    
    //  // CUDA Event API
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);

    // Calculate maximum occupancy
    // int THREADS;
    // int BLOCKS;
    // cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, accumulate_buckets_kernel, 0, 0);
    // cout << "NUM_POINTS is: " << NUM_POINTS << endl;
    // cout << "max threads is: " << THREADS << endl;
    // cout << "max blocks is: " << BLOCKS << endl;

    // The kernel launch parameters need to be changed!
    // What’s the primary reason why the max occupancy (threads + blocks) of a kernel is much lower than the 
    // theoretical occupancy (# SMs * max threads / SM)? I have a complex kernel function that exceeds all the 
    // registers past 512 threads / block. I have to lower the thread count per block and launch more blocks as a 
    // result. But i’m hitting an upper-block limit for this kernel. Is the only solution here splitting up the problem
    // between multiple seperate kernel launches? And if so, what’s the best approach? The naive solution is moving from
    // single P100 to A10. Update: the issue was an if statement in the kernel, which solved the problem. 
    // accumulate_buckets_kernel<<<NUM_BLOCKS_3, NUM_THREADS_3>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, points, num_buckets);
    accumulate_buckets_kernel<<<NUM_BLOCKS_3, NUM_THREADS_3>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, points, num_buckets);
    cudaDeviceSynchronize();
    
    // cudaEventRecord(stop);

    // // Synchronization barrier that blocks CPU execution until the specified event is recorded
    // cudaEventSynchronize(stop);

    // // Calculate duraion of execution time 
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // cout << "Time taken by function: " << milliseconds << " milliseconds\n" << endl; 

    // exit(0);

    // cudaDeviceSynchronize();

    // // End timer
    // high_resolution_clock::time_point t2 = high_resolution_clock::now();
    // duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    // std::cout << "It took me " << time_span.count() << " seconds." << endl;
    // exit(0);



    // Need to reconcile the lauch paramerters here............ 

    unsigned *bucket_index;
    cudaMallocManaged(&bucket_index, num_bucket_modules * sizeof(unsigned));
    // int count = 0;
    // for (int i = 0; i < num_buckets; i++) {
    //     if (buckets[i].x.data[0] != 0) {
    //         cout << "bucket index is: " << i << endl;
    //         bucket_index[count] = i;
    //         cout << "bucket_index[count] is; " << bucket_index[count] << endl;
    //         cout << "x: " << "{ " << buckets[i].x.data[0] << "," << buckets[i].x.data[1] << "," << buckets[i].x.data[2] << "," << buckets[i].x.data[3] << " }; "  
    //         << " y: " << "{ " << buckets[i].y.data[0] << "," << buckets[i].y.data[1] << "," << buckets[i].y.data[2] << "," << buckets[i].y.data[3] << " }; "  
    //         << " z: " << "{ " << buckets[i].z.data[0] << "," << buckets[i].z.data[1] << "," << buckets[i].z.data[2] << "," << buckets[i].z.data[3] << " }; " << endl;
    //         count++;
    //     }
    // }
    // cout << "count is: " << count << endl;
    // cout << "??????????????????????????????????????????????\n" << endl;

    // for (int i = 0; i < num_bucket_modules; i++) {
    //     cout << "bucket_index is: " << bucket_index[i] << endl;
    // }
    cudaDeviceSynchronize();


    // At this point we have n buckets and m bucket modules. Need to first sum up the n buckets per bucket module, and then
    // perform a final accumulation of the bucket modules. Launch 4 threads per bucket module.
    
    // Define parameters 
    // unsigned M = 1;
    // unsigned U = (1 << c) / M; // Need to add some offset here and in for loop

    // cout << "U is: " << U << endl;

    // point_t *S;
    // point_t *G;
    // point_t *result;
    point_t *final_sum;
    // cudaMallocManaged(&S, num_bucket_modules * M * 3 * 4 * sizeof(uint64_t));
    // cudaMallocManaged(&G, num_bucket_modules * M * 3 * 4 * sizeof(uint64_t));
    // cudaMallocManaged(&result, num_bucket_modules * 3 * 4 * sizeof(uint64_t));
    cudaMallocManaged(&final_sum, num_bucket_modules * 3 * 4 * sizeof(uint64_t));
    // or change kernel parameters to num_bucket_modules, 4
    // need to look into replacing this with known sum reduction techniques, since it dominates 90% of the runtime
    // Here we're launching 256 with 8 blocks, where each group of 4 threads handles adding 2 points.
    // so 256 / 4 = 64 x 2 x 8 = 1024 points 
    // kernels in default stream are invoked sequentially
    // These parameters will need to be more generalized below
    bucket_module_sum_reduction_lernel_0<<<26, 4>>>(buckets, final_sum, c);
    cudaDeviceSynchronize();

    // cout << "PRINTING bucket_module_sum_reduction_lernel_0: " << endl;
    // for (int i = 0; i < 26; i++) {
    //     for (int j = 0; j < LIMBS; j++) {
    //         printf("result is: %zu\n", final_sum[i].x.data[j]);
    //     }
    //     printf("\n");
    //     for (int y = 0; y < LIMBS; y++) {
    //         printf("result is: %zu\n", final_sum[i].y.data[y]);
    //     }
    //     printf("\n");
    //     for (int z = 0; z < LIMBS; z++) {
    //         printf("result is: %zu\n", final_sum[i].z.data[z]);
    //     }
    //     printf("!!!!!!!!!\n");
    // }
    // printf("\n");

    // cudaSetDevice(0);
    // size_t free_device_mem = 0;
    // size_t total_device_mem = 0;
    // cudaMemGetInfo(&free_device_mem, &total_device_mem);
    // printf("Currently available amount of device memory: %zu bytes\n", free_device_mem);
    // printf("Total amount of device memory: %zu bytes\n", total_device_mem);
    
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    
    // bucket_module_sum_reduction_kernel_1<<<26, 4>>>(buckets, S, G, M, U);
    // cudaDeviceSynchronize();

    // cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
    // cout << "PRINTING bucket_module_sum_reduction_kernel_1: " << endl;
    // for (int i = 0; i < 26; i++) {
    //     for (int j = 0; j < LIMBS; j++) {
    //         printf("result is: %zu\n", G[i].x.data[j]);
    //     }
    //     printf("\n");
    //     for (int y = 0; y < LIMBS; y++) {
    //         printf("result is: %zu\n", G[i].y.data[y]);
    //     }
    //     printf("\n");
    //     for (int z = 0; z < LIMBS; z++) {
    //         printf("result is: %zu\n", G[i].z.data[z]);
    //     }
    //     printf("!!!!!!!!\n");
    // }
    // printf("\n");

    // launch where each group of 4 threads will handle a seperate window k and all M segments within a window K -- problem is small
    // bucket_module_sum_reduction_kernel_2<<<26, 4>>>(result, S, G, M, U);
    // cudaDeviceSynchronize();

    // for (int i = 0; i < 1; i++) {
    //     for (int j = 0; j < LIMBS; j++) {
    //         printf("result is: %zu\n", result[i].x.data[j]);
    //     }
    //     printf("\n");
    //     for (int j = 0; j < LIMBS; j++) {
    //         printf("result is: %zu\n", result[i].y.data[j]);
    //     }
    //     printf("\n");
    //     for (int j = 0; j < LIMBS; j++) {
    //         printf("result is: %zu\n", result[i].z.data[j]);
    //     }
    //     printf("!!!!!!!!\n");
    // }
    // printf("\n");
    // printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

    // exit(0);

    printf("FINAL ACCUMULATION TEST\n");

    // This is still unified memory, which might impact performance when paired with cudaDeviceSynchronize?
    point_t *res;
    cudaMallocManaged(&res, 3 * 4 * sizeof(uint64_t));
    // change points to partial sums
    final_accumulation_kernel<<<1,4>>>(final_sum, res, num_bucket_modules, c);
    // final_accumulation_kernel_test<<<1,4>>>(points, res, num_bucket_modules, c);
    cudaDeviceSynchronize();

    cout << "final_accumulation_kernel_test is: " << res[0].x.data[0] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].x.data[1] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].x.data[2] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].x.data[3] << endl;
    cout << endl;

    cout << "final_accumulation_kernel_test is: " << res[0].y.data[0] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].y.data[1] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].y.data[2] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].y.data[3] << endl;
    cout << endl;

    cout << "final_accumulation_kernel_test is: " << res[0].z.data[0] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].z.data[1] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].z.data[2] << endl;
    cout << "final_accumulation_kernel_test is: " << res[0].z.data[3] << endl;
    cout << endl;

    return res;

    // free memory
}

/***************************************** Function declerations for 'device_ptr' class  *****************************************/

/**
 * Allocate memory using cudaMallocHost
 */
template <class T>
size_t device_ptr<T>::allocate(size_t bytes) {
    T* d_ptr;
    CUDA_WRAPPER(cudaMallocHost(&d_ptr, bytes));
    // CUDA_WRAPPER(cudaMallocHost(&context->h_scalars, context->pipp.get_size_scalars(context->pipp)));

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