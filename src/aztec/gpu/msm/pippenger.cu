#include "pippenger.cuh"
#include <iostream>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;

namespace pippenger_common {

/**
 * Entry point into "Pippenger's Bucket Method"
 */ 
template <class P, class S>
Context<point_t, scalar_t> *msm_t<P, S>::pippenger_initialize(g1::affine_element* points, fr *scalars) {
    try {
        // Initialize 'Context' object 
        Context<point_t, scalar_t> *context = new Context<point_t, scalar_t>();

        // Initialize MSM parameters
        context->pipp = context->pipp.initialize_msm(context->pipp, NUM_POINTS);   

        // Allocate GPU storage for bases, scalars, and buckets 
        context->d_points_idx = context->pipp.allocate_bases(context->pipp);
        context->d_scalar_idx = context->pipp.allocate_scalars(context->pipp);
        // context->d_buckets_idx = context->pipp.allocate_buckets(context->pipp);

        // -------------------------------
        // Allocate pinned memory on host and device memory -- add cudaMallocHost
        cout << "amount of host memory allocated is: " << context->pipp.get_size_scalars(context->pipp) << endl;
        // CUDA_WRAPPER(cudaMallocHost(&context->h_scalars, context->pipp.get_size_scalars(context->pipp)));
        S *test_scalars;
        S *test_scalars_new;
        CUDA_WRAPPER(cudaMalloc(&test_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t)));
        test_scalars_new = (S*)malloc(NUM_POINTS * LIMBS * sizeof(uint64_t));
        // CUDA_WRAPPER(cudaMalloc(&test_scalars_new, NUM_POINTS * LIMBS * sizeof(uint64_t)));
        // // Transfer bases to device
        // // But we haven't used cudaMallocHost here?
        // context->pipp.transfer_bases_to_device(context->pipp, context->d_points_idx, points);
    

        // -------------------------------
        // Create auxilary stream
        stream_t aux_stream(context->pipp.device);

        // Return initialized context object

        // Don't use this currently
        // size_t scalar_size = context->pipp.get_size_scalars(context->pipp);
        // cudaMemcpyAsync(context->h_scalars, scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice, aux_stream);
        // cout << "printing scalars again!" << endl;
        // cout << context->h_scalars[0].data[0] << endl;
        // cout << context->h_scalars[0].data[1] << endl;
        // cout << context->h_scalars[0].data[2] << endl;
        // cout << context->h_scalars[0].data[3] << endl;

        // Convert affine to jacobian coordinates -- is there a way around this to make the conversion in the kernel?
           // Allocate memory
        g1_gpu::affine_element *a_points;
        g1_gpu::element *j_points;
        cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
        cudaMallocManaged(&a_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
        
        // Copy points and scalars to device
        cudaMemcpy(a_points, points, NUM_POINTS * LIMBS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        affine_to_jacobian<<<1, 4>>>(a_points, j_points, NUM_POINTS);

        cudaDeviceSynchronize();

        context->pipp.transfer_scalars_to_device(context->pipp, context->pipp.device_scalar_ptrs.d_ptrs[0], scalars, aux_stream);
        context->pipp.transfer_bases_to_device(context->pipp, context->pipp.device_base_ptrs.d_ptrs[0], j_points, aux_stream);
        cudaDeviceSynchronize();

        cout << "printing scalars again!" << endl;
        cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[0] << endl;
        cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[1] << endl;
        cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[2] << endl;
        cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[3] << endl;

        test_kernel<<<1, 4, 0, aux_stream>>>(context->pipp.device_scalar_ptrs.d_ptrs[0], test_scalars);
        cudaMemcpyAsync(test_scalars_new, test_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyDeviceToHost, aux_stream);

        // test_kernel<<<1, 4, 0, aux_stream>>>(context->h_scalars, test_scalars);
        cudaDeviceSynchronize();

        // cout << "h_scalars.data: " << context->h_scalars[0].data[0] << endl;
        // cout << "h_scalars.data: " << context->h_scalars[0].data[1] << endl;
        // cout << "h_scalars.data: " << context->h_scalars[0].data[2] << endl;
        // cout << "h_scalars.data: " << context->h_scalars[0].data[3] << endl;

        cout << "test_scalars: " << test_scalars_new[0].data[0] << endl;
        cout << "test_scalars: " << test_scalars_new[0].data[1] << endl;
        cout << "test_scalars: " << test_scalars_new[0].data[2] << endl;
        cout << "test_scalars: " << test_scalars_new[0].data[3] << endl;
        
        // exit(0);

        // switch cudaMemcpy DevcietoHost


        // Synchronize cuda stream with CPU thread, blocking execution until stream completed all operations
        // Synchronzation neccessary before launching kernel?
        // CUDA_WRAPPER(cudaStreamSynchronize(aux_stream));

        // Open questions before launching the kenerl:
            // 1. Why are bases and scalars allocated in different streams?
            // 2. Why does only scalars use cudaMallocHost, when both use cudaMemcpyAsync?
            // 3. Where is &context->h_scalars being used?

        // move the bucket_sum method functions inside here.

        return context;
    }
    catch (cudaError_t) {
        cout << "Failed to initialize MSM." << endl;
        throw;
    }
}

template <class P, class S>
g1_gpu::element* msm_t<P, S>::naive_double_and_add(
Context<point_t, scalar_t> *context, size_t npoints, g1::affine_element *points, fr *scalars) {
    S *d_scalars;
    g1_gpu::affine_element *a_points;
    g1_gpu::element *j_points;
    P *result;

    cout << "printing scalars inside naive_double_and_add" << endl;
    cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[0] << endl;
    cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[1] << endl;
    cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[2] << endl;
    cout << context->pipp.device_scalar_ptrs.d_ptrs[0][0].data[3] << endl;

    // Allocate memory
    cudaMallocManaged(&d_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&a_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    
    // Copy points and scalars to device
    cudaMemcpy(d_scalars, scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(a_points, points, NUM_POINTS * LIMBS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Convert from affine coordinates to jacobian
    affine_to_jacobian<<<1, 4>>>(a_points, j_points, npoints);

    // Launch double and add kernel
    double_and_add_kernel<<<1, 4>>>(context->pipp.device_scalar_ptrs.d_ptrs[0], context->pipp.device_base_ptrs.d_ptrs[0], result, NUM_POINTS);
    cudaDeviceSynchronize();

    return result;
}

/**
 * Perform MSM Bucket Method
 */ 
template <class P, class S>
g1_gpu::element* msm_t<P, S>::msm_bucket_method(
Context<point_t, scalar_t> *context, size_t npoints, g1::affine_element *points, fr *scalars) {
    S *d_scalars;
    g1_gpu::affine_element *a_points;
    g1_gpu::element *j_points;
    P *result;
    
     // Allocat memory
    cudaMallocManaged(&d_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&a_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));

    // Copy points and scalars to device
    cudaMemcpy(d_scalars, scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(a_points, points, NUM_POINTS * LIMBS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Convert from affine coordinates to jacobian -- will need to change this to be more performant or get ride of it completely
    // Change launch parameters for more efficient kernel invocation
    affine_to_jacobian<<<1, 4>>>(a_points, j_points, npoints);

    // MSM parameters -- need to construct method to dynamically choose the best depending on the size of the msm
    unsigned bitsize = 254;
    unsigned c = 10;
    
    // Start timer
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Launch pippenger kernel
    g1_gpu::element *res = context->pipp.execute_bucket_method(context->pipp, context->pipp.device_scalar_ptrs.d_ptrs[0], context->pipp.device_base_ptrs.d_ptrs[0], bitsize, c, npoints);
    
    // End timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took me " << time_span.count() << " seconds." << endl;

    return res;
}

}