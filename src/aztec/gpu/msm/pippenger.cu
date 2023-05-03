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
template <class A, class S, class J>
Context<bucket_t, point_t, scalar_t, affine_t>* msm_t<A, S, J>::pippenger_initialize(g1::affine_element* points) {
    try {
        // Initialize 'Context' object 
        Context<bucket_t, point_t, scalar_t, affine_t> *context = new Context<bucket_t, point_t, scalar_t, affine_t>();

        // Initialize MSM parameters
        context->pipp = context->pipp.initialize_msm(context->pipp, NUM_POINTS);    

        // // Allocate GPU storage for bases, scalars, and buckets 
        // context->d_points_idx = context->pipp.allocate_bases(context->pipp);
        // context->d_scalar_idx = context->pipp.allocate_scalars(context->pipp);
        // context->d_buckets_idx = context->pipp.allocate_buckets(context->pipp);

        // // Allocate pinned memory on host for scalars
        // CUDA_WRAPPER(cudaMallocHost(&context->h_scalars, context->pipp.get_size_scalars(context->pipp)));

        // // Transfer bases to device
        // // But we haven't used cudaMallocHost here?
        // context->pipp.transfer_bases_to_device(context->pipp, context->d_points_idx, points);
    
        // // Create results container
        // // don't think we need this either.
        // context->result0 = context->pipp.result_container(context->pipp);
        // context->result1 = context->pipp.result_container(context->pipp);


        // -------------------------------
        // Create auxilary stream
        // stream_t aux_stream(context->pipp.device);

        // Return initialized context object

           // Don't use this currently
        // size_t scalar_size = context->pipp.get_size_scalars(context->pipp);

        // Transfer scalars to device
        // context->pipp.transfer_scalars_to_device(context->pipp, context->d_scalar_idx[1], scalars, aux_stream);

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

template <class A, class S, class J>
g1_gpu::element* msm_t<A, S, J>::naive_double_and_add(
Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, g1::affine_element *points, fr *scalars) {
    S *d_scalars;
    A *a_points;
    J *j_points;
    J *result;
    
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
    double_and_add_kernel<<<1, 4>>>(d_scalars, j_points, result, NUM_POINTS);
    cudaDeviceSynchronize();

    return result;
}

/**
 * Perform MSM Bucket Method
 */ 
template <class A, class S, class J>
g1_gpu::element* msm_t<A, S, J>::msm_bucket_method(
Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, g1::affine_element *points, fr *scalars) {
    S *d_scalars;
    A *a_points;
    J *j_points;
    J *result;
    
     // Allocat memory
    cudaMallocManaged(&d_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&a_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));

    // Copy points and scalars to device
    cudaMemcpy(d_scalars, scalars, NUM_POINTS * LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(a_points, points, NUM_POINTS * LIMBS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Convert from affine coordinates to jacobian
    affine_to_jacobian<<<1, 4>>>(a_points, j_points, npoints);

    // MSM parameters
    unsigned bitsize = 254;
    unsigned c = 10;
    
    // Start timer
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Launch pippenger kernel
    g1_gpu::element *res = context->pipp.execute_bucket_method(d_scalars, j_points, bitsize, c, npoints);
    
    // End timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took me " << time_span.count() << " seconds." << endl;

    return res;
}

/**
 * Verify double-and-add and pippenger's bucket method results
 */ 
template <class A, class S, class J>
void msm_t<A, S, J>::verify_result(J *result_1, J *result_2) {
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

}