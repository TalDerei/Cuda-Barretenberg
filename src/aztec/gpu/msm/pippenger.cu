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
Context<bucket_t, point_t, scalar_t, affine_t>* msm_t<A, S, J>::pippenger_initialize(A* points) {
    try {
        // Initialize context object 
        Context<bucket_t, point_t, scalar_t, affine_t> *context = new Context<bucket_t, point_t, scalar_t, affine_t>();

        // Initialize MSM parameters
        context->pipp = context->pipp.initialize_msm(context->pipp, NUM_POINTS);    

        // Allocate GPU storage for bases, scalars, and buckets 
        context->d_points_idx = context->pipp.allocate_bases(context->pipp);
        context->d_buckets_idx = context->pipp.allocate_buckets(context->pipp);

        for (size_t i = 0; i < NUM_BATCH_THREADS; i++) {
            context->d_scalar_idx[i] = context->pipp.allocate_scalars(context->pipp);
        }

        // Allocate pinned memory on host for scalars
        CUDA_WRAPPER(cudaMallocHost(&context->h_scalars, context->pipp.get_size_scalars(context->pipp)));

        // Transfer bases to device
        context->pipp.transfer_bases_to_device(context->pipp, context->d_points_idx, points);
    
        // Create results container
        context->result0 = context->pipp.result_container(context->pipp);
        context->result1 = context->pipp.result_container(context->pipp);

        // Return initialized context object
        return context;
    }
    catch (cudaError_t) {
        cout << "Failed to initialize MSM." << endl;
        throw;
    }
}

/**
 * Perform MSM
 */ 
template <class A, class S, class J>
void msm_t<A, S, J>::pippenger_execute(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t num_points, A* points, S* scalars) {
    // Create auxilary stream
    // will need to change this?
    stream_t aux_stream(context->pipp.device);

    try {        
        // Store results
        typename pipp_t::result_container_t *kernel_result = &context->result0;
        typename pipp_t::result_container_t *accumulation_result = &context->result1;

        size_t d_scalars_xfer = context->d_scalar_idx[0];
        size_t d_scalars_compute = context->d_scalar_idx[1];

        // Create a channel_t object from thread pool
        channel_t<size_t> channel;

        size_t scalar_size = context->pipp.get_size_scalars(context->pipp);
        
        // Overlap bucket computation on the GPU with transfer of scalars
        int batch = 1;
        int work = 0;

        // Transfer scalars to device
        context->pipp.transfer_scalars_to_device(context->pipp, context->d_scalar_idx[1], scalars, aux_stream);

        // Synchronize cuda stream with CPU thread, blocking execution until stream completed all operations
        CUDA_WRAPPER(cudaStreamSynchronize(aux_stream));

        // Launch kernel
        context->pipp.launch_kernel(context->pipp, context->d_points_idx, context->d_scalar_idx[1], context->d_buckets_idx);
    }
    catch (cudaError_t) {
        cout << "Failed executing multi-scalar multiplication!" << endl;
        throw;
    }
}

/**
 * Perform naive MSM
 */ 
template <class A, class S, class J>
void msm_t<A, S, J>::naive_msm(Context<bucket_t,point_t,scalar_t,affine_t> *context, size_t npoints, A *points) {
    fr_gpu *d_scalars;
    J *j_points;
    J *result;
    J *final_result;
    J *result_acc;
    fq_gpu *result_2;

    // Allocate cuda memory 
    cudaMallocManaged(&d_scalars, POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&j_points, 3 * POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result_acc, 3 * POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, 3 * POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result_2, 3 * POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&final_result, 3 * POINTS * LIMBS * sizeof(uint64_t));

    // Read points and scalars
    g1::element *points_r = context->pipp.read_jacobian_curve_points(j_points);
    fr_gpu *scalars_r = context->pipp.read_scalars(d_scalars);

    // Perform MSM method 1
    simple_msm_naive<<<1, 4>>>(j_points, d_scalars, result, POINTS);
    cudaDeviceSynchronize();

    // Perform MSM method 2
    simple_msm_naive_2<<<4, 1024>>>(j_points, d_scalars, result_2, result_acc, POINTS);

    // Perform final accumulation by summing all elements in the vector
    sum_reduction<<<1, 8>>>(result_acc, final_result);
    sum_reduction_accumulate<<<1, 4>>>(final_result, final_result);
    cudaDeviceSynchronize();
    
    // Print results
    context->pipp.print_result(final_result);
}

/**
 * Perform MSM Bucket Method
 */ 
template <class A, class S, class J>
void msm_t<A, S, J>::msm_bucket_method(Context<bucket_t,point_t,scalar_t,affine_t> *context, size_t npoints, A *points) {
    J *j_points;
    S *d_scalars;
    J *result;

    // cout << "NUM_POINTS IS: " << npoints << endl;
    // exit(0);

    // Allocate unified memory
    cudaMallocManaged(&j_points, 3 * POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&d_scalars, POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, 3 * POINTS * LIMBS * sizeof(uint64_t));

    // Read points
    context->pipp.read_jacobian_curve_points(j_points);
    context->pipp.read_scalars(d_scalars);

    int THREADS;
    int BLOCKS;

    cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, accumulate_buckets_kernel, 0, 0);

    cout << "max threads is: " << THREADS << endl;
    cout << "max blocks is: " << BLOCKS << endl;

    // Parameters
    unsigned bitsize = 255;
    unsigned c = 10;
    // LOOK INTO MAKING C = 16, SO WE DON'T NEED TO WORRY ABOUT EDGE CASES BETWEEN BUCKETS

        // Calculate the number of windows 
    unsigned num_bucket_modules = bitsize / c; 
    if (bitsize / c) {  
        num_bucket_modules++;
    }
    
    // timer
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    context->pipp.initialize_buckets(d_scalars, j_points, bitsize, c, npoints);
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took me " << time_span.count() << " seconds.";
    std::cout << std::endl;
    

    // Scalars are 10-bits, so between 0-1024. There are 26 windows (or bucket modules).
    // Group elements with the same scalars go into the same buckets, i.e. 6 * G1 means
    // G1 goes into bucket 6. And you have 2^c total buckets per window.

    // Step 1. Partition b-bit scalars to c-bits
    // Step 2. Add points to buckets depending on their scalar
    // step 3. Sum the contents of each bucket into N bucket sums for each window
}

}