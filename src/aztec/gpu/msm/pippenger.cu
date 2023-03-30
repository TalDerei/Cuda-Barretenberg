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
void msm_t<A, S, J>::pippenger_execute(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t num_points, A* points) {
    // Read scalars
    fr_gpu *scalars;
    cudaMallocManaged(&scalars, NUM_POINTS * LIMBS * sizeof(uint64_t));
    context->pipp.read_scalars(scalars);
    
    // Create auxilary stream
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
    J *final_result;
    J *result;

    // Allocate cuda memory 
    cudaMallocManaged(&d_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&final_result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));

    // Read points and scalars
    context->pipp.read_jacobian_curve_points(j_points);
    context->pipp.read_scalars(d_scalars);

    // Naive MSM method
    msm_naive_kernel<<<8, 512>>>(j_points, d_scalars, result, NUM_POINTS);

    // Sum reduction 
    sum_reduction<<<1, 4>>>(result, final_result);

    // Final accumulation
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

    // Allocate unified memory
    cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&d_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));

    // Calculate maximum occupancy
    int THREADS;
    int BLOCKS;
    cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, bucket_module_sum_reduction_kernel_1, 0, 0);
    cout << "max threads is: " << THREADS << endl;
    cout << "max blocks is: " << BLOCKS << endl;

    // Read points
    context->pipp.read_jacobian_curve_points(j_points);
    context->pipp.read_scalars(d_scalars);

    unsigned bitsize = 254;
    unsigned c = 10;
    
    // Start timer
    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    context->pipp.initialize_buckets(d_scalars, j_points, bitsize, c, npoints);
    
    // End timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "It took me " << time_span.count() << " seconds." << endl;
}

/**
 * Perform naive MSM
 */ 
template <class A, class S, class J>
void msm_t<A, S, J>::pippenger_test(Context<bucket_t,point_t,scalar_t,affine_t> *context, size_t npoints, A *points) {
    point_t *j_points;
    S *d_scalars;
    J *final_result;
    point_t *result_jacobian;
    point_t *result_projective;

    // Allocate cuda memory 
    cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&d_scalars, NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result_jacobian, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&result_projective, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&final_result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));

    // Read points and scalars
    context->pipp.read_jacobian_curve_points(j_points);
    context->pipp.read_scalars(d_scalars);

    test_double_add<<<1,4>>>(j_points, d_scalars, final_result);
    cudaDeviceSynchronize();

    context->pipp.print_result(final_result);
}

}