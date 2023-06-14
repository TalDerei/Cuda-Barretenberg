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

        // Allocate GPU storage for bases (curve points) and scalars 
        context->d_points_idx = context->pipp.allocate_bases(context->pipp);
        context->d_scalar_idx = context->pipp.allocate_scalars(context->pipp);

        // Create auxilary stream
        stream_t aux_stream(context->pipp.device);

        // Allocate memory for converting affine to jacobian points ***THIS WILL NEED TO BE CLEANED UP***
        g1_gpu::affine_element *a_points;
        g1_gpu::element *j_points;
        cudaMallocManaged(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
        cudaMallocManaged(&a_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
        cudaMemcpy(a_points, points, NUM_POINTS * LIMBS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        affine_to_jacobian<<<1, 4, 0, aux_stream>>>(a_points, j_points, NUM_POINTS);

        // Transfer bases and scalars to device
        context->pipp.transfer_scalars_to_device(context->pipp, context->pipp.device_scalar_ptrs.d_ptrs[0], scalars, aux_stream);
        context->pipp.transfer_bases_to_device(context->pipp, context->pipp.device_base_ptrs.d_ptrs[0], j_points, aux_stream);

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
    // Allocate memory for result
    g1_gpu::element *result;
    cudaMallocManaged(&result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t));
    
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
    // MSM parameters
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