#include "pippenger.cuh"
#include <iostream>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;
using namespace std::chrono;

namespace pippenger_common {

/**
 * Entry point into "Pippenger's Bucket Method" 
 */ 
template <class P, class S>
Context<point_t, scalar_t> *msm_t<P, S>::pippenger_initialize(g1::affine_element* points, fr *scalars, int num_streams) {
    try {
        // Initialize 'Context' object 
        Context<point_t, scalar_t> *context = new Context<point_t, scalar_t>();

        // Dynamically allocate streams at runtime
        context->pipp.streams = new cudaStream_t[num_streams];
        for (int i = 0; i < num_streams; i++) {
            CUDA_WRAPPER(cudaStreamCreateWithFlags(&(context->pipp.streams[i]), cudaStreamNonBlocking));
        }

        // Allocate GPU storage for elliptic curve bases and scalars 
        for (int i = 0; i < num_streams; i++) { 
            context->pipp.allocate_bases(context->pipp);
            context->pipp.allocate_scalars(context->pipp);
        }

        // Convert affine to jacobian coordinates 
        g1_gpu::affine_element *a_points;
        g1_gpu::element *j_points;
        CUDA_WRAPPER(cudaMalloc(&j_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t)));
        CUDA_WRAPPER(cudaMalloc(&a_points, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t)));
        CUDA_WRAPPER(cudaMemcpy(a_points, points, NUM_POINTS * LIMBS * 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));
        affine_to_jacobian<<<(NUM_POINTS / 256), 256, 0, context->pipp.streams[0]>>>(a_points, j_points, NUM_POINTS);
        
        // Transfer bases and scalars to device
        for (int i = 0; i < num_streams; i++) { 
            context->pipp.transfer_scalars_to_device(
                context->pipp, context->pipp.device_scalar_ptrs.d_ptrs[i], scalars, context->pipp.streams[i]
            );
            context->pipp.transfer_bases_to_device(
                context->pipp, context->pipp.device_base_ptrs.d_ptrs[i], j_points, context->pipp.streams[i]
            );
        }
        
        return context;
    }
    catch (cudaError_t) {
        cout << "Failed to initialize MSM." << endl;
        throw;
    }
}

template <class P, class S>
g1_gpu::element* msm_t<P, S>::msm_double_and_add(
Context<point_t, scalar_t> *context, size_t npoints, g1::affine_element *points, fr *scalars) {
    // Allocate memory and launch kernel 
    g1_gpu::element *result;
    CUDA_WRAPPER(cudaMallocManaged(&result, 3 * NUM_POINTS * LIMBS * sizeof(uint64_t)));
    double_and_add_kernel<<<1, 4>>>(
        context->pipp.device_scalar_ptrs.d_ptrs[0], context->pipp.device_base_ptrs.d_ptrs[0], result, NUM_POINTS
    );
    cudaDeviceSynchronize();
    
    return result;
}

/**
 * Perform MSM Bucket Method
 */ 
template <class P, class S>
g1_gpu::element** msm_t<P, S>::msm_bucket_method(
Context<point_t, scalar_t> *context, size_t npoints, g1::affine_element *points, fr *scalars, int num_streams) {
    // Start timer
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Launch pippenger kernel
    g1_gpu::element **result = new g1_gpu::element*[num_streams];
    for (int i = 0; i < num_streams; i++) { 
        result[i] = context->pipp.execute_bucket_method(
            context->pipp, context->pipp.device_scalar_ptrs.d_ptrs[i], context->pipp.device_base_ptrs.d_ptrs[i], BITSIZE, C, npoints
        );
    }
    
    // End timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Pippenger executed in " << time_span.count() << " seconds." << endl;

    return result;
}

}