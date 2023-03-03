#include "pippenger.cuh"
#include <iostream>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

namespace pippenger_common {

/**
 * Read elliptic curve points from SRS
 */
template <class T, class B>
T* msm_t<T, B>::read_curve_points() {
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();

    cout << points[0].x.data[0] << endl;
    cout << points[0].x.data[1] << endl;
    cout << points[0].x.data[2] << endl;
    cout << points[0].x.data[3] << endl;
    cout << points[0].y.data[0] << endl;
    cout << points[0].y.data[1] << endl;
    cout << points[0].y.data[2] << endl;
    cout << points[0].y.data[3] << endl;

    return points;
}

/**
 * Read scalars from scalar field
 */
template <class T, class B>
B* msm_t<T, B>::read_scalars() {
    uint64_t temp[NUM_POINTS * 4];
    fr_gpu *scalars = new fr_gpu[NUM_POINTS];
    uint64_t scalar;

    // File stream
    ifstream stream;
    stream.open("../src/aztec/gpu/msm/scalars/scalars.txt", ios::in);

    // Read scalars
    if (stream.is_open()) {   
        int i = 0;  
        while (stream.good()) { 
            stream >> scalar;
            temp[i] = scalar;
            i++;
        }
    }
    
    for (size_t i = 0; i < (sizeof(temp) / sizeof(uint64_t)) / 4; ++i) {    
        fr_gpu element{ temp[i], temp[i + 1], temp[i + 2], temp[i + 3] };
        scalars[i] = element;        
    }
        
    return scalars;
}

/**
 * Entry point into "Pippenger's Bucket Method"
 */ 
template <class T, class B>
Context<bucket_t, point_t, scalar_t, affine_t>* msm_t<T, B>::pippenger_initialize(T* points) {
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
template <class T, class B>
void msm_t<T, B>::pippenger_execute(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t num_points, T* points, B* scalars) {
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

}