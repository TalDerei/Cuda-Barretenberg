#include "./init_pippenger.cuh"
#include <iostream>
#include <memory>

using namespace std;

namespace pippenger_common {

/**
 * Consume elliptic curve points (from SRS) and scalars
 */
template <class T>
T* msm_t<T>::read_points_scalars() {
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();

    // cout << points[0].x.data[0] << endl;
    // cout << points[0].x.data[1] << endl;
    // cout << points[0].x.data[2] << endl;
    // cout << points[0].x.data[3] << endl;

    return points;
}

/**
 * Entry point into "Pippenger's Bucket Method"
 */ 
template <class T>
Context<bucket_t, point_t, scalar_t, affine_t>* msm_t<T>::pippenger_initialize(T* points) {
    try {
        // Initialize context object 
        Context<bucket_t, point_t, scalar_t, affine_t> *context = new Context<bucket_t, point_t, scalar_t, affine_t>();

        // Initialize MSM parameters
        context->pipp = context->pipp.initialize_msm(NUM_POINTS);    

        // Allocate GPU storage for bases, scalars, and buckets 
        context->d_points_idx = context->pipp.allocate_bases(context->pipp);
        context->d_buckets_idx = context->pipp.allocate_buckets(context->pipp);
        for (size_t i = 0; i < NUM_BATCH_THREADS; i++) {
            context->d_scalar_idx[i] = context->pipp.allocate_scalars(context->pipp);
        }
 
        // Allocate pinned memory on host for scalars
        cudaError_t status = cudaMallocHost(&context->h_scalars, context->pipp.get_size_scalars(context->pipp));
        if (status != cudaSuccess) {
            cout << "Error allocating pinned host memory: " << cudaGetErrorString(status) << endl;
            throw cudaGetErrorString(status);
        }

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
    }
}

/**
 * Perform MSM
 */ 
template <class T>
void msm_t<T>::pippenger_execute(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t num_points, T* points) {
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
        
        // Overlap bucket computation on the GPU with transfer of the next set of scalars
        int batch = 1;
        int work = 0;

        // Transfer scalars to device
        context->pipp.transfer_scalars_to_device(context->pipp, context->d_scalar_idx[1], points, aux_stream.stream);

        
    }
    catch (cudaError_t) {
        cout << "ADD CUDA ERROR MESSAGE!" << endl;
    }
}

}