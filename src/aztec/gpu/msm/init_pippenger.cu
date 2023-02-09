#include "./init_pippenger.cuh"
#include <iostream>
#include <memory>

using namespace std;

namespace pippenger_common {

/**
 * Consume elliptic curve points (from SRS) and scalars
 */
template <class T>
T msm_t<T>::read_points_scalars() {
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();
    return points;
}

/**
 * Entry point into "Pippenger's Bucket Method"
 */ 
template <class T>
void msm_t<T>::pippenger_init(T points) {
    // Dynamically initialize new context object
    Context<bucket_t, point_t, scalar_t, affine_t> *context = new Context<bucket_t, point_t, scalar_t, affine_t>();

    try {
        // Initialize parameters for MSM  
        context->pipp = context->pipp.initialize_msm(NUM_POINTS);    

        // Allocate GPU storage
        context->d_points_idx = context->pipp.allocate_bases(context->pipp);
        context->d_buckets_idx = context->pipp.allocate_buckets(context->pipp);
        for (size_t i = 0; i < NUM_BATCH_THREADS; i++) {
            context->d_scalar_idx[i] = context->pipp.allocate_scalars(context->pipp);
        }

        // Allocate pinned memory on host
        cudaError_t status = cudaMallocHost(&context->h_scalars, context->pipp.get_size_scalars(context->pipp));
        if (status != cudaSuccess) {
            printf("Error allocating pinned host memory\n");
        }

        // Transfer bases to device
        context->pipp.transfer_bases_to_device(context->pipp, context->d_points_idx, points, context->ffi_affine_sz);

        // Create results container
        context->result0 = context->pipp.result_container(context->pipp);
        context->result1 = context->pipp.result_container(context->pipp);
    }
    catch (cudaError_t) {
        cout << "ADD CUDA ERROR MESSAGE!" << endl;
    }
}

/**
 * Perform MSM
 */ 
template <class T>
void msm_t<T>::pippenger_execute(T points) {
    
}

}