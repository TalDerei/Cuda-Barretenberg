#include "./common.cu"
#include <iostream>
#include <memory>

using namespace std;

namespace pippenger_common {

/**
 * Consume elliptic curve points and scalars
 */ 
void read_points_scalars() {
    auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
    g1::affine_element* points = reference_string->get_monomials();
}

/**
 * Entry point into "Pippenger's Bucket Method"
 */ 
void pippenger_init() {
    // Dynamically initialize new context
    Context<bucket_t, point_t, scalar_t, affine_t> *context = new Context<bucket_t, point_t, scalar_t, affine_t>();

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
}

}