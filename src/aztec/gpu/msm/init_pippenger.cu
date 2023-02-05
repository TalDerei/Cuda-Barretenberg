#include "./common.cu"
#include <iostream>
#include <memory>

using namespace std;

namespace pippenger_common {

size_t NUM_POINTS = 1 << 15;

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
    Context<bucket_t, point_t, scalar_t> *context = new Context<bucket_t, point_t, scalar_t>();

    // Initialize parameters for MSM  
    context->pipp = context->pipp.initialize_msm(NUM_POINTS);    
}

}