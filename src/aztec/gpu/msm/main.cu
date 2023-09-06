#include "pippenger.cu"
#include <chrono>
#include <common/assert.hpp>
#include <cstdlib>
#include <ecc/curves/bn254/scalar_multiplication/scalar_multiplication.hpp>
#include <plonk/reference_string/file_reference_string.hpp>
#include <polynomials/polynomial_arithmetic.hpp>
#include <iostream>
#include <fstream>
#include "msm.h"
#include "ntt.h"
#include "projective.h"
#include "ve_mod_mult.h"
#include <dlfcn.h>

using namespace std;
using namespace pippenger_common;
using namespace waffle;
using namespace barretenberg;

int main(int, char**) {
    
    void* lib_handle = dlopen("/home/ubuntu/Cuda-Barretenberg/src/aztec/gpu", RTLD_LAZY);

    BN254_projective_t* (*myfunc_ptr)() = (BN254_projective_t * (*)())dlsym(lib_handle, "projective_zero_bn254");

    BN254_projective_t result = (*myfunc_ptr)();

    cout << "result is: " << result[0] << endl;

    exit(0);
    
    // Initialize dynamic 'msm_t' object 
    // msm_t<point_t, scalar_t> *msm = new msm_t<point_t, scalar_t>();
    
    // // Construct elliptic curve points from SRS
    // auto reference_string = std::make_shared<waffle::FileReferenceString>(NUM_POINTS, "../srs_db/ignition");
    // g1::affine_element* points = reference_string->get_monomials();

    // // Construct random scalars 
    // std::vector<fr> scalars;
    // scalars.reserve(NUM_POINTS);
    // for (size_t i = 0; i < NUM_POINTS; ++i) {
    //     scalars.emplace_back(fr::random_element());
    // }

    // g1::affine_element* icicle_result;
    // msm_cuda_bn254(icicle_result, points, &scalars[0], scalars.size(), 0);

    // Number of streams
    // int num_streams = 1;

    // // Initialize dynamic pippenger 'context' object
    // Context<point_t, scalar_t> *context = msm->pippenger_initialize(points, &scalars[0], num_streams, NUM_POINTS);

    // // Execute "Double-And-Add" reference kernel
    // g1_gpu::element *result_1 = msm->msm_double_and_add(context, NUM_POINTS, points, &scalars[0]);

    // // Execute "Pippenger's Bucket Method" kernel
    // g1_gpu::element **result_2 = msm->msm_bucket_method(context, points, &scalars[0], num_streams);

    // // Print results 
    // context->pipp.print_result(result_1, result_2);

    // // Verify the final results are equal
    // context->pipp.verify_result(result_1, result_2);
}