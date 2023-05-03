#include "common.cu"
#include <iostream>
#include <memory>
#include <ecc/curves/bn254/fr.hpp>
#include <plonk/proof_system/types/program_settings.hpp>
#include <plonk/reference_string/file_reference_string.hpp>

using namespace std;
using namespace barretenberg;

namespace pippenger_common {

/**
 * Pippenger top-level function prototypes 
 */
template < typename A, typename S, typename J >
class msm_t {
    public: 
        msm_t() {}
        Context<bucket_t, point_t, scalar_t, affine_t>* pippenger_initialize(g1::affine_element* points);
                
        // Add back the native sum reduction method
        // void naive_sum_reduction(Context<, point_t, scalar_t, affine_t> *context, size_t npoints, g1::affine_element* points);

        g1_gpu::element* naive_double_and_add(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, g1::affine_element* points, fr *scalars);

        g1_gpu::element* msm_bucket_method(Context<bucket_t, point_t, scalar_t, affine_t> *context, size_t npoints, g1::affine_element* points, fr *scalars);
                
        void verify_result(J *result_1, J *result_2);
};

}