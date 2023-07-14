#include "pippenger.cu"

using namespace std;
using namespace pippenger_common;
using namespace waffle;

int main(int, char**) {
    // Initialize dynamic 'msm_t' object 
    msm_t<point_t, scalar_t> *msm = new msm_t<point_t, scalar_t>();
    
    // Construct elliptic curve points from SRS
    auto reference_string = std::make_shared<waffle::FileReferenceString>(NUM_POINTS, "../srs_db/ignition");
    g1::affine_element* points = reference_string->get_monomials();

    // Construct random scalars 
    std::vector<fr> scalars;
    fr element = fr::random_element();
    fr accumulator = element;
    scalars.reserve(NUM_POINTS);
    for (size_t i = 0; i < NUM_POINTS; ++i) {
        accumulator *= element;
        scalars.emplace_back(accumulator);
    }

    int num_streams = 1;

    // Initialize dynamic pippenger 'context' object
    Context<point_t, scalar_t> *context = msm->pippenger_initialize(points,  &scalars[0], num_streams);

    // Execute "Double-And-Add" reference kernel
    g1_gpu::element *result_1 = msm->msm_double_and_add(context, NUM_POINTS, points, &scalars[0]);

    // Execute "Pippenger's Bucket Method" kernel
    g1_gpu::element **result_2 = msm->msm_bucket_method(context, NUM_POINTS, points, &scalars[0], num_streams);

    // Print results 
    context->pipp.print_result(result_1, result_2);

    // Verify the final results are equal
    context->pipp.verify_result(result_1, result_2);
}