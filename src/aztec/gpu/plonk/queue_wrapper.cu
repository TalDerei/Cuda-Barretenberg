#include "queue_wrapper.cuh"
#include "../msm/pippenger.cu"

using namespace queue_gpu_wrapper;
using namespace pippenger_common;

void QueueWrapper::process_queue() {
    cout << "Entered virtual process_queue()" << endl;

    for (const auto& item : work_item_queue) {
        switch (item.work_type) {
            case WorkType::SCALAR_MULTIPLICATION: {
                cout << "SCALAR_MULTIPLICATION!" << endl;
                if (item.constant == MSMSize::N_PLUS_ONE) {
                    if (key->reference_string->get_size() < key->small_domain.size + 1) {
                        info("Reference string too small for Pippenger.");
                    }

                    auto reference_string = std::make_shared<waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
                    g1::affine_element* pointss = reference_string->get_monomials();

                    // cout << "points is: " << pointss[0].x.data[0] << endl;
                    // cout << "points is: " << pointss[0].x.data[1] << endl;
                    // cout << "points is: " << pointss[0].x.data[2] << endl;
                    // cout << "points is: " << pointss[0].x.data[3] << endl;
                    // cout << "points is: " << pointss[0].y.data[0] << endl;
                    // cout << "points is: " << pointss[0].y.data[1] << endl;
                    // cout << "points is: " << pointss[0].y.data[2] << endl;
                    // cout << "points is: " << pointss[0].y.data[3] << endl;

                    // cout << "points is: " << pointss[1].x.data[0] << endl;
                    // cout << "points is: " << pointss[1].x.data[1] << endl;
                    // cout << "points is: " << pointss[1].x.data[2] << endl;
                    // cout << "points is: " << pointss[1].x.data[3] << endl;
                    // cout << "points is: " << pointss[1].y.data[0] << endl;
                    // cout << "points is: " << pointss[1].y.data[1] << endl;
                    // cout << "points is: " << pointss[1].y.data[2] << endl;
                    // cout << "points is: " << pointss[1].y.data[3] << endl;

                    // Naive implementation for quick testing
                    // msm_t<g1_gpu::affine_element, scalar_t, point_t> *msm = new msm_t<g1_gpu::affine_element, fr_gpu, point_t>();
                    
                    // auto reference_string = std::make_shared<gpu_waffle::FileReferenceString>(NUM_POINTS, "../srs_db");
                    // g1_gpu::affine_element* points = reference_string->get_monomials();
                    // Context<bucket_t, point_t, scalar_t, affine_t> *context = msm->pippenger_initialize(points);

                    // g1_gpu::element *final_result_2 = msm->msm_bucket_method(context, item.mul_scalars, key->reference_string->get_monomials(), 1 << 10);
                    // exit(0);


                    auto runtime_state = barretenberg::scalar_multiplication::pippenger_runtime_state(key->small_domain.size + 1);
                    barretenberg::g1::affine_element result(barretenberg::scalar_multiplication::pippenger_unsafe(
                        item.mul_scalars,
                        key->reference_string->get_monomials(),
                        key->small_domain.size + 1,
                        runtime_state));

                    cout << "result is: " << result << endl;

                    transcript->add_element(item.tag, result.to_buffer());
                } else {
                    ASSERT(item.constant == MSMSize::N);
                    if (key->reference_string->get_size() < key->small_domain.size) {
                        info("Reference string too small for Pippenger.");
                    }
                    barretenberg::g1::affine_element result(barretenberg::scalar_multiplication::pippenger_unsafe(item.mul_scalars,
                        key->reference_string->get_monomials(),
                        key->small_domain.size,
                        key->pippenger_runtime_state));
                    transcript->add_element(item.tag, result.to_buffer());
                }
                break;
            }

            case WorkType::SMALL_FFT: {
                cout << "SMALL_FFT!" << endl;
                const size_t n = key->n;
                barretenberg::polynomial& wire = witness->wires.at(item.tag);
                barretenberg::polynomial& wire_fft = key->wire_ffts.at(item.tag + "_fft");
                barretenberg::polynomial wire_copy(wire, n);
                wire_copy.coset_fft_with_generator_shift(key->small_domain, item.constant);

                for (size_t i = 0; i < n; ++i) {
                    wire_fft[4 * i + item.index] = wire_copy[i];
                }
                wire_fft[4 * n + item.index] = wire_copy[0];
                break;
            }
            
            case WorkType::FFT: {
                cout << "FFT!" << endl;
                barretenberg::polynomial& wire = witness->wires.at(item.tag);
                barretenberg::polynomial& wire_fft = key->wire_ffts.at(item.tag + "_fft");
                barretenberg::polynomial_arithmetic::copy_polynomial(&wire[0], &wire_fft[0], key->n, 4 * key->n + 4);
                wire_fft.coset_fft(key->large_domain);
                wire_fft.add_lagrange_base_coefficient(wire_fft[0]);
                wire_fft.add_lagrange_base_coefficient(wire_fft[1]);
                wire_fft.add_lagrange_base_coefficient(wire_fft[2]);
                wire_fft.add_lagrange_base_coefficient(wire_fft[3]);
                break;
            }
            
            case WorkType::IFFT: {
                cout << "IFFT!" << endl;
                barretenberg::polynomial& wire = witness->wires.at(item.tag);
                wire.ifft(key->small_domain);
                break;
            }
            
            default: {
            }
        }
    }
    work_item_queue = std::vector<work_item>();
}   

/**
 * TODO: Connect MSM GPU kernel to queue wrapper
 * TODO: Change queue to interact with multiple streams
 * TODO: Change memory model from unified memory to pinned host memory
 */
