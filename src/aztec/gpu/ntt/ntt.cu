#include "ntt.cuh"
#include <iostream>

using namespace std;

namespace ntt_common {

/**
 * Top-level function for Cooley-Tuckey
 */
template<class S>  
int ntt_t<S>::ntt_end2end(fr_gpu *arr, uint64_t n, bool inverse) {
    uint64_t logn = uint64_t(log(n) / log(2));
    uint64_t n_twiddles = n; // n_twiddles is set to 4096 as scalar_t::omega() is of that order
    S *d_twiddles;
    // if (inverse) {
    //     d_twiddles = fill_twiddle_factors_array(n_twiddles);
    // }
}

template<class S> 
fr_gpu *ntt_t<S>::fill_twiddle_factors_array(uint64_t n_twiddles, fr_gpu omega) {
  
}

template<class S> 
fr_gpu *ntt_t<S>::ntt_template(fr_gpu *arr, uint64_t n, fr_gpu *d_twiddles, uint64_t n_twiddles, bool inverse) {

}

}