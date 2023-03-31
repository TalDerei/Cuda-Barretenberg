#include <iostream>
#include "group.cu"

using namespace std;

namespace ntt_common {

/**
 * Cooley-Tuckey NTT function decleration 
 */
template < typename S >
class ntt_t {
    public: 
        ntt_t() {}
        
        ntt_end2end(fr_gpu *arr, uint64_t n, bool inverse);
        
        fr_gpu *fill_twiddle_factors_array(uint64_t n_twiddles, fr_gpu omega);

        fr_gpu *ntt_template(fr_gpu *arr, uint64_t n, fr_gpu *d_twiddles, uint64_t n_twiddles, bool inverse);
};

}