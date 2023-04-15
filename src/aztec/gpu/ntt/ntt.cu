#include "kernel.cu"

using namespace std;
using namespace ntt_common;

/**
 * Read scalars from scalar field
 */
template<class S>  
S *ntt_t<S>::read_scalars(S *scalars) {
    ifstream stream;
    // stream.open("../src/aztec/gpu/msm/points/scalars_copy.txt", ios::in);
    stream.open("../src/aztec/gpu/msm/points/scalars_copy.txt", ios::in);

    if ( stream.is_open() ) {   
        for (size_t i = 0; i < 32; i++) {
            for (size_t j = 0; j < 4; j++) {
                stream >> scalars[i].data[j];
            }
        }   
    }
    stream.close();
        
    return scalars;
}

/**
 * Top-level function for Cooley-Tuckey
 */
template<class S>  
void ntt_t<S>::ntt_execute(uint64_t n, bool inverse) {
    cout << "entered ntt_end2end!" << endl;

    // Allocate unified memory for scalars 
    // TODO: change to memcpy
    S *d_scalars;
    cudaMallocManaged(&d_scalars, POINTS * LIMBS * sizeof(uint64_t));
    read_scalars(d_scalars);

    uint64_t logn_size = uint64_t(log(n) / log(2));
    uint64_t n_twiddles = n; 
    S *d_twiddles;
    if (inverse) {
        d_twiddles = fill_twiddle_factors_array(n_twiddles, ntt_common::omega_inverse(logn_size));
    }
    else {
        d_twiddles = fill_twiddle_factors_array(n_twiddles, ntt_common::omega(logn_size));
    }

    cout << "print twiddle factors: " << endl;
    cout << d_twiddles[0].data[0] << endl;
    cout << d_twiddles[0].data[1] << endl;
    cout << d_twiddles[0].data[2] << endl;
    cout << d_twiddles[0].data[3] << endl;
    cout << endl;

    cout << "print scalars: " << endl;
    cout << d_scalars[0].data[0] << endl;
    cout << d_scalars[0].data[1] << endl;
    cout << d_scalars[0].data[2] << endl;
    cout << d_scalars[0].data[3] << endl;
    cout << endl;

    S *result = ntt_template(d_scalars, n, d_twiddles, n_twiddles, inverse);
}

/**
 * Fill twiddles array with twiddle factors. 
 */
template<class S> 
S *ntt_t<S>::fill_twiddle_factors_array(uint64_t n_twiddles, S omega) {   
    S *d_twiddles;
    cudaMallocManaged(&d_twiddles, n_twiddles * sizeof(S));
    twiddle_factors_kernel<<<1,4>>>(d_twiddles, n_twiddles, omega);
    cudaDeviceSynchronize();
    return d_twiddles;
}

/**
 * Cooley-Tuckey NTT
 */
template<class S> 
S *ntt_t<S>::ntt_template(S *arr, uint64_t n, S *d_twiddles, uint64_t n_twiddles, bool inverse) {
    cout << "entered ntt_template!" << endl;

    uint64_t logn_size = uint64_t(log(n) / log(2));
    size_t size_S = n * sizeof(S);

    // Need to check that this bit reversal works with how scalar elements are structured
    S *arr_reversed = template_reverse_order(arr, n, logn_size);

    cout << "print reversed array: " << endl;
    cout << arr_reversed[0].data[0] << endl;
    cout << arr_reversed[1].data[0] << endl;
    cout << arr_reversed[2].data[0] << endl;
    cout << arr_reversed[3].data[0] << endl;

    // Allocate unified memory for reversed scalars array 
    S *d_arr_reversed;
    size_t size_E = n * sizeof(S);
    cudaMalloc(&d_arr_reversed, size_E);
    cudaMemcpy(d_arr_reversed, arr_reversed, size_E, cudaMemcpyHostToDevice);

    // Call NTT function 
    template_ntt_on_device_memory(d_arr_reversed, n, logn_size, d_twiddles, n_twiddles);

    if (inverse == true) {
        int NUM_THREADS = MAX_NUM_THREADS;
        int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

        cout << "NUM_THREADS is: " << NUM_THREADS << endl;
        cout << "NUM_BLOCKS is: " << NUM_BLOCKS << endl;
        exit(0);

        // template_normalize_kernel<<<NUM_THREADS, NUM_BLOCKS>>>(d_arr_reversed, d_arr_reversed, n, S::inv_log_size(logn_size));
    }

    cudaMemcpy(arr_reversed, d_arr_reversed, size_E, cudaMemcpyDeviceToHost);
    cudaFree(d_arr_reversed);
    return arr_reversed;
}


/**
 * Bit-reversal ordering of original input array
 */
template<class S> 
S *ntt_t<S>::template_reverse_order(S *arr, uint64_t n, uint64_t logn) {
    S *reversedArray = new S[n];
    for (uint64_t i = 0; i < n; i++) {
        uint64_t reversed = reverseBits(i, logn);
        reversedArray[i] = arr[reversed];
    }
    return reversedArray;
}

/**
 * Perform the bit reversal 
 */
template<class S> 
uint64_t ntt_t<S>::reverseBits(uint64_t num, uint64_t logn) {
  unsigned int reverse_num = 0;
  int i;
  for (i = 0; i < logn; i++) {
    if ((num & (1 << i))) reverse_num |= 1 << ((logn - 1) - i);
  }
  return reverse_num;
}

template<class S> 
void ntt_t<S>::template_ntt_on_device_memory(S *d_arr, uint64_t n, uint64_t logn, S *d_twiddles, uint64_t n_twiddles) {
    uint64_t m = 2;
    for (uint64_t s = 0; s < logn; s++) {
        for (uint64_t i = 0; i < n; i += m) {
            int shifted_m = m >> 1;
            // need to look into why kernel configuration is only a single thread and block
            int number_of_threads = MAX_NUM_THREADS ^ ((shifted_m ^ MAX_NUM_THREADS) & -(shifted_m < MAX_NUM_THREADS));
            int number_of_blocks = shifted_m / MAX_NUM_THREADS + 1;

            cout << "number_of_threads is: " << number_of_threads << endl;
            cout << "number_of_blocks is: " << number_of_blocks << endl;
            exit(0);

            template_butterfly_kernel<<<number_of_threads, number_of_blocks>>>(d_arr, d_twiddles, n, n_twiddles, m, i, m >> 1);
        }
        m <<= 1;
    }
}
