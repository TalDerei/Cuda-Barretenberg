#include "ntt.cuh"

/**
 * Calculate twiddle factors kernel
 */
__global__ void twiddle_factors_kernel(fr_gpu *d_twiddles, uint64_t n_twiddles, fr_gpu omega) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize twiddle factors to zero
    for (uint64_t i = 0; i < n_twiddles; i++) {
        fr_gpu::load(fr_gpu::zero().data[tid], d_twiddles[0].data[tid]);
    }

    fr_gpu::load(fr_gpu::one().data[tid], d_twiddles[0].data[tid]);

    // Calculate twiddle factors by multiplying by omega
    for (uint64_t i = 0; i < n_twiddles - 1; i++) {
        fr_gpu::mul(omega.data[tid], d_twiddles[i].data[tid], d_twiddles[i + 1].data[tid]); 
    }
}

/**
 * Cooley-Tuckey butterfly kernel
 */
__global__ void template_butterfly_kernel(
fr_gpu *arr, fr_gpu *twiddles, uint64_t n, uint64_t n_twiddles, uint64_t m, uint64_t i, uint64_t max_thread_num) {
    // need to fix this
    // Need to add subgroup using cooperative groups 
    fr_gpu v;
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (j < max_thread_num) {
        uint64_t g = j * (n / m);
        uint64_t k = i + j + (m >> 1);
        fr_gpu u = arr[i + j];
        fr_gpu::mul(twiddles[g * n_twiddles / n].data[j % 4], arr[k].data[j % 4], v.data[j % 4]); 
        fr_gpu::add(u.data[j % 4], v.data[j % 4], arr[i + j].data[j % 4]); 
        fr_gpu::sub(u.data[j % 4], v.data[j % 4], arr[k].data[j % 4]); 
    }
}

/**
 * Set the elements of arr to be the elements of res multiplied by some scalar
 */
__global__ void template_normalize_kernel(fr_gpu *arr, fr_gpu *res, uint64_t n, fr_gpu scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n) {
        fr_gpu::mul(scalar.data[tid], arr[0].data[tid], res[0].data[tid]); 
    }
}