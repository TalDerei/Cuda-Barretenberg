#include "fq.cuh"
#include <ecc/curves/grumpkin/grumpkin.hpp>

using namespace gpu_barretenberg;
using namespace std;

// Global functions are also called "kernels". It's the functions that you may call from the host side 
// using CUDA kernel call semantics (<<<...>>>). Device functions can only be called from other device or 
// global functions. __device__ functions cannot be called from host code.
__global__ void cuda_hello() {
    printf("Hello World from GPU!\n");
}

template< int fn_bytes, typename word_fixnum >
void test() {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    printf("test!\n");
}

int main(int, char**) {
    test<ELT_LIMBS, u64_fixnum>();

    // CUDA kernel launches are asynchronous, but all GPU-related tasks placed in one stream are executed sequentially. 
    // Calling 'cudaDeviceSynchronize' ensures the kernel finishes and flushes output buffer.
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
}