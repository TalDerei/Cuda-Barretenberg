namespace pippenger_common {

/**
 * Wrapper for error handling
 */
#define CUDA_WRAPPER(expression) do {                                                       \
    cudaError_t status = expression;                                                        \
    if (status != cudaSuccess) {                                                            \
        cout << "CUDA error: " << cudaGetErrorString(status) << endl;                       \
        throw cudaGetErrorString(status);                                                   \
    }                                                                                       \
} while(0)

}