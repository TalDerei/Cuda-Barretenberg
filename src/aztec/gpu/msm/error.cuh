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

/**
 * Macro that checks for errors in runtime API
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

}