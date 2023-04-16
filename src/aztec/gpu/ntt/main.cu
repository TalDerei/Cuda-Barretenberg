#include "ntt.cu"

using namespace std;
using namespace ntt_common;

int main(int, char**) {
    // Define domain size
    uint64_t domain_size = 4;
    
    // Dynamically initialize new 'ntt_t' object
     ntt_t<fr_gpu> *ntt_object = new  ntt_t<fr_gpu>();

    // Initialize NTT with twiddle factors
    ntt_object->ntt_execute(domain_size, false);
}