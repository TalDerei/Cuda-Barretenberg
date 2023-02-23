#include <cooperative_groups.h>
#include "common.cuh"

namespace pippenger_common {

/**
 * Kernel function for "Pippenger's Bucket Method"
 */
__global__ void pippenger(
const point_t* points, size_t npoints, const scalar_t *scalars_, 
bucket_t(* buckets)[NWINS][1<<WBITS], bucket_t(* ret)[NWINS][NTHREADS][2]) {

}

}