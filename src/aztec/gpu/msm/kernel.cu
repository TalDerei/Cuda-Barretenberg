#include "common.cuh"
#include <cooperative_groups.h>
#include <cuda.h>

using namespace cooperative_groups;

namespace pippenger_common {

// // Wrapper around uint64_t with template parameter 'scalar_t'
// class scalar_T {
//     uint64_t val[sizeof(scalar_t) / sizeof(uint64_t)][WARP];

//     public:
//         __device__ uint64_t& operator[](size_t i) { 
//             return val[i][0]; 
//         }
//         __device__ const uint64_t& operator[](size_t i) const { 
//             return val[i][0]; 
//         }
//         // __device__ scalar_T& operator=(const scalar_t& rhs) {
//         //     for (size_t i = 0; i < sizeof(scalar_t) / sizeof(uint64_t); i++) {
//         //         val[i][0] = rhs[i];
//         //     }
//         //     return *this;
//         // }
// };

// // Wrapper around a pointer to an array of scalar_T values,
// // providing a way to index into array of scalars. The operator[] 
// // function takes an index i, which represents the index of the scalar 
// // value in the array, and returns a reference to the scalar_T value at that index.
// class scalars_T {
//     scalar_T* ptr;

//     public:
//         __device__ scalars_T(void* rhs) { 
//             ptr = (scalar_T*)rhs; 
//         }
//         __device__ scalar_T& operator[](size_t i) {   
//             return *(scalar_T*)&(&ptr[i / WARP][0])[i % WARP];   
//         }
//         __device__ const scalar_T& operator[](size_t i) const {   
//             return *(const scalar_T*)&(&ptr[i / WARP][0])[i % WARP];   
//         }
// };

// constexpr static __device__ int dlog2(int n) {   
//     int ret = 0; while (n >>= 1) ret++; return ret;   
// }

/**
 * Kernel function for "Pippenger's Bucket Method"
 */
__global__ void pippenger(
affine_t *points, size_t npoints, const scalar_t *scalars_, 
bucket_t(* buckets)[NWINS][1<<WBITS], bucket_t(* ret)[NWINS][NTHREADS][2]) {
    // // Assert number of threads per block, number of points, and number of points
    // assert(blockDim.x == NTHREADS);         
    // assert(gridDim.x == NWINS);
    // assert(npoints == (uint64_t)npoints);

    // // Divide input points and scalars between brid blocks along y-axis, 
    // // and round to nearest warp-size
    // if (gridDim.y > 1) {
    //     uint64_t delta = ((uint64_t)npoints + gridDim.y - 1) / gridDim.y;
    //     delta = (delta + WARP - 1) & (0U - WARP); 
    //     uint64_t off = delta * blockIdx.y;

    //     // Calculate offset into the points and scalars array for this grid block
    //     // based on delta and blockIdx.y. 
    //     points  += off;
    //     scalars_ += off;

    //     // Update value of npoints to reflect number of points this grid block should process.
    //     if (blockIdx.y == gridDim.y - 1) {
    //         npoints -= off;
    //     }
    //     else {
    //         npoints = delta;
    //     }
    // }

    // // Creates mutable copy of scalars_ array, allowing the algorithm
    // // to later sort the array in-place
    // scalars_T scalars = const_cast<scalar_t*>(scalars_);
    // const int NTHRBITS = dlog2(NTHREADS);
    // const uint64_t tid = threadIdx.x; 
    // const uint64_t bid = blockIdx.x;
    // const uint64_t bit0 = bid * WBITS;          // Computes the index of the first bucket that the current block will process
    // bucket_t* row = buckets[blockIdx.y][bid];   // Pointer to first bucket the current block will proccess

    // // Parameters for coperative groups
    // auto grp = fixnum::layout();
    // int subgroup = grp.meta_group_rank();
    // int subgroup_size = grp.meta_group_size();

    // #pragma unroll 1
    // for (uint64_t i = NTHREADS * bid + tid; i < npoints; i += NTHREADS * NWINS) {
    //     scalar_t s = scalars_[i];
    //     // convert from montgomery form
    //     fq_gpu::from_monty(s.data[i % 4], s.data[i % 4]);
        
    //     // load scalar back to scalars array
    //     fq_gpu::load(s.data[i % 4], scalars[i]);
    // }

    // // Sync to avoid race conditions when writing scalars back to array
    // grp.sync();
}

/* ----------------------------------------- Helper Kernel Functions ---------------------------------------------- */

/**
 * Convert affine to projective coordinates 
 */
__global__ void affine_to_projective(g1::affine_element *a_point, g1::projective_element *p_point, size_t npoints) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (size_t i = 0; i < npoints; i++) {
        fq_gpu::load(a_point[i].x.data[tid], p_point[i].x.data[tid]);
        fq_gpu::load(a_point[i].y.data[tid], p_point[i].y.data[tid]);
        fq_gpu::load(field_gpu<fq_gpu>::one().data[tid], p_point[0].z.data[tid]);
    }
}

/**
 * Convert affine to jacobian coordinates 
 */
__global__ void affine_to_jacobian(g1::affine_element *a_point, g1::element *j_point, size_t npoints) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = 0; i < npoints; i++) {
        fq_gpu::load(a_point[i].x.data[tid], j_point[i].x.data[tid]);
        fq_gpu::load(a_point[i].y.data[tid], j_point[i].y.data[tid]);
        fq_gpu::load(field_gpu<fq_gpu>::one().data[tid], j_point[i].z.data[tid]);
    }
}

/**
 * Compare group elements kernel
 */
__global__ void comparator_kernel(g1::element *point, g1::element *point_2, uint64_t *result) {     
    fq_gpu lhs_zz;
    fq_gpu lhs_zzz;
    fq_gpu rhs_zz;
    fq_gpu rhs_zzz;
    fq_gpu lhs_x;
    fq_gpu lhs_y;
    fq_gpu rhs_x;
    fq_gpu rhs_y;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    lhs_zz.data[tid] =  fq_gpu::square(point[0].z.data[tid], lhs_zz.data[tid]);
    lhs_zzz.data[tid] = fq_gpu::mul(lhs_zz.data[tid], point[0].z.data[tid], lhs_zzz.data[tid]);
    rhs_zz.data[tid] = fq_gpu::square(point_2[0].z.data[tid], rhs_zz.data[tid]);
    rhs_zzz.data[tid] = fq_gpu::mul(rhs_zz.data[tid], point_2[0].z.data[tid], rhs_zzz.data[tid]);
    lhs_x.data[tid] = fq_gpu::mul(point[0].x.data[tid], rhs_zz.data[tid], lhs_x.data[tid]);
    lhs_y.data[tid] = fq_gpu::mul(point[0].y.data[tid], rhs_zzz.data[tid], lhs_y.data[tid]);
    rhs_x.data[tid] = fq_gpu::mul(point_2[0].x.data[tid], lhs_zz.data[tid], rhs_x.data[tid]);
    rhs_y.data[tid] = fq_gpu::mul(point_2[0].y.data[tid], lhs_zzz.data[tid], rhs_y.data[tid]);
    result[tid] = ((lhs_x.data[tid] == rhs_x.data[tid]) && (lhs_y.data[tid] == rhs_y.data[tid]));
}

/* ----------------------------------------- Naive MSM Functions ---------------------------------------------- */

/**
 * Naive double and add using sequential implementation 
 */
__global__ void simple_msm_naive(g1::element *point, fr_gpu *scalar, fq_gpu *result, g1::element *result_vec, size_t npoints) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    // 3 * Fq field multiplications
    fq_gpu::mul(point[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4], 
                scalar[(subgroup + (subgroup_size * blockIdx.x))].data[tid % 4], 
                result_vec[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]);
    fq_gpu::mul(point[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4], 
                scalar[(subgroup + (subgroup_size * blockIdx.x))].data[tid % 4], 
                result_vec[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]);
    fq_gpu::mul(point[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4], 
                scalar[(subgroup + (subgroup_size * blockIdx.x))].data[tid % 4], 
                result_vec[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);
}

/**
 * Sum reduction with shared memory 
 */
__global__ void sum_reduction(g1::element *v, g1::element *result) {     
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction in shared memory
    __shared__ g1::element partial_sum[256];

    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    // Print cooperative group metadata
    // printf("thread_block my_block = this_thread_block(): %d\n", grp.meta_group_rank());
    // printf("size of the group: %d\n", grp.meta_group_size());

    // Each thread loads two points into shared memory
    fq_gpu::load(v[(subgroup + (subgroup_size * blockIdx.x)) * 2].x.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x)) * 2].x.data[tid % 4]);
    fq_gpu::load(v[(subgroup + (subgroup_size * blockIdx.x)) * 2].y.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x)) * 2].y.data[tid % 4]);
    fq_gpu::load(v[(subgroup + (subgroup_size * blockIdx.x)) * 2].z.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x)) * 2].z.data[tid % 4]);

    fq_gpu::load(v[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].x.data[tid % 4], 
                partial_sum[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].x.data[tid % 4]);
    fq_gpu::load(v[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].y.data[tid % 4], 
                partial_sum[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].y.data[tid % 4]);
    fq_gpu::load(v[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].z.data[tid % 4], 
                partial_sum[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].z.data[tid % 4]);

    // Sychronization barrier after loading elements
    __syncthreads();

    /*
        For 2^10 (1024) points, the unrolled loop iteration is as follows:
            * First pass -- 1024 threads --> 512 points to 256 points
            * Second pass -- 512 threads --> 256 points to 128 points
            * Third pass -- 256 threads --> 128 points to 64 points
            * Fourth pass -- 128 threads --> 64 points to 32 points
            * Fifth pass -- 64 threads --> 32 points to 16 points
            * Sixth pass -- 32 threads --> 16 points to 8 points
            * Seventh pass -- 16 threads --> 8 points to 4 points
            * Eighth pass -- 8 threads --> 4 points to 2 points
            * Ninth pass -- 4 threads --> 2 points to 1 point
    */
    
    int t = blockDim.x;
    for (int s = 0; s < 2; s++) {
        if (threadIdx.x < t)
            g1::add(
                partial_sum[(subgroup + (subgroup_size * blockIdx.x)) * 2].x.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x)) * 2].y.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x)) * 2].z.data[tid % 4], 
                partial_sum[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].x.data[tid % 4], 
                partial_sum[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].y.data[tid % 4], 
                partial_sum[((subgroup + (subgroup_size * blockIdx.x)) * 2) + 1].z.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4], 
                partial_sum[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]
            );
        __syncthreads();
        t -= t / 2;
    }

    // Global synchronization directive -- might not be neccessary
    grp.sync();

    // Accumulate result into current block
    if (threadIdx.x < 4)
        fq_gpu::load(partial_sum[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4], 
                    result[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]);
        fq_gpu::load(partial_sum[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4], 
                    result[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]);
        fq_gpu::load(partial_sum[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4], 
                    result[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);
}

/**
 * Sum reduction accumulation
 */
__global__ void sum_reduction_accumulate(g1::element *v, g1::element *v_r) {     
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert results from montgomery form
    fq_gpu::from_monty(v[0].x.data[tid % 4], v_r[0].x.data[tid % 4]);
    fq_gpu::from_monty(v[0].y.data[tid % 4], v_r[0].y.data[tid % 4]);
    fq_gpu::from_monty(v[0].z.data[tid % 4], v_r[0].z.data[tid % 4]);
}

/* ----------------------------------------- Pippenger's "Bucket Method" MSM Functions ---------------------------------------------- */

/**
 * Initialize buckets kernel for large MSM
 */
__global__ void initialize_buckets_kernel(g1::element *bucket) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].x.data[tid % 4]);
    fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].y.data[tid % 4]);
    fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup + (subgroup_size * blockIdx.x)].z.data[tid % 4]);
}

/**
 * Scalar digit decomposition 
 */
__device__ uint64_t decompose_scalar_digit(fr_gpu scalar, unsigned num, unsigned width) {    
    // Determine which 64-bit limb to access 
    const uint64_t limb_lsb_idx = (num * width) / 64;  
    const uint64_t shift_bits = (num * width) % 64;  

    // Shift limb to right to extract scalar digit
    uint64_t rv = scalar.data[limb_lsb_idx] >> shift_bits; 

    // Check if scalar digit crosses boundry of current limb
    if ((shift_bits + width > 64) && (limb_lsb_idx + 1 < 4)) {
        rv += scalar.data[limb_lsb_idx + 1] << (64 - shift_bits);
    }
    // Bit mask to extract LSB of size width
    rv &= ((1 << width) - 1);
    
    return rv;
}

/**
 * Decompose b-bit scalar into c-bit scalar, where c <= b
 */
__global__ void split_scalars_kernel
(unsigned *bucket_indices, unsigned *point_indices, fr_gpu *scalars, unsigned npoints, unsigned num_bucket_modules, unsigned c) {         
    unsigned bucket_index;
    unsigned current_index;
    fr_gpu scalar;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < npoints) {
        for (int i = 0; i < num_bucket_modules; i++) {
            bucket_index = decompose_scalar_digit(scalars[tid], i, c);
            current_index = i * npoints + tid; 
            
            // Bitwise performs addition here -- packing information about bucket module and specific bucket index
            bucket_indices[current_index] = (i << c) | bucket_index; 
            point_indices[current_index] = tid;
        }
    }   
}

/**
 * Accumulation kernel adds up points in each bucket
 */
__global__ void accumulate_buckets_kernel
(g1::element *buckets, unsigned *bucket_offsets, unsigned *bucket_sizes, unsigned *single_bucket_indices, 
unsigned *point_indices, g1::element *points, unsigned num_buckets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    // Stores the indices, sizes, and offsets of the buckets and points
    unsigned bucket_index = single_bucket_indices[(subgroup + (subgroup_size * blockIdx.x))];
    unsigned bucket_size = bucket_sizes[(subgroup + (subgroup_size * blockIdx.x))];
    unsigned bucket_offset = bucket_offsets[(subgroup + (subgroup_size * blockIdx.x))];

    // Sync loads
    grp.sync();

    // If bucket is empty, return
    if (bucket_size == 0) { 
        return;
    }

    // Add points starting from the relevant offset up to the bucket size
    for (unsigned i = 0; i < bucket_sizes[(subgroup + (subgroup_size * blockIdx.x))]; i++) { 
        g1::add(
            buckets[bucket_index].x.data[tid % 4], 
            buckets[bucket_index].y.data[tid % 4], 
            buckets[bucket_index].z.data[tid % 4], 
            points[point_indices[bucket_offset + i]].x.data[tid % 4], 
            points[point_indices[bucket_offset + i]].y.data[tid % 4], 
            points[point_indices[bucket_offset + i]].z.data[tid % 4], 
            buckets[bucket_index].x.data[tid % 4], 
            buckets[bucket_index].y.data[tid % 4], 
            buckets[bucket_index].z.data[tid % 4]
        );
    }
}

/**
 * Sum reduction kernel that accumulates bucket sums in bucket modules
 */
__global__ void bucket_module_sum_reduction_kernel(g1::element *buckets, g1::element *final_result, size_t num_buckets, unsigned c) {
    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    g1::element line_sum;

    fq_gpu::load(buckets[((subgroup + (subgroup_size * blockIdx.x)) + 1) * (1 << c) - 1].x.data[tid % 4], line_sum.x.data[tid % 4]);
    fq_gpu::load(buckets[((subgroup + (subgroup_size * blockIdx.x)) + 1) * (1 << c) - 1].y.data[tid % 4], line_sum.y.data[tid % 4]);
    fq_gpu::load(buckets[((subgroup + (subgroup_size * blockIdx.x)) + 1) * (1 << c) - 1].z.data[tid % 4], line_sum.z.data[tid % 4]);

    fq_gpu::load(line_sum.x.data[tid % 4], final_result[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4]);
    fq_gpu::load(line_sum.y.data[tid % 4], final_result[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4]);
    fq_gpu::load(line_sum.z.data[tid % 4], final_result[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]);

    for (unsigned i = (1 << c) - 2; i > 0; i--) {
        // Running sum method
        g1::add(
            line_sum.x.data[tid % 4], 
            line_sum.y.data[tid % 4], 
            line_sum.z.data[tid % 4], 
            buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << c) + i].x.data[tid % 4], 
            buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << c) + i].y.data[tid % 4], 
            buckets[(subgroup + (subgroup_size * blockIdx.x)) * (1 << c) + i].z.data[tid % 4], 
            line_sum.x.data[tid % 4], 
            line_sum.y.data[tid % 4], 
            line_sum.z.data[tid % 4]
        );

        g1::add(
            final_result[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4], 
            final_result[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4], 
            final_result[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4], 
            line_sum.x.data[tid % 4], 
            line_sum.y.data[tid % 4], 
            line_sum.z.data[tid % 4],
            final_result[(subgroup + (subgroup_size * blockIdx.x))].x.data[tid % 4], 
            final_result[(subgroup + (subgroup_size * blockIdx.x))].y.data[tid % 4], 
            final_result[(subgroup + (subgroup_size * blockIdx.x))].z.data[tid % 4]
        );
    }
}

/**
 * Final bucket accumulation to produce single group element
 */
__global__ void final_accumulation_kernel(g1::element *final_result, g1::element *res, size_t num_bucket_modules, unsigned c) {
    g1::element R;
    g1::element Q;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t digit_base = {unsigned(1 << c)};

    fq_gpu res_x_temp{ 0, 0, 0, 0 };
    fq_gpu res_y_temp{ 0, 0, 0, 0 };
    fq_gpu res_z_temp{ 0, 0, 0, 0 };

    fq_gpu::load(res_x_temp.data[tid], res[0].x.data[tid]);
    fq_gpu::load(res_y_temp.data[tid], res[0].y.data[tid]);
    fq_gpu::load(res_z_temp.data[tid], res[0].z.data[tid]);

    // Double and add implementation using bit-decomposition for each bucket module with time complexity: O(k)
    for (unsigned i = num_bucket_modules; i > 0; i--) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.x.data[tid]); 
        fq_gpu::load(0, R.y.data[tid]); 
        fq_gpu::load(0, R.z.data[tid]); 

        fq_gpu::load(res[0].x.data[tid], Q.x.data[tid]);
        fq_gpu::load(res[0].y.data[tid], Q.y.data[tid]);
        fq_gpu::load(res[0].z.data[tid], Q.z.data[tid]);

        // Traverses the bits of the scalar from MSB to LSB and extracts the i-th bit of scalar in limb
        for (int i = 253; i >= 0; i--) {
            if (((digit_base >> i) & 1) ? 1 : 0)
                g1::add(
                    R.x.data[tid], R.y.data[tid], R.z.data[tid], 
                    Q.x.data[tid], Q.y.data[tid], Q.z.data[tid], 
                    R.x.data[tid], R.y.data[tid], R.z.data[tid]
                );
            if (i != 0) 
                g1::doubling(
                    R.x.data[tid], R.y.data[tid], R.z.data[tid], 
                    R.x.data[tid], R.y.data[tid], R.z.data[tid]
                );
        }

        g1::add(
            R.x.data[tid], 
            R.x.data[tid], 
            R.x.data[tid],
            final_result[i - 1].x.data[tid],
            final_result[i - 1].y.data[tid],
            final_result[i - 1].z.data[tid],
            res[0].x.data[tid], 
            res[0].y.data[tid], 
            res[0].z.data[tid]
        );
    }
}

}