#include "common.cuh"
#include <cooperative_groups.h>
#include <cuda.h>

using namespace cooperative_groups;

namespace pippenger_common {

/**
 * Kernel function for "Pippenger's Bucket Method"
 */
__global__ void pippenger(
affine_t *points, size_t npoints, const scalar_t *scalars, 
bucket_t(* buckets)[NWINS][1<<WBITS], bucket_t(* ret)[NWINS][NTHREADS][2]) {
    
}

/**
 * Naive double and add using sequential implementation 
 */
__global__ void simple_msm_naive(
g1::element *point, fr_gpu *scalar, g1::element *result, size_t npoints) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    g1::element temp;
    g1::element temp_accumulator;
    fq_gpu res_x_temp{ 0, 0, 0, 0 };
    fq_gpu res_y_temp{ 0, 0, 0, 0 };
    fq_gpu res_z_temp{ 0, 0, 0, 0 };

    fq_gpu::load(res_x_temp.data[tid], temp.x.data[tid]);
    fq_gpu::load(res_y_temp.data[tid], temp.y.data[tid]);
    fq_gpu::load(res_z_temp.data[tid], temp.z.data[tid]);
    fq_gpu::load(res_x_temp.data[tid], temp_accumulator.x.data[tid]);
    fq_gpu::load(res_y_temp.data[tid], temp_accumulator.y.data[tid]);
    fq_gpu::load(res_z_temp.data[tid], temp_accumulator.z.data[tid]);

    for (int i = 0; i < 1024; i++) {
        fq_gpu::mul(point[i].x.data[tid], scalar[i].data[tid], temp.x.data[tid]);
        fq_gpu::mul(point[i].y.data[tid], scalar[i].data[tid], temp.y.data[tid]);
        fq_gpu::mul(point[i].z.data[tid], scalar[i].data[tid], temp.z.data[tid]);
        fq_gpu::add(temp.x.data[tid], temp_accumulator.x.data[tid], temp_accumulator.x.data[tid]);
        fq_gpu::add(temp.y.data[tid], temp_accumulator.y.data[tid], temp_accumulator.y.data[tid]);
        fq_gpu::add(temp.z.data[tid], temp_accumulator.z.data[tid], temp_accumulator.z.data[tid]);
    }
    
    fq_gpu::load(temp_accumulator.x.data[tid], result[0].x.data[tid]);
    fq_gpu::load(temp_accumulator.y.data[tid], result[0].y.data[tid]);
    fq_gpu::load(temp_accumulator.z.data[tid], result[0].z.data[tid]);

    fq_gpu::from_monty(result[0].x.data[tid], result[0].x.data[tid]);
    fq_gpu::from_monty(result[0].y.data[tid], result[0].y.data[tid]);
    fq_gpu::from_monty(result[0].z.data[tid], result[0].z.data[tid]);   
}

/**
 * Naive double and add using sequential implementation 
 */
__global__ void simple_msm_naive_2(g1::element *point, fr_gpu *scalar, fq_gpu *result, g1::element *result_vec, size_t npoints) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();

    // 3 * Fq field multiplications
    fq_gpu::mul(point[subgroup].x.data[tid % 4], scalar[subgroup].data[tid % 4], result_vec[subgroup].x.data[tid % 4]);
    fq_gpu::mul(point[subgroup].y.data[tid % 4], scalar[subgroup].data[tid % 4], result_vec[subgroup].y.data[tid % 4]);
    fq_gpu::mul(point[subgroup].z.data[tid % 4], scalar[subgroup].data[tid % 4], result_vec[subgroup].z.data[tid % 4]);
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
    fq_gpu::load(v[subgroup * 2].x.data[tid % 4], partial_sum[subgroup * 2].x.data[tid % 4]);
    fq_gpu::load(v[subgroup * 2].y.data[tid % 4], partial_sum[subgroup * 2].y.data[tid % 4]);
    fq_gpu::load(v[subgroup * 2].z.data[tid % 4], partial_sum[subgroup * 2].z.data[tid % 4]);

    fq_gpu::load(v[(subgroup * 2) + 1].x.data[tid % 4], partial_sum[(subgroup * 2) + 1].x.data[tid % 4]);
    fq_gpu::load(v[(subgroup * 2) + 1].y.data[tid % 4], partial_sum[(subgroup * 2) + 1].y.data[tid % 4]);
    fq_gpu::load(v[(subgroup * 2) + 1].z.data[tid % 4], partial_sum[(subgroup * 2) + 1].z.data[tid % 4]);

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
                partial_sum[subgroup * 2].x.data[tid % 4], 
                partial_sum[subgroup * 2].y.data[tid % 4], 
                partial_sum[subgroup * 2].z.data[tid % 4], 
                partial_sum[(subgroup * 2) + 1].x.data[tid % 4], 
                partial_sum[(subgroup * 2) + 1].y.data[tid % 4], 
                partial_sum[(subgroup * 2) + 1].z.data[tid % 4], 
                partial_sum[subgroup].x.data[tid % 4], 
                partial_sum[subgroup].y.data[tid % 4], 
                partial_sum[subgroup].z.data[tid % 4]
            );
        __syncthreads();
        t -= t / 2;
    }

    // Global synchronization directive -- might not be neccessary
    grp.sync();

    // Accumulate result into current block
    if (threadIdx.x < 4)
        fq_gpu::load(partial_sum[subgroup].x.data[tid % 4], result[subgroup].x.data[tid % 4]);
        fq_gpu::load(partial_sum[subgroup].y.data[tid % 4], result[subgroup].y.data[tid % 4]);
        fq_gpu::load(partial_sum[subgroup].z.data[tid % 4], result[subgroup].z.data[tid % 4]);
}

/**
 * Sum reduction accumulation
 */
__global__ void sum_reduction_accumulate(g1::element *v, g1::element *v_r) {     
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert from montgomery form
    fq_gpu::from_monty(v[0].x.data[tid % 4], v_r[0].x.data[tid % 4]);
    fq_gpu::from_monty(v[0].y.data[tid % 4], v_r[0].y.data[tid % 4]);
    fq_gpu::from_monty(v[0].z.data[tid % 4], v_r[0].z.data[tid % 4]);
}

__global__ void affine_to_projective(g1::affine_element *a_point, g1::projective_element *p_point, size_t npoints) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (size_t i = 0; i < npoints; i++) {
        fq_gpu::load(a_point[i].x.data[tid], p_point[i].x.data[tid]);
        fq_gpu::load(a_point[i].y.data[tid], p_point[i].y.data[tid]);
        fq_gpu::load(field_gpu<fq_gpu>::one().data[tid], p_point[0].z.data[tid]);
    }
}

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

/**
 * Initialize buckets kernel for large MSM
 */
__global__ void initialize_buckets_kernel(g1::element *bucket) {     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();

    // Initialize buckets with zero points
    fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup].x.data[tid % 4]);
    fq_gpu::load(fq_gpu::one().data[tid % 4], bucket[subgroup].y.data[tid % 4]);
    fq_gpu::load(fq_gpu::zero().data[tid % 4], bucket[subgroup].z.data[tid % 4]);
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
        // Access next limb and left shift by '32 - shift_bits' bits
        rv += scalar.data[limb_lsb_idx + 1] << (32 - shift_bits);
    }
    rv &= ((1 << width) - 1);
    // printf("rv is: %d\n", rv); 

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
            // Need to check if the decomposition is correct, i.e. if each thread can handle it's own scalar
            bucket_index = decompose_scalar_digit(scalars[tid], i, c);
            current_index = i * npoints + tid;
            bucket_indices[current_index] = (i << c) | bucket_index;
            printf("rv is: %d\n", (i << c) | bucket_index); 
            point_indices[current_index] = tid;
        }
    }   
}

}