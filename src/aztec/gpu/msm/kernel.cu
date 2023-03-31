#include "common.cuh"
#include <cooperative_groups.h>
#include <cuda.h>

using namespace cooperative_groups;

namespace pippenger_common {

/* ----------------------------------------- Naive MSM Functions ---------------------------------------------- */

/**
 * Kernel function for "Pippenger's Bucket Method"
 */
__global__ void pippenger(
affine_t *points, size_t npoints, const scalar_t *scalars_, bucket_t(* buckets)[NWINS][1<<WBITS], 
bucket_t(* ret)[NWINS][NTHREADS][2], g1::element *final_result) {

}

/**
 * Naive double and add using sequential implementation 
 */
__global__ void msm_naive_kernel(g1::element *point, fr_gpu *scalar, g1::element *result_vec, size_t npoints) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    // 3 * Fq field multiplications
    // not correct 
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
__global__ void sum_reduction(g1::element *points, g1::element *result) {     
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform reduction in shared memory
    // is this shared across grid blocks?
    __shared__ g1::element partial_sum[128]; // change size of shared memory -- allocate less for fewer points (128)

    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x)].x.data[tid % 4], 
                partial_sum[subgroup * 2].x.data[tid % 4]);
    fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x)].y.data[tid % 4], 
                partial_sum[subgroup * 2].y.data[tid % 4]);
    fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x)].z.data[tid % 4], 
                partial_sum[subgroup * 2].z.data[tid % 4]);

    fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x) + 1].x.data[tid % 4], 
                partial_sum[(subgroup * 2) + 1].x.data[tid % 4]);
    fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x) + 1].y.data[tid % 4], 
                partial_sum[(subgroup * 2) + 1].y.data[tid % 4]);
    fq_gpu::load(points[(subgroup * 2) + ((2 * subgroup_size) * blockIdx.x) + 1].z.data[tid % 4], 
                partial_sum[(subgroup * 2) + 1].z.data[tid % 4]);

    // Local sync barrier for load operations
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

    // #pragma unroll 7
    int t = blockDim.x;
    // we can reduce this number to 6 by launching more blocks
    // this loop will need to be chanmged potentially depending on how we split up 
    // the bucket segmentation problem
    for (int s = 0; s < log2f(blockDim.x) - 1; s++) {
        if (threadIdx.x < t) {
            g1::add(
                // This indexing is not correct!
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
        }
        __syncthreads();
        t -= t / 2;
    }

    // Global synchronization directive for entire grid -- need this?
    grp.sync();

    // Perform intermediary load to scratch -- moving data from shared to global memory
    if (threadIdx.x < 4) {
        fq_gpu::load(partial_sum[subgroup].x.data[tid % 4], result[blockIdx.x].x.data[tid % 4]);
        fq_gpu::load(partial_sum[subgroup].y.data[tid % 4], result[blockIdx.x].y.data[tid % 4]);
        fq_gpu::load(partial_sum[subgroup].z.data[tid % 4], result[blockIdx.x].z.data[tid % 4]); 
    }    
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

    // if (tid < npoints) {
        for (int i = 0; i < num_bucket_modules; i++) {
            bucket_index = decompose_scalar_digit(scalars[tid], i, c);
            current_index = i * npoints + tid; 
            
            // Bitwise performs addition here -- packing information about bucket module and specific bucket index
            bucket_indices[current_index] = (i << c) | bucket_index; 
            point_indices[current_index] = tid;
        }
    // }   
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

    // MY ALGORITHM IS COMPLETELY INCORRECT. THIS IS A SIMPLY REDUCTION -- NEED TO
    // DO THIS REDUCTION WITH RESPECT TO THE PARTIAL SUMS ASSOCIATED WITH EACH BUCKET
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
 * Sum reduction kernel that accumulates bucket sums
 * References PipeMSM (Algorithm 2) -- https://eprint.iacr.org/2022/999.pdf
 */
__global__ void bucket_module_sum_reduction_kernel_1(g1::element *buckets, g1::element *S_, g1::element *G_, unsigned M, unsigned U) {     
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Parameters for coperative groups
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    // Define variables
    g1::element G;
    g1::element S;

    // Initialize G and S with 0
    fq_gpu::load(0x0, G.x.data[tid % 4]);
    fq_gpu::load(0x0, G.y.data[tid % 4]);
    fq_gpu::load(0x0, G.z.data[tid % 4]);
    
    fq_gpu::load(0x0, S.x.data[tid % 4]);
    fq_gpu::load(0x0, S.y.data[tid % 4]);
    fq_gpu::load(0x0, S.z.data[tid % 4]);
    
     // Sync loads
    __syncthreads();
        
    // Each of the M segment sums (of size U each) can be computed seperately
    // Each of the windows K (of size 2^c each) can be computed seperately
    // Handle M and K in the kernel configurations
    // This loop goes from 128 to 1... need to index based on cooperative groups 
    // The issue with this kernel is you have 208 (26 * 8) blocks with a single thre
    // each running this workload
    // add loop unroll here
    // 2.0
    for (unsigned u = U - 1; u < U; u--) { 
        g1::add(
            G.x.data[tid % 4],
            G.y.data[tid % 4],
            G.z.data[tid % 4],
            S.x.data[tid % 4],
            S.y.data[tid % 4],
            S.z.data[tid % 4], 
            G.x.data[tid % 4],
            G.y.data[tid % 4],
            G.z.data[tid % 4]
        );

        g1::add(
            S.x.data[tid % 4],
            S.y.data[tid % 4],
            S.z.data[tid % 4], 
            buckets[blockIdx.x * (1 << (M - 1)) + u].x.data[tid % 4],
            buckets[blockIdx.x * (1 << (M - 1)) + u].y.data[tid % 4],
            buckets[blockIdx.x * (1 << (M - 1)) + u].z.data[tid % 4],
            S.x.data[tid % 4],
            S.y.data[tid % 4],
            S.z.data[tid % 4]
        );

        // comment out conditional code
        if (fq_gpu::is_zero(G.x.data[tid % 4]) && fq_gpu::is_zero(G.y.data[tid % 4]) && fq_gpu::is_zero(G.z.data[tid % 4])) {
            g1::doubling(
                buckets[blockIdx.x * (1 << (M - 1)) + u].x.data[tid % 4],
                buckets[blockIdx.x * (1 << (M - 1)) + u].y.data[tid % 4],
                buckets[blockIdx.x * (1 << (M - 1)) + u].z.data[tid % 4],
                G.x.data[tid % 4],
                G.y.data[tid % 4],
                G.z.data[tid % 4]
            );
        }
    }

    fq_gpu::load(S.x.data[tid % 4], S_[blockIdx.x].x.data[tid % 4]);
    fq_gpu::load(S.y.data[tid % 4], S_[blockIdx.x].y.data[tid % 4]);
    fq_gpu::load(S.z.data[tid % 4], S_[blockIdx.x].z.data[tid % 4]);
    fq_gpu::load(G.x.data[tid % 4], G_[blockIdx.x].x.data[tid % 4]);
    fq_gpu::load(G.y.data[tid % 4], G_[blockIdx.x].y.data[tid % 4]);
    fq_gpu::load(G.z.data[tid % 4], G_[blockIdx.x].z.data[tid % 4]);
}

__global__ void bucket_module_sum_reduction_kernel_2(g1::element *result, g1::element *S_, g1::element *G_, unsigned M, unsigned U) {     
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Define variables
    g1::element S;
    g1::element S_k;

    // Initialize G and S with 0
    fq_gpu::load(0x0, S.x.data[tid % 4]);
    fq_gpu::load(0x0, S.y.data[tid % 4]);
    fq_gpu::load(0x0, S.z.data[tid % 4]);
    
    fq_gpu::load(0x0, S_k.x.data[tid % 4]);
    fq_gpu::load(0x0, S_k.y.data[tid % 4]);
    fq_gpu::load(0x0, S_k.z.data[tid % 4]);

    // Sync loads
    __syncthreads();

    // Add up each segement M for each window K
    // change indexing and variable names
    // 2.1
    // need to fix the indexing here
    for (unsigned m = 0; m < M - 1; m++) {  
        g1::add(
            S_[blockIdx.x * (M - 1) + m].x.data[tid % 4],
            S_[blockIdx.x * (M - 1) + m].y.data[tid % 4],
            S_[blockIdx.x * (M - 1) + m].z.data[tid % 4],
            S.x.data[tid % 4],
            S.y.data[tid % 4],
            S.z.data[tid % 4], 
            S.x.data[tid % 4],
            S.y.data[tid % 4],
            S.z.data[tid % 4]
        );

        g1::add(
            S.x.data[tid % 4],
            S.y.data[tid % 4],
            S.z.data[tid % 4],
            S_k.x.data[tid % 4],
            S_k.y.data[tid % 4],
            S_k.z.data[tid % 4], 
            S_k.x.data[tid % 4],
            S_k.y.data[tid % 4],
            S_k.z.data[tid % 4]
        );
    }

    __syncthreads();

    // 2.2
    unsigned v = log2f(U);
    for (unsigned m = 0; m < v; m++) {  
        g1::doubling(
            S_k.x.data[tid % 4],
            S_k.y.data[tid % 4],
            S_k.z.data[tid % 4], 
            S_k.x.data[tid % 4],
            S_k.y.data[tid % 4],
            S_k.z.data[tid % 4]
        );
    }

    __syncthreads();
  
    g1::element G_k;

    // Initialize G and S with 0
    fq_gpu::load(0x0, G_k.x.data[tid % 4]);
    fq_gpu::load(0x0, G_k.y.data[tid % 4]);
    fq_gpu::load(0x0, G_k.z.data[tid % 4]);

    // 2.3
    // need to fix the indexing here
    for (unsigned m = 0; m < M; m++) {  
        g1::add(
            G_k.x.data[tid % 4],
            G_k.y.data[tid % 4],
            G_k.z.data[tid % 4],
            G_[blockIdx.x * M + m].x.data[tid % 4],
            G_[blockIdx.x * M + m].y.data[tid % 4],
            G_[blockIdx.x * M + m].z.data[tid % 4], 
            G_k.x.data[tid % 4],
            G_k.y.data[tid % 4],
            G_k.z.data[tid % 4]
        );
    }

    __syncthreads();

    // 2.4
    g1::add(
        G_k.x.data[tid % 4],
        G_k.y.data[tid % 4],
        G_k.z.data[tid % 4],
        S_k.x.data[tid % 4],
        S_k.y.data[tid % 4],
        S_k.z.data[tid % 4],
        G_k.x.data[tid % 4],
        G_k.y.data[tid % 4],
        G_k.z.data[tid % 4]
    );

    // load result
    fq_gpu::load(G_k.x.data[tid % 4], result[blockIdx.x].x.data[tid % 4]);
    fq_gpu::load(G_k.y.data[tid % 4], result[blockIdx.x].y.data[tid % 4]);
    fq_gpu::load(G_k.z.data[tid % 4], result[blockIdx.x].z.data[tid % 4]);
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

/**
 * Test kernel
 */
__global__ void test_kernel(g1::element *points, g1::element *final_result) {
    auto grp = fixnum::layout();
    int subgroup = grp.meta_group_rank();
    int subgroup_size = grp.meta_group_size();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // g1::add(
    //     points[1].x.data[tid], 
    //     points[1].y.data[tid], 
    //     points[1].z.data[tid], 
    //     points[0].x.data[tid], 
    //     points[0].y.data[tid], 
    //     points[0].z.data[tid], 
    //     final_result[0].x.data[tid], 
    //     final_result[0].y.data[tid], 
    //     final_result[0].z.data[tid]
    // );

    // Initialize G and S with 0
    // fq_gpu::load(0x0, final_result[0].x.data[tid % 4]);
    // fq_gpu::load(0x0, final_result[0].y.data[tid % 4]);
    // fq_gpu::load(0x0, final_result[0].z.data[tid % 4]);
    

    // for (int i = 0; i < 80; i++) {
    //     g1::add(
    //         points[0].x.data[tid], 
    //         points[0].y.data[tid], 
    //         points[0].z.data[tid], 
    //         final_result[0].x.data[tid], 
    //         final_result[0].y.data[tid], 
    //         final_result[0].z.data[tid], 
    //         final_result[0].x.data[tid], 
    //         final_result[0].y.data[tid], 
    //         final_result[0].z.data[tid]
    //     );

    //     if (fq_gpu::is_zero(final_result[0].x.data[tid % 4]) 
    //                         && fq_gpu::is_zero(final_result[0].y.data[tid % 4]) 
    //                         && fq_gpu::is_zero(final_result[0].z.data[tid % 4])) {
    //         g1::doubling(
    //             points[0].x.data[tid], 
    //             points[0].y.data[tid], 
    //             points[0].z.data[tid], 
    //             final_result[0].x.data[tid], 
    //             final_result[0].y.data[tid], 
    //             final_result[0].z.data[tid]
    //         );
    //     }
    // }
}

__global__ void simple_jacobian_addition(g1::element *points, g1::element *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid < LIMBS) {
        g1::add(
            points[0].x.data[tid % 4], 
            points[0].y.data[tid % 4], 
            points[0].z.data[tid % 4], 
            points[1].x.data[tid % 4], 
            points[1].y.data[tid % 4], 
            points[1].z.data[tid % 4], 
            result[0].x.data[tid % 4], 
            result[0].y.data[tid % 4], 
            result[0].z.data[tid % 4]
        );

        __syncthreads();

        g1::add(
            result[0].x.data[tid % 4], 
            result[0].y.data[tid % 4], 
            result[0].z.data[tid % 4], 
            result[0].x.data[tid % 4], 
            result[0].y.data[tid % 4], 
            result[0].z.data[tid % 4], 
            result[1].x.data[tid % 4], 
            result[1].y.data[tid % 4], 
            result[1].z.data[tid % 4]
        );

        if (fq_gpu::is_zero(result[1].x.data[tid % 4]) 
                            && fq_gpu::is_zero(result[1].y.data[tid % 4]) 
                            && fq_gpu::is_zero(result[1].z.data[tid % 4])) {
            g1::doubling(
                result[0].x.data[tid], 
                result[0].y.data[tid], 
                result[0].z.data[tid], 
                result[1].x.data[tid], 
                result[1].y.data[tid], 
                result[1].z.data[tid]
            );
        }
}

// __global__ void simple_projective_addition(g1::element *points, g1::element *result) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     // if (tid < LIMBS) {
//         g1::add_projective(
//             points[0].x.data[tid % 4], 
//             points[0].y.data[tid % 4], 
//             points[0].z.data[tid % 4], 
//             points[1].x.data[tid % 4], 
//             points[1].y.data[tid % 4], 
//             points[1].z.data[tid % 4], 
//             result[0].x.data[tid % 4], 
//             result[0].y.data[tid % 4], 
//             result[0].z.data[tid % 4]
//         );

//         __syncthreads();

//         g1::add_projective(
//             result[0].x.data[tid % 4], 
//             result[0].y.data[tid % 4], 
//             result[0].z.data[tid % 4], 
//             result[0].x.data[tid % 4], 
//             result[0].y.data[tid % 4], 
//             result[0].z.data[tid % 4], 
//             result[1].x.data[tid % 4], 
//             result[1].y.data[tid % 4], 
//             result[1].z.data[tid % 4]
//         );
// }

// Double and add implementation using bit-decomposition with time complexity: O(k)
__global__ void test_double_add(g1::element *point, fr_gpu *scalar, g1::element *result) {
    // Holders for points and scalars
    g1::element R;
    g1::element Q;
    
    // Inner and outer accumulators
    g1::element inner_accumulator;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < LIMBS) {
        fq_gpu::load(0, R.x.data[tid]); 
        fq_gpu::load(0, R.y.data[tid]); 
        fq_gpu::load(0, R.z.data[tid]); 

        fq_gpu::load(0, inner_accumulator.x.data[tid]); 
        fq_gpu::load(0, inner_accumulator.y.data[tid]); 
        fq_gpu::load(0, inner_accumulator.z.data[tid]); 

        fq_gpu::load(point[0].x.data[tid], Q.x.data[tid]);
        fq_gpu::load(point[0].y.data[tid], Q.y.data[tid]);
        fq_gpu::load(point[0].z.data[tid], Q.z.data[tid]);

        fq_gpu::to_monty(point[0].x.data[tid], Q.x.data[tid]);
        fq_gpu::to_monty(point[0].y.data[tid], Q.y.data[tid]);
        fq_gpu::to_monty(point[0].z.data[tid], Q.z.data[tid]);

        for (int j = 0; j < LIMBS; j++) {
            for (int i = 63; i >= 0; i--) {
                // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
                // and extracting the i-th bit of scalar in limb.
                if (((scalar[0].data[j] >> i) & 1) ? 1 : 0)
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
            // Inner accumulator 
            g1::add(
                R.x.data[tid], R.y.data[tid], R.z.data[tid], 
                inner_accumulator.x.data[tid], inner_accumulator.y.data[tid], inner_accumulator.z.data[tid], 
                inner_accumulator.x.data[tid], inner_accumulator.y.data[tid], inner_accumulator.z.data[tid]
            );
        }

        fq_gpu::load(inner_accumulator.x.data[tid], result[0].x.data[tid]);
        fq_gpu::load(inner_accumulator.y.data[tid], result[0].y.data[tid]);
        fq_gpu::load(inner_accumulator.z.data[tid], result[0].z.data[tid]);

        fq_gpu::from_monty(result[0].x.data[tid], result[0].x.data[tid]);
        fq_gpu::from_monty(result[0].y.data[tid], result[0].y.data[tid]);
        fq_gpu::from_monty(result[0].z.data[tid], result[0].z.data[tid]);
    }
}

}