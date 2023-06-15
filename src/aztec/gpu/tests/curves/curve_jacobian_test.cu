#include "group.cu"

using namespace std;
using namespace std::chrono;
using namespace gpu_barretenberg;

static constexpr size_t LIMBS_NUM = 4;
static constexpr size_t BLOCKS = 1;
static constexpr size_t THREADS = 1;

/* -------------------------- Mixed Addition Test ---------------------------------------------- */

__global__ void initialize_mixed_add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z) {
    fq_gpu a_x{ 0x92716caa6cac6d26, 0x1e6e234136736544, 0x1bb04588cde00af0, 0x9a2ac922d97e6f5 };
    fq_gpu a_y{ 0x9e693aeb52d79d2d, 0xf0c1895a61e5e975, 0x18cd7f5310ced70f, 0xac67920a22939ad };
    fq_gpu a_z{ 0xfef593c9ce1df132, 0xe0486f801303c27d, 0x9bbd01ab881dc08e, 0x2a589badf38ec0f9 };
    fq_gpu b_x{ 0xa1ec5d1398660db8, 0x6be3e1f6fd5d8ab1, 0x69173397dd272e11, 0x12575bbfe1198886 };
    fq_gpu b_y{ 0xcfbfd4441138823e, 0xb5f817e28a1ef904, 0xefb7c5629dcc1c42, 0x1a9ed3d6f846230e };
    fq_gpu exp_x{ 0x2a9d0201fccca20, 0x36f969b294f31776, 0xee5534422a6f646, 0x911dbc6b02310b6 };
    fq_gpu exp_y{ 0x14c30aaeb4f135ef, 0x9c27c128ea2017a1, 0xf9b7d80c8315eabf, 0x35e628df8add760 };
    fq_gpu exp_z{ 0xa43fe96673d10eb3, 0x88fbe6351753d410, 0x45c21cc9d99cb7d, 0x3018020aa6e9ede5 };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        x[i] = b_x.data[i];
        y[i] = b_y.data[i];
        expected_x[i] = exp_x.data[i];
        expected_y[i] = exp_y.data[i];
        expected_z[i] = exp_z.data[i];
    }
}

__global__ void mixed_add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *res_x, var *res_y, var *res_z) {
    g1_gpu::element lhs;
    g1_gpu::affine_element rhs;
    g1_gpu::element result;
    g1_gpu::element expected;
    
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        lhs.x.data[tid] = fq_gpu::to_monty(a[tid], res_x[tid]);
        lhs.y.data[tid] = fq_gpu::to_monty(b[tid], res_x[tid]);
        lhs.z.data[tid] = fq_gpu::to_monty(c[tid], res_x[tid]);
        rhs.x.data[tid] = fq_gpu::to_monty(x[tid], res_x[tid]);
        rhs.y.data[tid] = fq_gpu::to_monty(y[tid], res_x[tid]);

        // lhs + rhs (affine element + jacobian element)
        g1_gpu::mixed_add(
            lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
            rhs.x.data[tid], rhs.y.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // Return results from montgomery form 
        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);
    }
}

/* -------------------------- Doubling Test ---------------------------------------------- */

__global__ void initialize_dbl_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z) {
    fq_gpu a_x{ 0x8d1703aa518d827f, 0xd19cc40779f54f63, 0xabc11ce30d02728c, 0x10938940de3cbeec };
    fq_gpu a_y{ 0xcf1798994f1258b4, 0x36307a354ad90a25, 0xcd84adb348c63007, 0x6266b85241aff3f };
    fq_gpu a_z{ 0xe213e18fd2df7044, 0xb2f42355982c5bc8, 0xf65cf5150a3a9da1, 0xc43bde08b03aca2 };
    fq_gpu exp_x{ 0xd5c6473044b2e67c, 0x89b185ea20951f3a, 0x4ac597219cf47467, 0x2d00482f63b12c86 };
    fq_gpu exp_y{ 0x4e7e6c06a87e4314, 0x906a877a71735161, 0xaa7b9893cc370d39, 0x62f206bef795a05 };
    fq_gpu exp_z{ 0x8813bdca7b0b115a, 0x929104dffdfabd22, 0x3fff575136879112, 0x18a299c1f683bdca };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        expected_x[i] = exp_x.data[i];
        expected_y[i] = exp_y.data[i];
        expected_z[i] = exp_z.data[i];
    }
}

__global__ void dbl_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *res_x, var *res_y, var *res_z) {
    g1_gpu::element lhs;
    g1_gpu::element result;
    g1_gpu::element expected;
    
    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        lhs.x.data[tid] = fq_gpu::to_monty(a[tid], res_x[tid]);
        lhs.y.data[tid] = fq_gpu::to_monty(b[tid], res_y[tid]);
        lhs.z.data[tid] = fq_gpu::to_monty(c[tid], res_z[tid]);

        // lhs.doubling
        g1_gpu::doubling(
            lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );
        //  (lhs.doubling).doubling
        g1_gpu::doubling(
            res_x[tid], res_y[tid], res_z[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );
        //  ((lhs.doubling).doubling).doubling
        g1_gpu::doubling(
            res_x[tid], res_y[tid], res_z[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // Return results from montgomery form 
        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);
    }
}

/* -------------------------- Addition Test ---------------------------------------------- */

__global__ void initialize_add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z) {
    fq_gpu a_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu a_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu a_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu b_x{ 0xafdb8a15c98bf74c, 0xac54df622a8d991a, 0xc6e5ae1f3dad4ec8, 0x1bd3fb4a59e19b52 };
    fq_gpu b_y{ 0x21b3bb529bec20c0, 0xaabd496406ffb8c1, 0xcd3526c26ac5bdcb, 0x187ada6b8693c184 };
    fq_gpu b_z{ 0xffcd440a228ed652, 0x8a795c8f234145f1, 0xd5279cdbabb05b95, 0xbdf19ba16fc607a };
    fq_gpu exp_x{ 0x18764da36aa4cd81, 0xd15388d1fea9f3d3, 0xeb7c437de4bbd748, 0x2f09b712adf6f18f };
    fq_gpu exp_y{ 0x50c5f3cab191498c, 0xe50aa3ce802ea3b5, 0xd9d6125b82ebeff8, 0x27e91ba0686e54fe };
    fq_gpu exp_z{ 0xe4b81ef75fedf95, 0xf608edef14913c75, 0xfd9e178143224c96, 0xa8ae44990c8accd };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        x[i] = b_x.data[i];
        y[i] = b_y.data[i];
        z[i] = b_z.data[i];
        expected_x[i] = exp_x.data[i];
        expected_y[i] = exp_y.data[i];
        expected_z[i] = exp_z.data[i];
    }
}

__global__ void add_check_against_constants
(var *a, var *b, var *c, var *x, var *y, var *z, var *res_x, var *res_y, var *res_z) {
    g1_gpu::element lhs;
    g1_gpu::element rhs;
    g1_gpu::element result;
    g1_gpu::element expected;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        lhs.x.data[tid] = fq_gpu::to_monty(a[tid], res_x[tid]);
        lhs.y.data[tid] = fq_gpu::to_monty(b[tid], res_x[tid]);
        lhs.z.data[tid] = fq_gpu::to_monty(c[tid], res_x[tid]);
        rhs.x.data[tid] = fq_gpu::to_monty(x[tid], res_x[tid]);
        rhs.y.data[tid] = fq_gpu::to_monty(y[tid], res_x[tid]);
        rhs.z.data[tid] = fq_gpu::to_monty(z[tid], res_x[tid]);

        // lhs + rhs (affine element + affine element)
        g1_gpu::add(
            lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
            rhs.x.data[tid], rhs.y.data[tid], rhs.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );
        
        // Transform results from montgomery form 
        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);
    }
}

/* -------------------------- Add Exception Test ---------------------------------------------- */

__global__ void initialize_add_exception_test_dbl
(var *a, var *b, var *c, var *x, var *y, var *z) {
    fq_gpu a_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu a_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu a_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu b_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu b_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu b_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        x[i] = b_x.data[i];
        y[i] = b_y.data[i];
        z[i] = b_z.data[i];
    }
}

__global__ void add_exception_test_dbl
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res_x, var *res_y, var *res_z) {
    g1_gpu::element lhs;
    g1_gpu::element rhs;
    g1_gpu::element result;
    g1_gpu::element expected;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        lhs.x.data[tid] = fq_gpu::load(a[tid], expected_x[tid]);
        lhs.y.data[tid] = fq_gpu::load(b[tid], expected_x[tid]);
        lhs.z.data[tid] = fq_gpu::load(c[tid], expected_x[tid]);
        rhs.x.data[tid] = fq_gpu::load(x[tid], expected_x[tid]);
        rhs.y.data[tid] = fq_gpu::load(y[tid], expected_x[tid]);
        rhs.z.data[tid] = fq_gpu::load(z[tid], expected_x[tid]);

        // lhs + rhs
        g1_gpu::add(
            lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
            rhs.x.data[tid], rhs.y.data[tid], rhs.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // Temporarily handle case where P = Q -- NEED TO MOVE TO 'group.cu' file
        if (fq_gpu::is_zero(res_x[tid]) && fq_gpu::is_zero(res_y[tid]) && fq_gpu::is_zero(res_z[tid])) {
            g1_gpu::doubling(
                lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
                res_x[tid], res_y[tid], res_z[tid]
            );
        }

        // lhs.doubling
        g1_gpu::doubling(
            lhs.x.data[tid], lhs.y.data[tid], lhs.z.data[tid], 
            expected_x[tid], expected_y[tid], expected_z[tid]
        );

        // Transform results from montgomery form 
        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);

        // Transform results from montgomery form 
        fq_gpu::from_monty(expected_x[tid], expected_x[tid]);
        fq_gpu::from_monty(expected_y[tid], expected_y[tid]);
        fq_gpu::from_monty(expected_z[tid], expected_z[tid]);

        // EXPECT(lsh + rhs == lhs.doubling);
    }
}

/* -------------------------- Add Double Consistency Test ---------------------------------------------- */

__global__ void initialize_add_dbl_consistency
(var *a, var *b, var *c, var *x, var *y, var *z) {
    fq_gpu a_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu a_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu a_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu b_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu b_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu b_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        x[i] = b_x.data[i];
        y[i] = b_y.data[i];
        z[i] = b_z.data[i];
    }
}

__global__ void add_dbl_consistency
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res_x, var *res_y, var *res_z) {
    g1_gpu::element a_element;
    g1_gpu::element b_element;
    g1_gpu::element c_element;
    g1_gpu::element d_element;
    g1_gpu::element add_result;
    g1_gpu::element dbl_result;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        a_element.x.data[tid] = fq_gpu::load(a[tid], res_x[tid]);
        a_element.y.data[tid] = fq_gpu::load(b[tid], res_x[tid]);
        a_element.z.data[tid] = fq_gpu::load(c[tid], res_x[tid]);
        b_element.x.data[tid] = fq_gpu::load(x[tid], res_x[tid]);
        b_element.y.data[tid] = fq_gpu::load(y[tid], res_x[tid]);
        b_element.z.data[tid] = fq_gpu::load(z[tid], res_x[tid]);

        // c = a + b
        g1_gpu::add(
            a_element.x.data[tid], a_element.y.data[tid], a_element.z.data[tid], 
            b_element.x.data[tid], b_element.y.data[tid], b_element.z.data[tid], 
            c_element.x.data[tid], c_element.y.data[tid], c_element.z.data[tid]
        ); 
        
        // b = -b
        fq_gpu::neg(b_element.y.data[tid], b_element.y.data[tid]);                                                                                                                                                      
        
        // d = a + b
        g1_gpu::add(
            a_element.x.data[tid], a_element.y.data[tid], a_element.z.data[tid], 
            b_element.x.data[tid], b_element.y.data[tid], b_element.z.data[tid], 
            d_element.x.data[tid], d_element.y.data[tid], d_element.z.data[tid]
        );
       
        // result = c + d
        g1_gpu::add(
            c_element.x.data[tid], c_element.y.data[tid], c_element.z.data[tid], 
            d_element.x.data[tid], d_element.y.data[tid], d_element.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // Temporarily handle case where P = Q -- NEED TO MOVE TO 'group.cu' file
        if (fq_gpu::is_zero(res_x[tid]) && fq_gpu::is_zero(res_y[tid]) && fq_gpu::is_zero(res_z[tid])) {
            g1_gpu::doubling(
                a_element.x.data[tid], a_element.y.data[tid], a_element.z.data[tid], 
                res_x[tid], res_y[tid], res_z[tid]
            );
        }

        // a.doubling
        g1_gpu::doubling(
            a_element.x.data[tid], a_element.y.data[tid], a_element.z.data[tid], 
            expected_x[tid], expected_y[tid], expected_z[tid]
        );
         
        // Transform results from montgomery form 
        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);
        
        // Transform results from montgomery form 
        fq_gpu::from_monty(expected_x[tid], expected_x[tid]);
        fq_gpu::from_monty(expected_y[tid], expected_y[tid]);
        fq_gpu::from_monty(expected_z[tid], expected_z[tid]);

        // EXPECT (c + d == a.doubling);
    }
}

/* -------------------------- Add Double Consistency Repeated Test ---------------------------------------------- */

__global__ void initialize_add_dbl_consistency_repeated
(var *a, var *b, var *c) {
    fq_gpu a_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu a_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu a_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
    }
}

__global__ void add_dbl_consistency_repeated
(var *a, var *b, var *c, var *expected_x, var *expected_y, var *expected_z, var *res_x, var *res_y, var *res_z) {
    g1_gpu::element a_element;
    g1_gpu::element b_element;
    g1_gpu::element c_element;
    g1_gpu::element d_element;
    g1_gpu::element e_element;
    g1_gpu::element result;
    g1_gpu::element expected;

    // Calculate global thread ID, and boundry check
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < LIMBS) {
        a_element.x.data[tid] = fq_gpu::load(a[tid], res_x[tid]);
        a_element.y.data[tid] = fq_gpu::load(b[tid], res_x[tid]);
        a_element.z.data[tid] = fq_gpu::load(c[tid], res_x[tid]);

        // b = 2a
        g1_gpu::doubling(
            a_element.x.data[tid], a_element.y.data[tid], a_element.z.data[tid], 
            b_element.x.data[tid], b_element.y.data[tid], b_element.z.data[tid]
        );

        // c = 4a
        g1_gpu::doubling(
            b_element.x.data[tid], b_element.y.data[tid], b_element.z.data[tid], 
            c_element.x.data[tid], c_element.y.data[tid], c_element.z.data[tid]
        );
         
        // d = 3a
        g1_gpu::add(
            a_element.x.data[tid], a_element.y.data[tid], a_element.z.data[tid], 
            b_element.x.data[tid], b_element.y.data[tid], b_element.z.data[tid], 
            d_element.x.data[tid], d_element.y.data[tid], d_element.z.data[tid]
        ); 

        // e = 5a
        g1_gpu::add(
            a_element.x.data[tid], a_element.y.data[tid], a_element.z.data[tid], 
            c_element.x.data[tid], c_element.y.data[tid], c_element.z.data[tid], 
            e_element.x.data[tid], e_element.y.data[tid], e_element.z.data[tid]
        ); 
  
        // result = 8a
        g1_gpu::add(
            d_element.x.data[tid], d_element.y.data[tid], d_element.z.data[tid], 
            e_element.x.data[tid], e_element.y.data[tid], e_element.z.data[tid], 
            res_x[tid], res_y[tid], res_z[tid]
        );

        // c.doubling
        g1_gpu::doubling(
            c_element.x.data[tid], c_element.y.data[tid], c_element.z.data[tid], 
            expected_x[tid], expected_y[tid], expected_z[tid]
        );

        // Transform results from montgomery form 
        fq_gpu::from_monty(res_x[tid], res_x[tid]);
        fq_gpu::from_monty(res_y[tid], res_y[tid]);
        fq_gpu::from_monty(res_z[tid], res_z[tid]);

        // Transform results from montgomery form 
        fq_gpu::from_monty(expected_x[tid], expected_x[tid]);
        fq_gpu::from_monty(expected_y[tid], expected_y[tid]);
        fq_gpu::from_monty(expected_z[tid], expected_z[tid]);

        // EXPECT (d + e == c.doubling)
    }
}

/* -------------------------- Group Exponentiation Check Against Constants Test ---------------------------------------------- */

__global__ void initialize_group_exponentiation
(var *a, var *b, var *c, var *expected_x, var *expected_y, var *expected_z) {
    fq_gpu a_x{ 0x184b38afc6e2e09a, 0x4965cd1c3687f635, 0x334da8e7539e71c4, 0xf708d16cfe6e14 };
    fq_gpu a_y{ 0x2a6ff6ffc739b3b6, 0x70761d618b513b9, 0xbf1645401de26ba1, 0x114a1616c164b980 };
    fq_gpu a_z{ 0x10143ade26bbd57a, 0x98cf4e1f6c214053, 0x6bfdc534f6b00006, 0x1875e5068ababf2c };
    fq_gpu exp_x{ 0xC22BA855EE138794, 0xA61591A7E7FD82BF, 0xE156E7E491B4B7E2, 0x2F4620C8373C106A };
    fq_gpu exp_y{ 0xFAFBA721679C418, 0xE5491810D637BB55, 0x64B6FAD0A59D97B2, 0x111DA26AEEE41706 };
    fq_gpu exp_z{ 0x59F11DAE3A07BF31, 0xDB2756DB66333FB, 0x34F2D97DAD44161, 0xD1A485A89C277DA };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = a_x.data[i];
        b[i] = a_y.data[i];
        c[i] = a_z.data[i];
        expected_x[i] = exp_x.data[i];
        expected_y[i] = exp_y.data[i];
        expected_z[i] = exp_z.data[i];
    }
}

__global__ void group_exponentiation(uint64_t *a, uint64_t *b, uint64_t *c, var *res_x, var *res_y, var *res_z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    g1_gpu::element one; 
    g1_gpu::element R;
    g1_gpu::element Q;

    fr_gpu exponent{ 0xb67299b792199cf0, 0xc1da7df1e7e12768, 0x692e427911532edf, 0x13dd85e87dc89978 };

    fq_gpu::load(gpu_barretenberg::one_x_bn_254[tid], one.x.data[tid]);
    fq_gpu::load(gpu_barretenberg::one_y_bn_254[tid], one.y.data[tid]);
    fq_gpu::load(fq_gpu::one().data[tid], one.z.data[tid]);

    if (tid < LIMBS) {
        // Initialize 'R' to the identity element, Q to the curve point
        fq_gpu::load(0, R.x.data[tid]); 
        fq_gpu::load(0, R.y.data[tid]); 
        fq_gpu::load(0, R.z.data[tid]); 

        fq_gpu::load(one.x.data[tid], Q.x.data[tid]);
        fq_gpu::load(one.y.data[tid], Q.y.data[tid]);
        fq_gpu::load(one.z.data[tid], Q.z.data[tid]);

        // Loop for each limb starting with the last limb
        for (int j = 3; j >= 0; j--) {
            // Loop for each bit of scalar
            for (int i = 64; i >= 0; i--) {
                // Performs bit-decompositon by traversing the bits of the scalar from MSB to LSB
                // and extracting the i-th bit of scalar in limb.
                if (((exponent.data[j] >> i) & 1) ? 1 : 0)
                    g1_gpu::add(
                        R.x.data[tid], R.y.data[tid], R.z.data[tid], 
                        Q.x.data[tid], Q.y.data[tid], Q.z.data[tid], 
                        R.x.data[tid], R.y.data[tid], R.z.data[tid]
                    );
                if (i != 0) 
                    g1_gpu::doubling(
                        R.x.data[tid], R.y.data[tid], R.z.data[tid], 
                        R.x.data[tid], R.y.data[tid], R.z.data[tid]
                    );
            }
        }
    }

    // Store the final value of R into the result array for this limb
    fq_gpu::load(R.x.data[tid], res_x[tid]);
    fq_gpu::load(R.y.data[tid], res_y[tid]);
    fq_gpu::load(R.z.data[tid], res_z[tid]);

    // Convert back from montgomery form
    fq_gpu::from_monty(res_x[tid], res_x[tid]);
    fq_gpu::from_monty(res_y[tid], res_y[tid]);
    fq_gpu::from_monty(res_z[tid], res_z[tid]);
}

/* -------------------------- Operator Ordering Test ---------------------------------------------- */

__global__ void initialize_operator_ordering(var *a, var *b, var *c, var *d) {
    fr_gpu scalar{ 0xcfbfd4441138823e, 0xb5f817e28a1ef904, 0xefb7c5629dcc1c42, 0x1a9ed3d6f846230e };

    for (int i = 0; i < LIMBS_NUM; i++) {
        a[i] = gpu_barretenberg::one_x_bn_254[i];
        b[i] = gpu_barretenberg::one_y_bn_254[i];
        c[i] = fq_gpu::one().data[i];
        d[i] = scalar.data[i];
    }
}

__global__ void operator_ordering(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, var *res_x, var *res_y, var *res_z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    g1_gpu::element b_new;
    g1_gpu::element c_new;
    g1_gpu::element d_new;

    // Copy a into b
    fq_gpu::load(a[tid], b_new.x.data[tid]);
    fq_gpu::load(b[tid], b_new.y.data[tid]);
    fq_gpu::load(c[tid], b_new.z.data[tid]);

    // c = a + b
    g1_gpu::add(
        a[tid], b[tid], c[tid], 
        b_new.x.data[tid], b_new.y.data[tid], b_new.z.data[tid], 
        c_new.x.data[tid], c_new.y.data[tid], c_new.z.data[tid]
    );

    // Double is c == 0
    if (fq_gpu::is_zero(c_new.x.data[tid]) && fq_gpu::is_zero(c_new.y.data[tid]) && fq_gpu::is_zero(c_new.z.data[tid])) {
        g1_gpu::doubling(
            a[tid], b[tid], c[tid], 
            c_new.x.data[tid], c_new.y.data[tid], c_new.z.data[tid]
        );
    }

    // d = b + a
    g1_gpu::add(
        b_new.x.data[tid], b_new.y.data[tid], b_new.z.data[tid], 
        a[tid], b[tid], c[tid], 
        d_new.x.data[tid], d_new.y.data[tid], d_new.z.data[tid]
    );

    // Double is d == 0
    if (fq_gpu::is_zero(d_new.x.data[tid]) && fq_gpu::is_zero(d_new.y.data[tid]) && fq_gpu::is_zero(d_new.z.data[tid])) {
        g1_gpu::doubling(
            b_new.x.data[tid], b_new.y.data[tid], b_new.z.data[tid], 
            d_new.x.data[tid], d_new.y.data[tid], d_new.z.data[tid]
        );
    }

    // Return final result. Expect c == d
    fq_gpu::load(d_new.x.data[tid], res_x[tid]);
    fq_gpu::load(d_new.y.data[tid], res_y[tid]);
    fq_gpu::load(d_new.z.data[tid], res_z[tid]);

    fq_gpu::from_monty(res_x[tid], res_x[tid]);
    fq_gpu::from_monty(res_y[tid], res_y[tid]);
    fq_gpu::from_monty(res_z[tid], res_z[tid]);
}

/* -------------------------- Executing Initialization and Workload Kernels ---------------------------------------------- */

void assert_checks(var *expected, var *result) {
    // Explicit synchronization barrier
    cudaDeviceSynchronize();
    
    // Assert clause
    assert(expected[0] == result[0]);
    assert(expected[1] == result[1]);
    assert(expected[2] == result[2]);
    assert(expected[3] == result[3]);

    // Print statements
    printf("expected[0] is: %zu\n", expected[0]);
    printf("expected[1] is: %zu\n", expected[1]);
    printf("expected[2] is: %zu\n", expected[2]);
    printf("expected[3] is: %zu\n", expected[3]);
    printf("result[0] is: %zu\n", result[0]);
    printf("result[1] is: %zu\n", result[1]);
    printf("result[2] is: %zu\n", result[2]);
    printf("result[3] is: %zu\n", result[3]);
}

void execute_kernels
(var *a, var *b, var *c, var *x, var *y, var *z, var *expected_x, var *expected_y, var *expected_z, var *res_x, var *res_y, var *res_z) {
    // Mixed Addition Test
    initialize_mixed_add_check_against_constants<<<BLOCKS, THREADS>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z);
    mixed_add_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Doubling Test
    initialize_dbl_check_against_constants<<<BLOCKS, THREADS>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z);
    dbl_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Addition Test
    initialize_add_check_against_constants<<<BLOCKS, THREADS>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z);
    add_check_against_constants<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Add Exception Test
    initialize_add_exception_test_dbl<<<BLOCKS, THREADS>>>(a, b, c, x, y, z);
    add_exception_test_dbl<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Add Double Consistency Test
    initialize_add_dbl_consistency<<<BLOCKS, THREADS>>>(a, b, c, x, y, z);
    add_dbl_consistency<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, y, z, expected_x, expected_y, expected_z, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Add Double Consistency Repeated Test
    initialize_add_dbl_consistency_repeated<<<BLOCKS, THREADS>>>(a, b, c);
    add_dbl_consistency_repeated<<<BLOCKS, LIMBS_NUM>>>(a, b, c, expected_x, expected_y, expected_z, res_x, res_y, res_z);

    // Group Exponentiation Consistency Test
    initialize_group_exponentiation<<<BLOCKS, THREADS>>>(a, b, c, expected_x, expected_y, expected_z);
    group_exponentiation<<<BLOCKS, LIMBS_NUM>>>(a, b, c, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);

    // Operator Ordering Test
    initialize_operator_ordering<<<BLOCKS, THREADS>>>(a, b, c, x);
    operator_ordering<<<BLOCKS, LIMBS_NUM>>>(a, b, c, x, res_x, res_y, res_z);
    assert_checks(expected_x, res_x);
    assert_checks(expected_y, res_y);
    assert_checks(expected_z, res_z);
}

/* -------------------------- Main Entry Function ---------------------------------------------- */

int main(int, char**) {
    // Start timer
    auto start = high_resolution_clock::now();

    // Define pointers to 'uint64_t' type
    var *a, *b, *c, *x, *y, *z, *expected_x, *expected_y, *expected_z, *res_x, *res_y, *res_z;    

    // Allocate unified memory accessible by host and device
    cudaMallocManaged(&a, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&b, LIMBS_NUM * sizeof(uint64_t));
    cudaMallocManaged(&c, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&expected_z, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_x, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_y, LIMBS * sizeof(uint64_t));
    cudaMallocManaged(&res_z, LIMBS * sizeof(uint64_t));

    // Execute kernel functions
    execute_kernels(a, b, c, x, y, z, expected_x, expected_y, expected_z, res_x, res_y, res_z);

    // Successfull execution of unit tests
    cout << "******* All 'g1_gpu BN-254 Curve' unit tests passed! **********" << endl;

    // End timer
    auto stop = high_resolution_clock::now();

    // Calculate duraion of execution time 
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds\n" << endl; 

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(expected_x);
    cudaFree(expected_y);
    cudaFree(expected_z);
    cudaFree(res_x);
    cudaFree(res_y);
    cudaFree(res_z);

    cout << "Completed sucessfully!" << endl;

    return 0;
}
