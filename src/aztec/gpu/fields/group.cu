#include "group.cuh"

using namespace std;
using namespace gpu_barretenberg;

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::load_affine(affine_element & X, const var * y) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::load_jacobian(element & X, const var * y) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::is_affine(const affine_element &X) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::is_affine_equal(const affine_element & X) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::is_jacobian_equal(const affine_element & X) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::store_affine(const affine_element &X, const var *y) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::store_jacobian(const element & X, const var * y) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::set_zero(const element &X) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::is_zero(const element &X) {

}

template <class fq_gpu>
__device__ void group_gpu<fq_gpu>::mixed_add(fq_gpu &X, fq_gpu &Y) noexcept {
    // fq_gpu z1z1, u2, s2, h, hh, i, j, r, v;
    // fq_gpu t0, t1;

    // fq_gpu::square(X.data[0], Y.data[0]);

    // // T0 = z1.z1
    // Fq T0 = z.sqr();

    // // T1 = x2.t0 - x1 = x2.z1.z1 - x1
    // Fq T1 = other.x * T0;
    // T1 -= x;

    // // T2 = T0.z1 = z1.z1.z1
    // // T2 = T2.y2 - y1 = y2.z1.z1.z1 - y1
    // Fq T2 = z * T0;
    // T2 *= other.y;
    // T2 -= y;

    // if (__builtin_expect(T1.is_zero(), 0)) {
    //     if (T2.is_zero()) {
    //         self_dbl();
    //         return *this;
    //     } else {
    //         self_set_infinity();
    //         return *this;
    //     }
    // }

    // // T2 = 2T2 = 2(y2.z1.z1.z1 - y1) = R
    // // z3 = z1 + H
    // T2 += T2;
    // z += T1;

    // // T3 = T1*T1 = HH
    // Fq T3 = T1.sqr();

    // // z3 = z3 - z1z1 - HH
    // T0 += T3;

    // // z3 = (z1 + H)*(z1 + H)
    // z.self_sqr();
    // z -= T0;

    // // T3 = 4HH
    // T3 += T3;
    // T3 += T3;

    // // T1 = T1*T3 = 4HHH
    // T1 *= T3;

    // // T3 = T3 * x1 = 4HH*x1
    // T3 *= x;

    // // T0 = 2T3
    // T0 = T3 + T3;

    // // T0 = T0 + T1 = 2(4HH*x1) + 4HHH
    // T0 += T1;
    // x = T2.sqr();

    // // x3 = x3 - T0 = R*R - 8HH*x1 -4HHH
    // x -= T0;

    // // T3 = T3 - x3 = 4HH*x1 - x3
    // T3 -= x;

    // T1 *= y;
    // T1 += T1;

    // // T3 = T2 * T3 = R*(4HH*x1 - x3)
    // T3 *= T2;

    // // y3 = T3 - T1
    // y = T3 - T1;
    // return *this;
}
