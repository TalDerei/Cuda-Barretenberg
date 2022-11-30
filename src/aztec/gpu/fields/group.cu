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
__device__ void group_gpu<fq_gpu>::store_jacobian(const element &X, const var *y) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::set_zero(const element &X) {

}

template <class fq_gpu> 
__device__ void group_gpu<fq_gpu>::is_zero(const element &X) {

}

// Elliptic curve algorithms: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-zadd-2007-m
template <class fq_gpu>
__device__ void group_gpu<fq_gpu>::mixed_add(var X, var Y, var Z, var A, var B, var &res_x, var &res_y, var &res_z) noexcept {
    var z1z1, u2, s2, h, hh, i, j, r, v, t0, t1;

    // X Element
    fq_gpu::square(Z, z1z1);   
    fq_gpu::mul(A, z1z1, u2); 
    fq_gpu::mul(B, Z, s2);
    fq_gpu::mul(s2, z1z1, s2); 
    // if (fq_gpu::equal(u2, X) && fq_gpu::equal(s2, Y)) {
    //     // double
    // }
    fq_gpu::sub(u2, X, h);   
    fq_gpu::square(h, hh);    
    fq_gpu::add(hh, hh, i);     
    fq_gpu::add(i, i, i);      
    fq_gpu::mul(i, h, j);      
    fq_gpu::sub(s2, Y, r);      
    fq_gpu::add(r, r, r);      
    fq_gpu::mul(X, i, v);      
    fq_gpu::square(r, t0);     
    fq_gpu::add(v, v, t1);
    fq_gpu::sub(t0, j, t0);    
    fq_gpu::sub(t0, t1, res_x); 

    // Y Element
    fq_gpu::sub(v, res_x, t0);  
    fq_gpu::mul(Y, j, t1);     
    fq_gpu::add(t1, t1, t1);
    fq_gpu::mul(t0, r, t0);     
    fq_gpu::sub(t0, t1, res_y);

    // Z Element
    fq_gpu::add(Z, h, t0);      
    fq_gpu::square(t0, t0);     
    fq_gpu::sub(t0, z1z1, t0);  
    fq_gpu::sub(t0, hh, res_z); 
}

template <class fq_gpu>
__device__ void group_gpu<fq_gpu>::doubling(var X, var Y, var Z, var &res_x, var &res_y, var &res_z) noexcept {
    var T0, T1, T2, T3;

    // X Element
    fq_gpu::square(X, T0);              // T0 = x*x
    fq_gpu::square(Y, T1);              // T1 = y*y
    fq_gpu::square(T1, T2);             // T2 = T1*T1 = y*y*y*y
    fq_gpu::add(T1, X, T1);             // T1 = T1+x = x + y*y
    fq_gpu::square(T1, T1);             // T1 = T1*T1
    fq_gpu::add(T0, T2, T3);            // T3 = T0 +T2 = x*x + y*y*y*y
    fq_gpu::sub(T1, T3, T1);            // T1 = T1-T3 = x*x + y*y*y*y + 2*x*x*y*y*y*y - x*x - y*y*y*y = 2*x*x*y*y*y*y = 2*S
    fq_gpu::add(T1, T1, T1);            // T1 = 2T1 = 4*S
    fq_gpu::add(T0, T0, T3);            // T3 = 2T0
    fq_gpu::add(T3, T0, T3);            // T3 = T3+T0 = 3T0
    fq_gpu::add(T1, T1, T0);            // T0 = 2T1
    fq_gpu::square(T3, X);              // X = T3*T3
    fq_gpu::sub(X, T0, X);              // X = X-T0 = X-2T1
    fq_gpu::load(X, res_x);             // X = X-T0 = X-2T1

    // Z Element
    fq_gpu::add(Z, Z, Z);               // Z2 = 2Z
    fq_gpu::mul(Z, Y, res_z);           // Z2 = Z*Y = 2*Y*Z

    // Y Element
    fq_gpu::add(T2, T2, T2);            // T2 = T2+T2 = 2T2
    fq_gpu::add(T2, T2, T2);            // T2 = T2+T2 = 4T2
    fq_gpu::add(T2, T2, T2);            // T2 = T2+T2 = 8T2
    fq_gpu::sub(T1, X, Y);              // Y = T1-X
    fq_gpu::mul(Y, T3, Y);              // Y = Y*T3
    fq_gpu::sub(Y, T2, res_y);          // Y = Y - T2
}