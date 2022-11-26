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

// Elliptic curve operations: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-zadd-2007-m
template <class fq_gpu>
__device__ void group_gpu<fq_gpu>::mixed_add(var X, var Y, var Z, var A, var B, var &res_x, var &res_y, var &res_z) noexcept {
    var z1z1, u2, s2, h, hh, i, j, r, v, t0, t1;

    // X element
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

    // Y element
    fq_gpu::sub(v, res_x, t0);  
    fq_gpu::mul(Y, j, t1);     
    fq_gpu::add(t1, t1, t1);
    fq_gpu::mul(t0, r, t0);     
    fq_gpu::sub(t0, t1, res_y);

    // Z element
    fq_gpu::add(Z, h, t0);      
    fq_gpu::square(t0, t0);     
    fq_gpu::sub(t0, z1z1, t0);  
    fq_gpu::sub(t0, hh, res_z); 
}

template <class fq_gpu>
__device__ void group_gpu<fq_gpu>::doubling(var X, var Y, var Z, var &res_x, var &res_y, var &res_z) noexcept {
    var xx, yy, yyyy, zz, s, m, t, t0, t1;
    
    fq_gpu::square(X, xx);
    fq_gpu::square(Y, yy);
    fq_gpu::square(yy, yyyy);
    fq_gpu::square(Z, zz);
    fq_gpu::add(X, yy, t0);
    fq_gpu::square(t0, t0);
    fq_gpu::add(xx, yyyy, t1);
    fq_gpu::sub(t0, t1, t0);
    fq_gpu::add(t0, t0, s);

    fq_gpu::add(t0, t0, xx);
    fq_gpu::add(xx, t0, xx);
    fq_gpu::square(zz, t1);
    fq_gpu::add(t1, t1, t1);
    fq_gpu::add(t0, t1, m);
    fq_gpu::square(m, t0);
    fq_gpu::add(s, s, t1);
    fq_gpu::sub(t0, t1, t);

    // X element
    res_x = t;

    // Z element
    fq_gpu::add(Y, Z, t0);
    fq_gpu::square(t0, t0);
    fq_gpu::square(t0, t0);
    fq_gpu::add(yy, zz, t1);
    fq_gpu::sub(t0, t1, res_z);
    
    // Y element
    fq_gpu::sub(s, t, t0);
    fq_gpu::mul(m, t0, t0); 
    fq_gpu::add(yyyy, yyyy, t1);
    fq_gpu::add(t1, t1, t1);
    fq_gpu::add(t1, t1, t1);
    fq_gpu::sub(t0, t1, res_y);
}
