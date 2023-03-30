#include "group.cuh"

using namespace std;
using namespace gpu_barretenberg;

/* -------------------------- Affine and Jacobian Coordinate Operations ---------------------------------------------- */

template <class fq_gpu, class fr_gpu> 
__device__ void group_gpu<fq_gpu, fr_gpu>::load_affine(affine_element &X, affine_element &result) {
    fq_gpu::load(X.x.data[0], result.x.data[0]);      
    fq_gpu::load(X.x.data[1], result.x.data[1]);      
    fq_gpu::load(X.x.data[2], result.x.data[2]);      
    fq_gpu::load(X.x.data[3], result.x.data[3]);    
        
    fq_gpu::load(X.x.data[0], result.y.data[0]);      
    fq_gpu::load(X.x.data[1], result.y.data[1]);      
    fq_gpu::load(X.x.data[2], result.y.data[2]);      
    fq_gpu::load(X.x.data[3], result.y.data[3]);  
}

template <class fq_gpu, class fr_gpu> 
__device__ void group_gpu<fq_gpu, fr_gpu>::load_jacobian(element &X, element &result) {
    fq_gpu::load(X.x.data[0], result.x.data[0]);      
    fq_gpu::load(X.x.data[1], result.x.data[1]);      
    fq_gpu::load(X.x.data[2], result.x.data[2]);      
    fq_gpu::load(X.x.data[3], result.x.data[3]);    
        
    fq_gpu::load(X.x.data[0], result.y.data[0]);      
    fq_gpu::load(X.x.data[1], result.y.data[1]);      
    fq_gpu::load(X.x.data[2], result.y.data[2]);      
    fq_gpu::load(X.x.data[3], result.y.data[3]);  

    fq_gpu::load(X.z.data[0], result.z.data[0]);      
    fq_gpu::load(X.z.data[1], result.z.data[1]);      
    fq_gpu::load(X.z.data[2], result.z.data[2]);      
    fq_gpu::load(X.z.data[3], result.z.data[3]);  
}


/**
 * Elliptic curve algorithms: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-zadd-2007-m
 */
template <class fq_gpu, class fr_gpu> 
__device__ void group_gpu<fq_gpu, fr_gpu>::mixed_add(var X, var Y, var Z, var A, var B, var &res_x, var &res_y, var &res_z) {
    var z1z1, u2, s2, h, hh, i, j, r, v, t0, t1;
    
    // X Element
    fq_gpu::square(Z, z1z1);   
    fq_gpu::mul(A, z1z1, u2); 
    fq_gpu::mul(B, Z, s2);
    fq_gpu::mul(s2, z1z1, s2); 

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

template <class fq_gpu, class fr_gpu> 
__device__ void group_gpu<fq_gpu, fr_gpu>::doubling(var X, var Y, var Z, var &res_x, var &res_y, var &res_z) {
    var T0, T1, T2, T3;

   // Check P == 0
    if (fq_gpu::is_zero(Z)) {
        fq_gpu::zero();
    }

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

/**
 * Jacobian addition has cost 16T multiplications
 */
template <class fq_gpu, class fr_gpu> 
__device__ void group_gpu<fq_gpu, fr_gpu>::add(var X1, var Y1, var Z1, var X2, var Y2, var Z2, var &res_x, var &res_y, var &res_z) {
    var Z1Z1, Z2Z2, U1, U2, S1, S2, F, H, I, J;

    // Check P == 0 or Q == 0
    if (fq_gpu::is_zero(Z1)) {
        res_x = X2;
        res_y = Y2;
        res_z = Z2;
        return;
    } else if (fq_gpu::is_zero(Z2)) {
        res_x = X1;
        res_y = Y1;
        res_z = Z1;
        return;
    }

    // X Element
    fq_gpu::square(Z1, Z1Z1);            // Z1Z1 = Z1^2
    fq_gpu::square(Z2, Z2Z2);            // Z1Z1 = Z2^2
    fq_gpu::mul(Z1Z1, Z1, S2);           // S2 = Z1Z1 * Z1 
    fq_gpu::mul(Z1Z1, X2, U2);           // U2 = Z1Z1 * X2
    fq_gpu::mul(S2, Y2, S2);             // S2 = S2 * Y2
    fq_gpu::mul(Z2Z2, X1, U1);           // U1 = Z2Z2 * X1
    fq_gpu::mul(Z2Z2, Z2, S1);           // S1 = Z2Z2 * Z2
    fq_gpu::mul(S1, Y1, S1);             // S1 = S1 * Y1
    fq_gpu::sub(S2, S1, F);              // F = S2 - S1
    fq_gpu::sub(U2, U1, H);              // H = U2 - U1
    fq_gpu::add(F, F, F);                // F = F + F
    fq_gpu::add(H, H, I);                // I = H + H
    fq_gpu::square(I, I);                // I = I * H
    fq_gpu::mul(H, I, J);                // J = H * H
    fq_gpu::mul(U1, I, U1);              // U1 = U1 * I
    fq_gpu::add(U1, U1, U2);             // U2 = U1 + U1
    fq_gpu::add(U2, J, U2);              // U2 = U2 * J
    fq_gpu::square(F, X1);               // X1 = F^2
    fq_gpu::sub(X1, U2, X1);             // X1 = X1 - U2
    fq_gpu::load(X1, res_x);             // res_x = X1

    // Y Element
    fq_gpu::mul(J, S1, J);               // J = J * S1
    fq_gpu::add(J, J, J);                // J = J + J
    fq_gpu::sub(U1, X1, Y1);             // Y1 = U1 - X1
    fq_gpu::mul(Y1, F, Y1);              // Y1 = Y1 + F
    fq_gpu::sub(Y1, J, Y1);              // Y1 = Y1 - J
    fq_gpu::load(Y1, res_y);             // res_y = Y1

    // Z Element
    fq_gpu::add(Z1, Z2, Z1);             // Z1 = Z1 + Z2
    fq_gpu::add(Z1Z1, Z2Z2, Z1Z1);       // Z1Z1 = Z2Z2 + Z1Z1
    fq_gpu::square(Z1, Z1);              // Z1 = Z1^2
    fq_gpu::sub(Z1, Z1Z1, Z1);           // Z1 = Z1 - Z1Z1
    fq_gpu::mul(Z1, H, Z1);              // Z1 = Z1 * H
    fq_gpu::load(Z1, res_z);             // res_z = Z1;
}

template <class fq_gpu, class fr_gpu> 
__device__ void group_gpu<fq_gpu, fr_gpu>::add_projective(
var X1, var Y1, var Z1, var X2, var Y2, var Z2, var &res_x, var &res_y, var &res_z) {
    var t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10;
    var t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21;
    var t22, t23, t24, t25, t26, t27, t28, t29, t30, t31;
    var X3, Y3, Z3;

    fq_gpu::mul(X1, X2, t00);                                       // t00 ← X1 · X2
    fq_gpu::mul(Y1, Y2, t01);                                       // t01 ← Y1 · Y2
    fq_gpu::mul(Z1, Z2, t02);                                       // t02 ← Z1 · Z2
    fq_gpu::add(X1, Y1, t03);                                       // t03 ← X1 + Y1
    fq_gpu::add(X2, Y2, t04);                                       // t04 ← X2 + Y2
    fq_gpu::mul(t03, t04, t03);                                     // t03 ← t03 + t04
    fq_gpu::add(t00, t01, t04);                                     // t04 ← t00 + t01
    fq_gpu::sub(t03, t04, t03);                                     // t03 ← t03 - t04
    fq_gpu::add(X1, Z1, t04);                                       // t04 ← X1 + Z1
    fq_gpu::add(X2, Z2, t05);                                       // t05 ← X2 + Z2
    fq_gpu::mul(t04, t05, t04);                                     // t04 ← t04 * t05
    fq_gpu::add(t00, t02, t05);                                     // t05 ← t00 + t02
    fq_gpu::sub(t04, t05, t04);                                     // t04 ← t04 - t05
    fq_gpu::add(Y1, Z1, t05);                                       // t05 ← Y1 + Z1
    fq_gpu::add(Y2, Z2, X3);                                        // X3 ← Y2 + Z23
    fq_gpu::mul(t05, X3, t05);                                      // t05 ← t05 * X3
    fq_gpu::add(t01, t02, X3);                                      // X3 ← t01 + t02
    fq_gpu::sub(t05, X3, t05);                                      // t05 ← t05 - X3
    fq_gpu::mul(0, t04, Z3);                                        // Z3 ← a * t04
    fq_gpu::mul(3 * gpu_barretenberg::b, t02, X3);                  // X3 ← b3 * t02 
    fq_gpu::add(X3, Z3, Z3);                                        // Z3 ← X3 + Z3
    fq_gpu::sub(t01, Z3, X3);                                       // X3 ← t01 - Z3
    fq_gpu::add(t01, Z3, Z3);                                       // Z3 ← t01 + Z3
    fq_gpu::mul(X3, Z3, Y3);                                        // Y3 ← X3 * Z3
    fq_gpu::add(t00, t00, t01);                                     // t01 ← t00 + t00
    fq_gpu::add(t01, t00, t01);                                     // t01 ← t01 + t00
    fq_gpu::mul(0, t02, t02);                                       // t02 ← a * t02
    fq_gpu::mul(3 * gpu_barretenberg::b, t04, t04);                 // t04 ← b3 * t04 
    fq_gpu::add(t01, t02, t01);                                     // t01 ← t01 + t02
    fq_gpu::sub(t00, t02, t02);                                     // t02 ← t00 - t02
    fq_gpu::mul(0, t02, t02);                                       // t02 ← a * t02
    fq_gpu::add(t04, t02, t04);                                     // t04 ← t04 + t02
    fq_gpu::mul(t01, t04, t00);                                     // t00 ← t01 * t04
    fq_gpu::add(Y3, t00, Y3);                                       // Y3 ← Y3 + t00
    fq_gpu::mul(t05, t04, t00);                                     // t00 ← t05 * t04
    fq_gpu::mul(t03, X3, X3);                                       // X3 ← t03 * X3
    fq_gpu::sub(X3, t00, X3);                                       // X3 ← X3 - t00
    fq_gpu::mul(t03, t01, t00);                                     // t00 ← t03 * t01
    fq_gpu::mul(t05, Z3, Z3);                                       // Z3 ← t05 * Z3
    fq_gpu::add(Z3, t00, Z3);                                       // Z3 ← Z3 + t00

    fq_gpu::load(X3, res_x);  
    fq_gpu::load(Y3, res_y);  
    fq_gpu::load(Z3, res_z);  

    // fq_gpu::mul(X1, X2, t00);                                   // t00 ← X1 · X2     < 2
    // fq_gpu::mul(Y1, Y2, t01);                                   // t01 ← Y1 · Y2     < 2
    // fq_gpu::mul(Z1, Z2, t02);                                   // t02 ← Z1 · Z2     < 2
    // fq_gpu::add(X1, Y1, t03);                                   // t03 ← X1 + Y1     < 4
    // fq_gpu::add(X2, Y2, t04);                                   // t04 ← X2 + Y2     < 4
    // fq_gpu::mul(t03, t04, t05);                                 // t05 ← t03 · t04   < 3
    // fq_gpu::add(t00, t01, t06);                                 // t06 ← t00 + t01   < 4
    // fq_gpu::sub(t05, t06, t07);                                 // t07 ← t05 − t06   < 2
    // fq_gpu::add(Y1, Z1, t08);                                   // t08 ← Y1 + Z1     < 4
    // fq_gpu::add(Y2, Z2, t09);                                   // t09 ← Y2 + Z2     < 4
    // fq_gpu::mul(t08, t09, t10);                                 // t10 ← t08 · t09   < 3
    // fq_gpu::add(t01, t02, t11);                                 // t11 ← t01 + t02   < 4
    // fq_gpu::sub(t10, t11, t12);                                 // t12 ← t10 − t11   < 2
    // fq_gpu::add(X1, Z1, t13);                                   // t13 ← X1 + Z1     < 4
    // fq_gpu::add(X2, Z2, t14);                                   // t14 ← X2 + Z2     < 4
    // fq_gpu::mul(t13, t14, t15);                                 // t15 ← t13 · t14   < 3
    // fq_gpu::add(t00, t02, t16);                                 // t16 ← t00 + t02   < 4
    // fq_gpu::sub(t15, t16, t17);                                 // t17 ← t15 − t16   < 2
    // fq_gpu::add(t00, t00, t18);                                 // t18 ← t00 + t00   < 2
    // fq_gpu::add(t18, t00, t19);                                 // t19 ← t18 + t00   < 2
    // fq_gpu::mul(3 * gpu_barretenberg::b, t02, t20);             // t20 ← b3 · t02    < 2
    // fq_gpu::add(t01, t20, t21);                                 // t21 ← t01 + t20   < 2
    // fq_gpu::sub(t01, t20, t22);                                 // t22 ← t01 − t20   < 2
    // fq_gpu::mul(3 * gpu_barretenberg::b, t17, t23);             // t23 ← b3 · t17    < 2
    // fq_gpu::mul(t12, t23, t24);                                 // t24 ← t12 · t23   < 2
    // fq_gpu::mul(t07, t22, t25);                                 // t25 ← t07 · t22   < 2
    // fq_gpu::sub(t25, t24, res_x);                               // X3 ← t25 − t24    < 2
    // fq_gpu::mul(t23, t19, t27);                                 // t27 ← t23 · t19   < 2
    // fq_gpu::mul(t22, t21, t28);                                 // t28 ← t22 · t21   < 2
    // fq_gpu::add(t28, t27, res_y);                               // Y3 ← t28 + t27    < 2
    // fq_gpu::mul(t19, t07, t30);                                 // t30 ← t19 · t07   < 2
    // fq_gpu::mul(t21, t12, t31);                                 // t31 ← t21 · t12   < 2
    // fq_gpu::add(t31, t30, res_z);                               // Z3 ← t31 + t30    < 2
}

/* -------------------------- Projective Coordinate Operations ---------------------------------------------- */

template <class fq_gpu, class fr_gpu> 
__device__ void group_gpu<fq_gpu, fr_gpu>::load_projective(projective_element &X, projective_element &result) {
    fq_gpu::load(X.x.data[0], result.x.data[0]);      
    fq_gpu::load(X.x.data[1], result.x.data[1]);      
    fq_gpu::load(X.x.data[2], result.x.data[2]);      
    fq_gpu::load(X.x.data[3], result.x.data[3]);    
        
    fq_gpu::load(X.x.data[0], result.y.data[0]);      
    fq_gpu::load(X.x.data[1], result.y.data[1]);      
    fq_gpu::load(X.x.data[2], result.y.data[2]);      
    fq_gpu::load(X.x.data[3], result.y.data[3]);  

    fq_gpu::load(X.z.data[0], result.z.data[0]);      
    fq_gpu::load(X.z.data[1], result.z.data[1]);      
    fq_gpu::load(X.z.data[2], result.z.data[2]);      
    fq_gpu::load(X.z.data[3], result.z.data[3]);  
}

template <class fq_gpu, class fr_gpu> 
projective_element<fq_gpu, fr_gpu> group_gpu<fq_gpu, fr_gpu>::from_affine(const affine_element &other) {
    projective_element projective;
    projective.x = other.x;
    projective.y = other.y;
    return { projective.x, projective.y, fq_gpu::one() };
}

/**
 * Projective addition has cost 14T multiplications
 */
// template <class fq_gpu, class fr_gpu> 
// __device__ void group_gpu<fq_gpu, fr_gpu>::add_projective(
// var X1, var Y1, var Z1, var X2, var Y2, var Z2, var &res_x, var &res_y, var &res_z) {
//     var t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10;
//     var t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21;
//     var t22, t23, t24, t25, t26, t27, t28, t29, t30, t31;
//     var X3, Y3, Z3;

//     fq_gpu::mul(X1, X2, t00);                                       // t00 ← X1 · X2
//     fq_gpu::mul(Y1, Y2, t01);                                       // t01 ← Y1 · Y2
//     fq_gpu::mul(Z1, Z2, t02);                                       // t02 ← Z1 · Z2
//     fq_gpu::add(X1, Y1, t03);                                       // t03 ← X1 + Y1
//     fq_gpu::add(X2, Y2, t04);                                       // t04 ← X2 + Y2
//     fq_gpu::mul(t03, t04, t03);                                     // t03 ← t03 + t04
//     fq_gpu::add(t00, t01, t04);                                     // t04 ← t00 + t01
//     fq_gpu::sub(t03, t04, t03);                                     // t03 ← t03 - t04
//     fq_gpu::add(X1, Z1, t04);                                       // t04 ← X1 + Z1
//     fq_gpu::add(X2, Z2, t05);                                       // t05 ← X2 + Z2
//     fq_gpu::mul(t04, t05, t04);                                     // t04 ← t04 * t05
//     fq_gpu::add(t00, t02, t05);                                     // t05 ← t00 + t02
//     fq_gpu::sub(t04, t05, t04);                                     // t04 ← t04 - t05
//     fq_gpu::add(Y1, Z1, t05);                                       // t05 ← Y1 + Z1
//     fq_gpu::add(Y2, Z2, X3);                                        // X3 ← Y2 + Z23
//     fq_gpu::mul(t05, X3, t05);                                      // t05 ← t05 * X3
//     fq_gpu::add(t01, t02, X3);                                      // X3 ← t01 + t02
//     fq_gpu::sub(t05, X3, t05);                                      // t05 ← t05 - X3
//     fq_gpu::mul(0, t04, Z3);                                        // Z3 ← a * t04
//     fq_gpu::mul(3 * gpu_barretenberg::b, t02, X3);                  // X3 ← b3 * t02 
//     fq_gpu::add(X3, Z3, Z3);                                        // Z3 ← X3 + Z3
//     fq_gpu::sub(t01, Z3, X3);                                       // X3 ← t01 - Z3
//     fq_gpu::add(t01, Z3, Z3);                                       // Z3 ← t01 + Z3
//     fq_gpu::mul(X3, Z3, Y3);                                        // Y3 ← X3 * Z3
//     fq_gpu::add(t00, t00, t01);                                     // t01 ← t00 + t00
//     fq_gpu::add(t01, t00, t01);                                     // t01 ← t01 + t00
//     fq_gpu::mul(0, t02, t02);                                       // t02 ← a * t02
//     fq_gpu::mul(3 * gpu_barretenberg::b, t04, t04);                 // t04 ← b3 * t04 
//     fq_gpu::add(t01, t02, t01);                                     // t01 ← t01 + t02
//     fq_gpu::sub(t00, t02, t02);                                     // t02 ← t00 - t02
//     fq_gpu::mul(0, t02, t02);                                       // t02 ← a * t02
//     fq_gpu::add(t04, t02, t04);                                     // t04 ← t04 + t02
//     fq_gpu::mul(t01, t04, t00);                                     // t00 ← t01 * t04
//     fq_gpu::add(Y3, t00, Y3);                                       // Y3 ← Y3 + t00
//     fq_gpu::mul(t05, t04, t00);                                     // t00 ← t05 * t04
//     fq_gpu::mul(t03, X3, X3);                                       // X3 ← t03 * X3
//     fq_gpu::sub(X3, t00, X3);                                       // X3 ← X3 - t00
//     fq_gpu::mul(t03, t01, t00);                                     // t00 ← t03 * t01
//     fq_gpu::mul(t05, Z3, Z3);                                       // Z3 ← t05 * Z3
//     fq_gpu::add(Z3, t00, Z3);                                       // Z3 ← Z3 + t00

//     fq_gpu::load(X3, res_x);  
//     fq_gpu::load(Y3, res_y);  
//     fq_gpu::load(Z3, res_z);  

//     // fq_gpu::mul(X1, X2, t00);                                   // t00 ← X1 · X2     < 2
//     // fq_gpu::mul(Y1, Y2, t01);                                   // t01 ← Y1 · Y2     < 2
//     // fq_gpu::mul(Z1, Z2, t02);                                   // t02 ← Z1 · Z2     < 2
//     // fq_gpu::add(X1, Y1, t03);                                   // t03 ← X1 + Y1     < 4
//     // fq_gpu::add(X2, Y2, t04);                                   // t04 ← X2 + Y2     < 4
//     // fq_gpu::mul(t03, t04, t05);                                 // t05 ← t03 · t04   < 3
//     // fq_gpu::add(t00, t01, t06);                                 // t06 ← t00 + t01   < 4
//     // fq_gpu::sub(t05, t06, t07);                                 // t07 ← t05 − t06   < 2
//     // fq_gpu::add(Y1, Z1, t08);                                   // t08 ← Y1 + Z1     < 4
//     // fq_gpu::add(Y2, Z2, t09);                                   // t09 ← Y2 + Z2     < 4
//     // fq_gpu::mul(t08, t09, t10);                                 // t10 ← t08 · t09   < 3
//     // fq_gpu::add(t01, t02, t11);                                 // t11 ← t01 + t02   < 4
//     // fq_gpu::sub(t10, t11, t12);                                 // t12 ← t10 − t11   < 2
//     // fq_gpu::add(X1, Z1, t13);                                   // t13 ← X1 + Z1     < 4
//     // fq_gpu::add(X2, Z2, t14);                                   // t14 ← X2 + Z2     < 4
//     // fq_gpu::mul(t13, t14, t15);                                 // t15 ← t13 · t14   < 3
//     // fq_gpu::add(t00, t02, t16);                                 // t16 ← t00 + t02   < 4
//     // fq_gpu::sub(t15, t16, t17);                                 // t17 ← t15 − t16   < 2
//     // fq_gpu::add(t00, t00, t18);                                 // t18 ← t00 + t00   < 2
//     // fq_gpu::add(t18, t00, t19);                                 // t19 ← t18 + t00   < 2
//     // fq_gpu::mul(3 * gpu_barretenberg::b, t02, t20);             // t20 ← b3 · t02    < 2
//     // fq_gpu::add(t01, t20, t21);                                 // t21 ← t01 + t20   < 2
//     // fq_gpu::sub(t01, t20, t22);                                 // t22 ← t01 − t20   < 2
//     // fq_gpu::mul(3 * gpu_barretenberg::b, t17, t23);             // t23 ← b3 · t17    < 2
//     // fq_gpu::mul(t12, t23, t24);                                 // t24 ← t12 · t23   < 2
//     // fq_gpu::mul(t07, t22, t25);                                 // t25 ← t07 · t22   < 2
//     // fq_gpu::sub(t25, t24, res_x);                               // X3 ← t25 − t24    < 2
//     // fq_gpu::mul(t23, t19, t27);                                 // t27 ← t23 · t19   < 2
//     // fq_gpu::mul(t22, t21, t28);                                 // t28 ← t22 · t21   < 2
//     // fq_gpu::add(t28, t27, res_y);                               // Y3 ← t28 + t27    < 2
//     // fq_gpu::mul(t19, t07, t30);                                 // t30 ← t19 · t07   < 2
//     // fq_gpu::mul(t21, t12, t31);                                 // t31 ← t21 · t12   < 2
//     // fq_gpu::add(t31, t30, res_z);                               // Z3 ← t31 + t30    < 2
// }