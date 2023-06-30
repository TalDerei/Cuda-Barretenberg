#include "element.cuh"

using namespace std;
using namespace gpu_barretenberg;
using namespace gpu_group_elements;

/* -------------------------- Jacobian Elements ---------------------------------------------- */

template <class fq_gpu, class fr_gpu> 
__device__ __forceinline__ element<fq_gpu, fr_gpu>::element(const fq_gpu &a, const fq_gpu &b, const fq_gpu &c) noexcept
    : x{a}, y{b}, z{c} {};

template <class fq_gpu, class fr_gpu> 
__device__ __forceinline__ element<fq_gpu, fr_gpu>::element(const element &other) noexcept
    : x(other.x), y(other.y), z(other.z) {};

/* -------------------------- Affine Elements ---------------------------------------------- */

template <class fq_gpu, class fr_gpu> 
__device__ __forceinline__ affine_element<fq_gpu, fr_gpu>::affine_element(const fq_gpu &a, const fq_gpu &b) noexcept 
    : x(a), y(b) {};

template <class fq_gpu, class fr_gpu> 
__device__ __forceinline__ affine_element<fq_gpu, fr_gpu>::affine_element(const affine_element &other) noexcept 
    : x(other.x), y(other.y) {};

/* -------------------------- Projective Elements ---------------------------------------------- */

template <class fq_gpu, class fr_gpu> 
__device__ __forceinline__ projective_element<fq_gpu, fr_gpu>::projective_element(const fq_gpu &a, const fq_gpu &b, const fq_gpu &c) noexcept
    : x{a}, y{b}, z{c} {};

template <class fq_gpu, class fr_gpu> 
__device__ __forceinline__ projective_element<fq_gpu, fr_gpu>::projective_element(const projective_element &other) noexcept
    : x(other.x), y(other.y), z(other.z) {};