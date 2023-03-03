#include "element.cuh"

using namespace std;
using namespace gpu_barretenberg;
using namespace group_elements;

template <class fq_gpu, class fr_gpu> 
__device__ element<fq_gpu, fr_gpu>::element(const fq_gpu &a, const fq_gpu &b, const fq_gpu &c) noexcept
    : x{a}, y{b}, z{c} {};

template <class fq_gpu, class fr_gpu> 
__device__ element<fq_gpu, fr_gpu>::element(const element &other) noexcept
    : x(other.x), y(other.y), z(other.z) {};

template <class fq_gpu, class fr_gpu> 
__device__ affine_element<fq_gpu, fr_gpu>::affine_element(const fq_gpu &a, const fq_gpu &b) noexcept 
    : x(a), y(b) {};

template <class fq_gpu, class fr_gpu> 
__device__ affine_element<fq_gpu, fr_gpu>::affine_element(const affine_element &other) noexcept 
    : x(other.x), y(other.y) {};

template <class fq_gpu, class fr_gpu> 
constexpr bool element<fq_gpu, fr_gpu>::operator==(const element& other) const noexcept
{
    const fq_gpu lhs_zz = z.sqr();
    const fq_gpu lhs_zzz = lhs_zz * z;
    const fq_gpu rhs_zz = other.z.sqr();
    const fq_gpu rhs_zzz = rhs_zz * other.z;

    const fq_gpu lhs_x = x * rhs_zz;
    const fq_gpu lhs_y = y * rhs_zzz;

    const fq_gpu rhs_x = other.x * lhs_zz;
    const fq_gpu rhs_y = other.y * lhs_zzz;
    return ((lhs_x == rhs_x) && (lhs_y == rhs_y));
}