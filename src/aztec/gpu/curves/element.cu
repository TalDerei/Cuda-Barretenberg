#include "element.cuh"

using namespace std;
using namespace gpu_barretenberg;
using namespace group_elements;

template <typename fq_gpu> 
__device__ element<fq_gpu>::element(const fq_gpu &a, const fq_gpu &b, const fq_gpu &c) noexcept
    : x{a}, y{b}, z{c} {};

template <typename fq_gpu> 
__device__ element<fq_gpu>::element(const element &other) noexcept
    : x(other.x), y(other.y), z(other.z) {};

template <typename fq_gpu> 
__device__ affine_element<fq_gpu>::affine_element(const fq_gpu &a, const fq_gpu &b) noexcept 
    : x(a), y(b) {};

template <typename fq_gpu> 
__device__ affine_element<fq_gpu>::affine_element(const affine_element &other) noexcept 
    : x(other.x), y(other.y) {};