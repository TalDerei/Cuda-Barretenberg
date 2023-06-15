project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../curves)
include_directories(../fields)
include_directories(../bignum)

add_library(reference_string_library STATIC 
    ./reference_string.cu
    ./reference_string_methods.cu
    ./io.cu
)