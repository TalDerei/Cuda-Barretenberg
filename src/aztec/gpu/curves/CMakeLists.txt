project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../fields)
include_directories(../bignum)

add_library(curve_library STATIC 
    ./group.cu
    ./element.cu
)

target_link_libraries(
    curve_library
    fixnum_library
)