project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../bignum)
include_directories(../fields)
include_directories(../curves)
include_directories(../srs)

add_executable(
    arithmetic_cuda
    arithmetic.cu
)

add_custom_target(
    run_arithmetic_cuda
    COMMAND arithmetic_cuda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)