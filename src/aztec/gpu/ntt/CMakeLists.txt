project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../arithmetic)
include_directories(../fields)
include_directories(../curves)

# NTT
add_executable(
    ntt
    ntt.cu
)

target_link_libraries(
    ntt
    PUBLIC
    ecc
    env
)

add_custom_target(
    run_ntt
    COMMAND ntt
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
