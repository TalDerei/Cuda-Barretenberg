project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../bignum)
include_directories(../fields)
include_directories(../curves)
include_directories(../msm)
include_directories(../srs)

# Plonk
add_executable(
    plonk_cu
    plonk.cu
)

target_link_libraries(
    plonk_cu
    PUBLIC
    curve_library
    field_library
    fixnum_library
    reference_string_library
    stdlib_primitives
    env
    msm_library
)

add_custom_target(
    run_plonk_cu
    COMMAND plonk_cu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

