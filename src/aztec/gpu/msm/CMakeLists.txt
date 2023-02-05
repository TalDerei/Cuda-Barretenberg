project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../arithmetic)
include_directories(../fields)
include_directories(../curves)
include_directories(../srs)

# Pippenger's bucket method
add_executable(
    pippenger_cu
    pippenger.cu
)

target_link_libraries(
    pippenger_cu
    PUBLIC
    field_library
    curve_library
    fixnum_library
    reference_string_library
)

add_custom_target(
    run_pippenger_cu
    COMMAND pippenger_cu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)