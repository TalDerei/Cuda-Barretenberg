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
    main.cu
)

target_link_libraries(
    pippenger_cu
    PUBLIC
    curve_library
    field_library
    fixnum_library
    reference_string_library
    stdlib_primitives
    env
)

add_custom_target(
    run_pippenger_cu
    COMMAND pippenger_cu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_library(msm_library STATIC 
    ./pippenger.cu
)

target_link_libraries(
    msm_library
    PUBLIC
    fixnum_library
    field_library
    curve_library
    reference_string_library
)