project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../arithmetic)
include_directories(../fields)
include_directories(../curves)

# montgomery multiplication
add_executable(
    montgomery
    common/montgomery.cpp
)

target_link_libraries(
    montgomery
    PUBLIC
    ecc
    env
)

add_custom_target(
    run_montgomery
    COMMAND montgomery
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# fq field test
add_executable(
    field_fq_test
    tests/field_fq_test.cu
)

target_link_libraries(
    field_fq_test
    PUBLIC
    field_library
    fixnum_library
)

add_custom_target(
    run_field_fq_test
    COMMAND field_fq_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# fr field test
add_executable(
    field_fr_test
    tests/field_fr_test.cu
)

target_link_libraries(
    field_fr_test
    PUBLIC
    field_library
    fixnum_library
)

add_custom_target(
    run_field_fr_test
    COMMAND field_fr_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# curve test
add_executable(
    curve_bn254_test
    tests/curve_test.cu
)

target_link_libraries(
    curve_bn254_test
    PUBLIC
    curve_library
    field_library
    fixnum_library
)

add_custom_target(
    run_curve_bn254_test
    COMMAND curve_bn254_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# arithmetic bench
add_executable(
    arithmetic_cuda
    bench/arithmetic.cu
)

target_link_libraries(
    arithmetic_cuda
    PUBLIC
    curve_library
    field_library
    fixnum_library
)

add_custom_target(
    run_arithmetic_cuda
    COMMAND arithmetic_cuda
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)