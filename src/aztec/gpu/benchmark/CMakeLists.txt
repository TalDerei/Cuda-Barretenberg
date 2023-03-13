project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../arithmetic)
include_directories(../fields)
include_directories(../curves)
include_directories(../srs)

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
    tests/fields/field_fq_test.cu
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
    tests/fields/field_fr_test.cu
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

# curve test for jacobian points
add_executable(
    curve_bn254_test
    tests/curves/curve_jacobian_test.cu
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

# curve test for projective points
add_executable(
    curve_bn254_test_2
    tests/curves/curve_projective_test.cu
)

target_link_libraries(
    curve_bn254_test_2
    PUBLIC
    curve_library
    field_library
    fixnum_library
)

add_custom_target(
    run_curve_bn254_test_2
    COMMAND curve_bn254_test_2
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

# msm test
add_executable(
    msm_test
    tests/msm/msm_test.cu
)

target_link_libraries(
    msm_test
    PUBLIC
    field_library
    fixnum_library
    reference_string_library
)

add_custom_target(
    run_msm_test
    COMMAND msm_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# example test
add_executable(
    msm_example_test
    tests/msm/example_test.cu
)

target_link_libraries(
    msm_example_test
    PUBLIC
    field_library
    fixnum_library
    reference_string_library
)

add_custom_target(
    run_msm_example_test
    COMMAND msm_example_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)