project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../bignum)
include_directories(../fields)
include_directories(../curves)
include_directories(../srs)

# Unit tests: (1) field (fq, fr), (2) curve (jacobian and projective), 
# (3) msm, (4) sum reduction

# Add executables
add_executable(
    field_fq_test
    fields/field_fq_test.cu
)
add_executable(
    field_fr_test
    fields/field_fr_test.cu
)
add_executable(
    curve_bn254_test
    curves/curve_jacobian_test.cu
)
add_executable(
    curve_bn254_test_2
    curves/curve_projective_test.cu
)
add_executable(
    msm_test_cpu
    msm/msm_test_cpu.cpp
)
add_executable(
    msm_test_gpu
    msm/msm_test_gpu.cu
)
add_executable(
    sum_reduction_test
    msm/sum_reduction_test.cu
)

# Add custom targets
add_custom_target(
    run_field_fq_test
    COMMAND field_fq_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_custom_target(
    run_field_fr_test
    COMMAND field_fr_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_custom_target(
    run_curve_bn254_test
    COMMAND curve_bn254_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_custom_target(
    run_curve_bn254_test_2
    COMMAND curve_bn254_test_2
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_custom_target(
    run_msm_test_cpu
    COMMAND msm_test_cpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_custom_target(
    run_msm_test_gpu
    COMMAND msm_test_gpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_custom_target(
    run_sum_reduction_test
    COMMAND sum_reduction_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)