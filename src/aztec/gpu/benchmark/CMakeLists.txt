project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../arithmetic)
include_directories(../fields)
include_directories(../curves)

# montgomery bench
add_executable(
    montgomery
    bench/montgomery.cpp
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

# field bench
add_executable(
    field_test
    tests/field_test.cu
)

target_link_libraries(
    field_test
    PUBLIC
    field_library
    fixnum_library
)

add_custom_target(
    run_field_test
    COMMAND field_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# BN254 curve test
add_executable(
    curve_bn254_test
    tests/curve_bn254_test.cu
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