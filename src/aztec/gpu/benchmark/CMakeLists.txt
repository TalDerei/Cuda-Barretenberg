project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../arithmetic)
include_directories(../fields)
include_directories(../groups)

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
    field_cu_test
    tests/field_test.cu
)

target_link_libraries(
    field_cu_test
    PUBLIC
    field_library
    fixnum_library
)

add_custom_target(
    run_field_cu_test
    COMMAND field_cu_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# group test
add_executable(
    group_test_cu
    tests/group_test.cu
)

target_link_libraries(
    group_test_cu
    PUBLIC
    group_library
    field_library
    fixnum_library
)

add_custom_target(
    run_group_test_cu
    COMMAND group_test_cu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)