project(cuda-fixnum CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../../cuda-fixnum)

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(
    fq_test
    fq_test.test.cu
)

target_link_libraries(
    fq_test
    PUBLIC
    sample_library
    cuda_main
    ecc
    env
    ${GTEST_LIBRARIES}
)

add_custom_target(
    run_fq
    COMMAND fq_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)