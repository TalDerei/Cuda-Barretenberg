project(cuda-fixnum CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../../cuda-fixnum)

add_executable(
    field_bench_cu
    field_bench.cu
)

target_link_libraries(
    field_bench_cu
    PUBLIC
    field_library
    fixnum_library
    ecc
    env
)

add_custom_target(
    run_field_bench_cu
    COMMAND field_bench_cu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(
    group_bench_cu
    group_bench.cu
)

target_link_libraries(
    group_bench_cu
    PUBLIC
    field_library
    fixnum_library
    ecc
    env
)

add_custom_target(
    run_group_bench_cu
    COMMAND group_bench_cu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)