project(cuda-fixnum CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../../cuda-fixnum)

add_executable(
    mont_mult
    mont_mult.cpp
)

target_link_libraries(
    mont_mult
    PUBLIC
    fixnum_library
    field_library
    ecc
    env
)

add_custom_target(
    run_mont_mult
    COMMAND mont_mult
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(
    mont_mult_cu
    mont_mult.cu
)

target_link_libraries(
    mont_mult_cu
    PUBLIC
    field_library
    fixnum_library
    ecc
    env
)

add_custom_target(
    run_mont_mult_cu
    COMMAND mont_mult_cu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)