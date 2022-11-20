project(cuda-fixnum CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../../cuda-fixnum)

add_executable(
    montgomery
    montgomery.cpp
)

target_link_libraries(
    montgomery
    PUBLIC
    fixnum_library
    field_library
    ecc
    env
)

add_custom_target(
    run_montgomery
    COMMAND montgomery
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)