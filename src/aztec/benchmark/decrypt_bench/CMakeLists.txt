project(cuda-fixnum CUDA)

# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_SPACES}")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fconstexpr-depth=100000000")
# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

include_directories(../../cuda-fixnum)

add_executable(
    decrypt_bench
    main.cu
)

target_link_libraries(
    decrypt_bench
    PUBLIC
    sample_library
    # cuda_main
    ecc
    env
)

add_custom_target(
    run_decrypt_bench
    COMMAND decrypt_bench
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)