project(cuda-fixnum CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CUDA_STANDARD 11)
# set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")

# # CUDA
# find_package(CUDA REQUIRED)
# set(CUDA_ARCH_LIST Auto CACHE LIST
#     "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
#      compute capability versions (6.1, 7.0, etc) to generate code for. \
#      Set to Auto for automatic detection (default)."
# )
# cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
# string(REPLACE ";" " " CUDA_ARCH_FLAGS_SPACES "${CUDA_ARCH_FLAGS}")

# string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_SPACES}")
# # string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fconstexpr-steps=100000000")
# set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# source
# include_directories(src)

# bench
# add_executable(cuda_bench bench/bench.cu)
# target_compile_definitions(cuda_bench PUBLIC -DNDEBUG)
# target_link_libraries(cuda_bench ecc env -lstdc++)

add_library(cuda_main STATIC main.cu)
target_include_directories(cuda_main PUBLIC src)
set_property(TARGET cuda_main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_library(sample_library STATIC 
    sample.cpp
    sample.h
)
target_compile_features(sample_library PRIVATE cxx_std_20)

# add_executable(example_cu example.cu)
# target_link_libraries(example_cu ecc env sample_library -lstdc++)

# add_custom_target(
#     run_example_cu
#     COMMAND example_cu
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
# )

# add_library(sample_library_new STATIC 
#     src/functions/sample-new.h
#     src/functions/sample-new.cpp
# )  
# target_compile_features(sample_library_new PRIVATE cxx_std_14)
# target_link_libraries(sample_library_new -lstdc++)

# add_library (
#     cuda_func STATIC
#     src/functions/modinv.cu
# )
# target_include_directories(cuda_func PUBLIC src)
# target_compile_features(cuda_func PUBLIC cxx_std_11)
# target_link_libraries(cuda_func -lstdc++)
