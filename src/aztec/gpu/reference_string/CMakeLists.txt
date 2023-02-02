project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../curves)
include_directories(../fields)
include_directories(../arithmetic)

add_library(reference_string_library STATIC 
    ./reference_string.cu
    ./reference_string_methods.cu
    ./io.cu
)

# add_executable(
#     sample_gpu_test
#     ./main.cu
# )

# target_link_libraries(
#     sample_gpu_test
#     PUBLIC
#     field_library
#     curve_library
#     fixnum_library
# )

# add_custom_target(
#     run_sample_gpu_test
#     COMMAND sample_gpu_test
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
# )

