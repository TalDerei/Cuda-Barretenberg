project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../arithmetic)

add_library(field_library STATIC 
    ./field.cu
)

target_link_libraries(
    field_library
    fixnum_library
)
