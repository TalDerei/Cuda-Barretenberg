project(cuda-fixnum CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../../cuda-fixnum)

add_library(field_library STATIC 
    ./field.cu
    ./group.cu
)