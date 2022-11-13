project(cuda-fixnum CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_library(fixnum_library STATIC 
    ./fixnum.cu
    ./primitives.cu
)