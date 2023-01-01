project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(../fields)
include_directories(../arithmetic)

add_library(group_library STATIC 
    ./group.cu
    ./element.cu
)

target_link_libraries(
    group_library
    fixnum_library
)