project(cuda-barretenberg CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Bypass JIT compilation of PTX code to SASS instructions
find_package(CUDA REQUIRED)
set(CUDA_ARCH_LIST Auto CACHE LIST
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
     compute capability versions (6.1, 7.0, etc) to generate code for. \
     Set to Auto for automatic detection (default)."
)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
string(REPLACE ";" " " CUDA_ARCH_FLAGS_SPACES "${CUDA_ARCH_FLAGS}")

string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_SPACES}")
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wall,-Wextra")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

add_subdirectory(arithmetic)
add_subdirectory(fields)
add_subdirectory(groups)
add_subdirectory(benchmark)