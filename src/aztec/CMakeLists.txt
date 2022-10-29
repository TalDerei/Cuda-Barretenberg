enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-Wall -Wextra -Wconversion -Wsign-conversion -Wno-deprecated -Wno-tautological-compare -Wfatal-errors)

add_compile_options($<$<COMPILE_LANGUAGE:CXX,Clang>:-fconstexpr-steps=100000000>)

# set(CMAKE_CXX_STANDARD 14)
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

# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     # add_compile_options(-Wno-unguarded-availability-new -Wno-c99-extensions -fconstexpr-steps=100000000)
#     add_compile_options($<$<COMPILE_LANGUAGE:CXX,Clang>:-fconstexpr-steps=100000000>)
#     if(MEMORY_CHECKS)
#         message(STATUS "Compiling with memory checks.")
#         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#     endif()
# endif()

# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#     add_compile_options(-Wno-deprecated-copy -fconstexpr-ops-limit=100000000)
# endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# I feel this should be limited to ecc, however it's currently used in headers that go across libraries,
# and there currently isn't an easy way to inherit the DDISABLE_SHENANIGANS parameter.
if(DISABLE_ASM)
    message(STATUS "Using fallback non-assembly methods for field multiplications.")
    add_definitions(-DDISABLE_SHENANIGANS=1)
else()
    message(STATUS "Using optimized assembly for field arithmetic.")
endif()

add_subdirectory(env)
add_subdirectory(numeric)
add_subdirectory(srs)
add_subdirectory(ecc)
add_subdirectory(crypto)
add_subdirectory(polynomials)
add_subdirectory(plonk)
add_subdirectory(stdlib)
add_subdirectory(rollup)
add_subdirectory(cuda-fixnum)

if(BENCHMARKS)
    add_subdirectory(benchmark)
endif()
