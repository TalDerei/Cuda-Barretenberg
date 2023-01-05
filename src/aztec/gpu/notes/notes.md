# Notes 
```1. Integrate cuda-fixnum external library and add cuda / nvcc support```

    The core problem was that I was unable to call Barretenberg dependencies (e.g. #include <ecc/curves/grumpkin/grumpkin.hpp>) from a cuda file (.cu extension). I set the default system compiler as clang/clang++ using the x86_64-linux-clang toolchain. Nvidia's NVCC compiler was forwarding certain flags to the host compiler using the "-XCompiler" command, since certain barretenberg dependencies relied on that flag being set and compiled by the host. Still the compilation errors persisted. Another approach was creating a static library containing these dependencies, and dynamically linking the library to the cuda file upon creating the executable. But from the cuda file, you can only call certain functions from that library (not dependencies) since a static library is just a collection of linkable object .o files. 

    Then I thought it was a cuda-version compatability issue, where the cuda version might not support the compiler setup: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements. Turns out there wasn't any nvcc compatability issues with any of the installed tooling.
    
        ubuntu: Ubuntu 22.04 LTS
        clang++: 14.0.0-1ubuntu1
        nvcc: 11.8
        
        gcc: can't use since flag that a dependency relies 
        on (e.g. -fconstexpr-steps=100000000) isn't           
        recognized by gcc compiler

    Finally, I looked deeper into constexpr (feature added in c++11) which adds performance improvements by evaluating computations at compile-time rather than run-time, which speeds up the execution. Digging deeper, I found that NVCC compiler does not currently support C++20. Barretenberg has files with C20++ features (i.e. is_constant_evaluated()), so the compilation fails. We can't offload this to be compiled by the host compiler either. Only solution is to remove this, and add -std=c++17 as a compilation flag. 

    It's also worth noting that NVCC will always require a general purpose C++ host compiler. You can set CMAKE_CXX_STANDARD and CMAKE_CUDA_STANDARD values to c++17 so C++-files and CUDA-files can both be compiled according to c++17.

```2. Finite Field Arithmetic and Elliptic Curve Operations (BN-254 and Grumpkin) on GPU```

    ## Montgomery Representation
    Montgomery representation is alternative way of representing elements of Fr/Fq for more efficient multiplication. Let r be BN-254.r or q be Grumpkin.q and R = 2^256. The Montgomery representation of the nubmer x (e.g. 5) is (xR) mod p. This number then is represented as a little-endian length 4 array of 64-bit integers, where each element in the array is called a limb. 

    Usually we're used to working 32-bit/64-bit integers. With SNARK provers, the integers are much larger. The integers are 256 bits and represented using arrays of integers. For example, we could represent them using an array of 4 64-bit integers (since 4 * 64 = 256 > 254). 

    Q. Difference between BN-254 and Grumpkin curves?
        BN-254 is pairing-friendly elliptic curve. Grumpkin is a curve on top of BN-254 for SNARKL efficient group operations. It forms a curve cycle with BN-254, so the field and group order of Grumpkin are equal group and field order of BN-254. 