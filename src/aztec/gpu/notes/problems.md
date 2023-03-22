# Problems
```Need to integrate cuda-fixnum external library and add cuda / nvcc support```

    - The core problem was that I was unable to call Barretenberg dependencies (e.g. #include <ecc/curves/grumpkin/grumpkin.hpp>) from a cuda file (.cu extension). I set the default system compiler as clang/clang++ using the x86_64-linux-clang toolchain. Nvidia's NVCC compiler was forwarding certain flags to the host compiler using the "-XCompiler" command, since certain barretenberg dependencies relied on that flag being set and compiled by the host. Still the compilation errors persisted. Another approach was creating a static library containing these dependencies, and dynamically linking the library to the cuda file upon creating the executable. But from the cuda file, you can only call certain functions from that library (not dependencies) since a static library is just a collection of linkable object .o files. 

    Then I thought it was a cuda-version compatability issue, where the cuda version might not support the compiler setup: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements. Turns out there wasn't any nvcc compatability issues with any of the installed tooling.
    
        ubuntu: Ubuntu 22.04 LTS
        clang++: 14.0.0-1ubuntu1
        nvcc: 11.8
        
        gcc: can't use since flag that a dependency relies 
        on (e.g. -fconstexpr-steps=100000000) isn't           
        recognized by gcc compiler

    Finally, I looked deeper into constexpr (feature added in c++11) which adds performance improvements by evaluating computations at compile-time rather than run-time, which speeds up the execution. Digging deeper, I found that NVCC compiler does not currently support C++20. Barretenberg has files with C20++ features (i.e. is_constant_evaluated()), so the compilation fails. We can't offload this to be compiled by the host compiler either. Only solution is to remove this, and add -std=c++17 as a compilation flag. 

    It's also worth noting that NVCC will always require a general purpose C++ host compiler. You can set CMAKE_CXX_STANDARD and CMAKE_CUDA_STANDARD values to c++17 so C++-files and CUDA-files can both be compiled according to c++17.

```Structure of the finite field arithmetic and elliptic curve operations (BN-254 and Grumpkin) on GPU```
    - Montgomery representation is alternative way of representing elements of Fr/Fq for more efficient multiplication. Let r be BN-254.r or q be Grumpkin.q and R = 2^256. The Montgomery representation of the nubmer x (e.g. 5) is (xR) mod p. This number then is represented as a little-endian length 4 array of 64-bit integers, where each element in the array is called a limb. 

    Usually we're used to working 32-bit/64-bit integers. With SNARK provers, the integers are much larger. The integers are 256 bits and represented using arrays of integers. For example, we could represent them using an array of 4 64-bit integers (since 4 * 64 = 256 > 254). 

    Q. Difference between BN-254 and Grumpkin curves?
        BN-254 is pairing-friendly elliptic curve. Grumpkin is a curve on top of BN-254 for SNARK efficient group operations. It forms a curve cycle with BN-254, so the field and group order of Grumpkin are equal group and field order of BN-254. 

```Starting to dive into the unit testing (each module has its own set of tests). Need to figure out what the hexadecimal represents for bn254/fq.hpp for example, and how simple calculations (i.e. addition/subtraction/multiplication/division) translate between hex numbers?```

    - Figured out the modulus is deconstructed into smaller parts, i.e. a 256-bit number is represented by 4 64-bit limbs.

```Currently the recompilation process for making any change is taking a long time. Need to figure out how to speed this up. Additionally, need to figure out how to print print statements in constant expressions?```

    - Fastest solution I found was to just comment out the other files containing the other test cases that aren't currenlty being benchmarked in order to speed up the compilation process. The issue is that printf format strings are interpreted at runtime rather than compiler time. Looking at external library called "Compile-Time Printer" for priting types and values at compile-time in C++ -- seems like a promising solution. Debugging compile-time statements seems harder than it should be....decided to get around this by using the GDB debugger instead of print statements in trying to debug what's going on. 

```Need to differentiate between Fq and Fr, and also need to figure out how to convert the hexadecimal representation of the fields to decimal to verify they match up?```

    - Fq represents the base field for BN-254, while Fr represents the scalar field. In order to get the hexidecimal representation of the number, convert the decimal representation to hexidecimal, split it up into 4 equal limbs, and take the reverse order. 

```Need to figure out an effective method for testing cuda functions correctness. There doesn't seem to be a framework for cuda testing.```

    - Depaul recommends simply writing reference kernels to test. In order to debug, need to also disable compiler optimizations -O3 and compiler with -O0 since the compiler is optimizing out a lot of the intermediary steps and not letting me see what's going on under the hood. Another option is to extract and isolate the workload into a seperate c++ file, test it out locally with print statements there since I can't add print statements to constant expressions. 

```Need to figure out how to integrate cuda-fixnum logic to use their montgomery multiplication implementation. ```

    - The problem is the field elements are being intialized on the device, and the cuda-fixnum functions make calls to host functions before calling a dispatch() function, which is a kernel. Tried doing a sample montgomery multiplication by modifying the cuda-fixnum base types from uint8_t * to uint64_t *. Seems to be an incorrect result. Need to verify if: [1] initializing the values is not done correctly, or [2] moving from uint8_t * to uint64_t * breaks the cuda-fixnum code, or [3] the issue is with how the numbers are being printed and need to convert it to/from montgomery representation, or [4] how i'm retrieveing the values from calling get_fixnum_array() or print()_fixnum_array. Update: I ended up scrapping using the ENTIRE cuda-fixnum library, and used the minimal implementation used in the groth16 codebase. The modular multiplication (CIOS) works now

```Need to figure out how the reduce() algorithm fits into the from_montgomery_form() and to_montgomery_form() algorithms, and if we get the same results by using the from_monty() and to_monty() functions in cud-fixnum. Also need to figure out how we're able to call montgomery multiplication algorithms (both on barretenberg and cuda-fixnum without expolierlty converting to montgomery form)?```

    - What migth be happening, from initial intution, is that EC calls are loading in the points and converting them to montgomery form, and then the FF arithemtic is operating on those representations. What doesn't make sense is why we can call montgomery multiplication before transforming them field elements into montgomery representation. One possibility is [1] the algorithm does the conversion for you and finds the Montgomery representation of the product, but doesn't do the conversion back, or [2] the algorithm doesn't do the conversion for you and finds the Montgomery representation of the product, but doesn't do the conversion back. In both cases, the product is still in montgomery representation. Update: ended up using cuda-fixnum for converting to and from montgomery representation. Still need to look deeper into how these reductions works.

```Need to understand the BN-254 G1 parameters?```

    - These are curve parameters a and b (y^2 = x^3 + ax + b), cofactor, and group generators
    
```Why is the first run of cuda functions much slower than the subsequent runs? ```

    - There might be some sort of caching effects happening, and need to verify these results with compiler optimizations -03 turned on, which are currently disabled for gdb testing purposes.  

```I have a struct T = {x, y, z} and Z = {x, y}, and each of the inner variables represent a length-4 array of 64-bit integers. I’m calling a device function mixed_addition (from a kernel) to add structs T + Z. One method is since there are 5 arrays w/ 4 elements each, I can execute the kernel on 4 threads, and each thread focuses on different array indices. Here’s my question. How can I perform the same parallelization by calling mixed_addition by passing in the top-level structs T and Z (instead of the individual arrays themselves)? And how does the kernel know how to operate more complex data structures / structs? ```

    - For arrays, you need to calculate the index each thread will be operating on with the following formula: tid = (blockDim.x * blockIdx.x) + threadIdx.x; For more complex data structures, currently looking into cuda's memory grid heirachy and streams. For testing purposes, I'll treat everything like a var and see if that works, but in production, will need to figure out how to parallelize on the element / affine element level. Also look into the concept of "grid-stride loops". 

```I'm having math correctness errors when trying to do group arithemtic for some reason? The math isn't consistent, as sometimes the values are what you expect, other times it isn't...```

    - The issue is apparent when trying to square a field element, and adding the original element to itself. For example, x = x^2 + x in my implementation with cuda-fixnum isn't yielding the same results as barretenberg for some reason, very strange! Further digging seems to suggest that low-level math primitives aren't the problem, but instead the extensive use of operator overloading that's sometimes a bit confusing. Update: The solution was sub_coarse() with double the field size as the modulus for some reason, but not sure when to use it? Seems like it's only needed for some group operations at the moment. 

```Getting correctness errors for some of the G1 unit tests when trying to do elliptic curve operations...still debugging the exact issues.```

    - It's still giving issues when trying to add two of the same numbers on the ECC add() function, still figuring out why? Barretenberg uses twice_modulus (twice_modulus = modulus + modulus), twice_not_modulus (-twice_modulus), and not_modulus (-modulus) constructs for some reason in the arithmetic. Additionally, the subtraction and addition logic operates on twice the modulus. There's a "modulus.data[3] >= 0x4000000000000000ULL" check at the beggining of many of the functions's but it's never executed. Also not sure why the threshold is set to 0x4000000000000000ULL, maybe this is an optimized value. Update: It's an issue between reading results to and from montgomery form, so to ensure correctness always remember to convert results back from mongomery form. Need to still investigate why I can't add the same number to itself in ECC::add() function.

```Execution times need to be looked further into + optimized down the line, as I'm only focusing on correctness right now.```

    - Firstly, need to research the limitations around running more threads. Currently running 4 threads, and the execution is running sequentially in a single SM. Also unsure why the execution times are so drastic, as there seems to be some caching built in that's speeding up subsequent runs of the same computation. Also remember to enable compiler optimizations -03 during benchmarking! Update: “just-in-time” compiling PTX code to SASS assembly instructions slows down the first kernel invocation, so need to pass in "-gencode arch=compute_86,code=sm_86" to get around that. 

```Need to figure out why the pippenger and polynomial bench isn't working?```

    - Ignition trusted setup ceremony says 100.8M (~2^26-2^27). It’s split up across 20 files, and ./bootstrap.sh was only configured to download the first 3 transcripts. Running “./download_ignition.sh” downloads the rest of the transcripts. Still, the pippenger bench is segfaulting (core dumped) for some reason. Update: I fixed the seg faults by pointing to the correct srs_db directory and commenting out some benchmark tests. Need to also figure out the difference in execution times between the pippenger and polynomial benches for Multi-scalar multiplication...update: seems like std::chrono time is generally slower in benchmarking. 

```Notes while performing simple double and add MSM```

    Questions:
    1. How to expand this to perform on an actual Fq point and Fr scalar?
        --> They operate over different prime fields, but Fr scalar doesn't participate 
        in the addition / multiplication calculations, only multiples of the Fq curve element. 
    2. How to expand to add a vector of Fq points?
    2. How to expand to add a vector of G1 curve points?
    3. How to expand this to perform on a vector of points and scalars?

    Notes:
    - Seems like barretenberg doesn't have methods to multipliy fq * g1 or g1 * g1, or fq * g1.x or g1.x * g1.x
    - Performing a double and then add, e.g. ec + ec = 2ec, then 2ec + ec = 3ec yields the same results in gpu tests
    and barretenberg. But ec + ec = 2ec, then 2ec + ec = 3ec, then 3ec + ec = 4ec isn't yielding the same results
    as 2ec + 2ec with doubling. Need to investigate, because seems like a montgomery representation problem. 

    Even on Barretenberg,the results might be different. It's converting everything to montgomery form before starting the calculation,
    and the assert check still passes. Some conversions going on in the equality '==' check. 

    To standardarize the results between barretenberg and my test suites, we'll do the following:
    1. For FF code, don't need to convert to and from montgomery representation unless it's a multiplication operation.
    2. For ECC code, always convert to and from montgomery representation code. 

    We'll follow the same spec as Barretenberg, without all of the extra confusing operator overloading ops. Need to perform both calculations in both ways to and compare the difference.

    Most important thing I've learned building these MSM tests is that a single thread cannot perform multiple calculations inside the kernel invocation, instead things like additions operate on thread blocks of width 4, otherwise the result is corrupted. 

```Where are extension fields used in the calculation?```
    Different extensions of the same base field would be used for different polynomials your committing to (e.g. permutation or quotient polynomial for example). The pairing groups, both G2 and GT, are specifically defined over extension fields. 

```Is MSM performed more commonly on affine, jacobian, or projective elements?```
    Affine elements are transformed to Jacobian or Projective elements for more efficient modular additions. 
