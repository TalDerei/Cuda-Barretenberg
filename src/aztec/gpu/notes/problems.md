# Problems
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

```Need to figure out why the pippenger and polynomial bench isn't working```

    - Ignition trusted setup ceremony says 100.8M (~2^26-2^27). It’s split up across 20 files, and ./bootstrap.sh was only configured to download the first 3 transcripts. Running “./download_ignition.sh” downloads the rest of the transcripts. Still, the pippenger bench is segfaulting (core dumped) for some reason. Update: I fixed the seg faults by pointing to the correct srs_db directory and commenting out some benchmark tests. Need to also figure out the difference in execution times between the pippenger and polynomial benches for Multi-scalar multiplication...seems like std::chrono time is generally slower in benchmarking. 

