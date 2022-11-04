```Problems encountered along completing tasks```

Starting to dive into the unit testing (each module has its own set of tests). Need to figure out what the hexadecimal represents for bn254/fq.hpp for example, and how simple calculations (i.e. addition/subtraction/multiplication/division) translate between hex numbers?
    - Figured out the modulus is deconstructed into smaller parts, i.e. a 256-bit number is represented by 4 64-bit limbs.

Currently the recompilation process for making any change is taking a long time. Need to figure out how to speed this up. Additionally, need to figure out how to print print statements in constant expressions?
    - Fastest solution I found was to just comment out the other files containing the other test cases that aren't currenlty being benchmarked in order to speed up the compilation process. The issue is that printf format strings are interpreted at runtime rather than compiler time. Looking at external library called "Compile-Time Printer" for priting types and values at compile-time in C++ -- seems like a promising solution. Debugging compile-time statements seems harder than it should be....decided to get around this by using the GDB debugger instead of print statements in trying to debug what's going on. 

Need to differentiate between Fq and Fr, and also need to figure out how to convert the hexadecimal representation of the fields to decimal to verify they match up?
    - Fq represents the base field for BN-254, while Fr represents the scalar field. In order to get the hexidecimal representation of the number, convert the decimal representation to hexidecimal, split it up into 4 equal limbs, and take the reverse order. 

