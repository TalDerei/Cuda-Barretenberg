# Implementation Plan
- [x] Set up a cloud infrastructure (NVIDIA A10 (Ampere) w/ 24 GB VRAM) on Oracle Cloud
- [x] Initialize Barretenberg repo
- [x] Integrate cuda-fixnum external library and add cuda / nvcc support
- [x] Implement FF logic on the GPU
    - [x] Understand the algorithm, and the difference between montogomery multiplication and montogomery reduction schemes
    - [x] Extract the montogomery multiplication (CIOS) implementation to a seperate benchmarking file and get it compiling
    - [x] Implement addition/subtraction operations on GPU
    - [x] Understand Barretenberg's C++ implementation, and benchmark for correctness and performance
    - [x] Implement unit tests logic for fq / fr (base and scalar fields)
- [ ] Implement BN254 / Grumpkin ECs logic on the GPU   
    - [ ] Understand the differences between BN-254 and Grumpkin ECs
    - [ ] Implement unit tests logic for BN-254 / Grumpkin EC
- [ ] Benchmark FF and ECC implementations for CPU / GPU
    - Benchmark decrypt_bench folder for initial baselines
    - Reimplement logic for decrypt_bench for GPU and check for correctness
    - Benchmark GPU implementation
- [x] Set up Nvsight Compute profiling software
- [x] Downgrade GPU from A10 to V100 to save on cloud computing costs, and create custom images to clone machines