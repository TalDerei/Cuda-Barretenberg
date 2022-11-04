# Implementation Plan
- [x] Set up a cloud infrastructure (NVIDIA A10 (Ampere) w/ 24 GB VRAM) on Oracle Cloud
- [x] Initialize Barretenberg repo
- [x] Integrate cuda-fixnum external library and add cuda / nvcc support
- [ ] Implement FF operations on the GPU
    - [ ] Understand the logic for the BN-254 base field (fq.test.cpp, fq.hpp, field.hpp, and field_impl.hpp)
    - [ ] Implement equivalent logic for GPU using cuda-fixnum