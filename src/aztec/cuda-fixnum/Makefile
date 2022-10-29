# CXX ?= clang-13
# -ccbin $(CXX) 
GENCODES ?= 86

INCLUDE_DIRS = -I/home/ubuntu/barretenberg/src/aztec
NVCC_FLAGS = -std=c++17 -Xcompiler -Wall,-Wextra,-fconstexpr-steps=100000000
NVCC_OPT_FLAGS = -DNDEBUG
NVCC_TEST_FLAGS = -lineinfo
NVCC_DBG_FLAGS = -g -G
NVCC_LIBS = -lstdc++ 

all:
	@echo "Please run 'make check' or 'make execute'."

execute: main.cu 
	nvcc $(NVCC_OPT_FLAGS) $(NVCC_FLAGS) $(GENCODES:%=--gpu-architecture=compute_%) $(GENCODES:%=--gpu-code=sm_%) $(INCLUDE_DIRS) $(NVCC_LIBS) -o $@ $<

# execute: execute

.PHONY: clean
clean:
	$(RM) execute
