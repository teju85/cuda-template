# Introduction
This is a template cuda project to be used for quickly creating from-scratch
CUDA PoC's.

# Requirements
* full CUDA SDK installation
* CUDA-enabled GPU(s)

# Setup
```bash
git clone https://github.com/teju85/cuda-template
cd cuda-template
make test  # for running unit-tests
make -j    # for building the final exe (or library)
```
Refer to the `Makefile` and files under `src/` and `tests/` folder and feel free
to edit/add source files as per your needs.
