# External Vectorization and Acceleration (EVA) Module for SPARC

Optimized, OpenMP parallelized CPU implementation and CUDA implementation for major computation kernels in SPARC.



## Compiling and Running SPARC with EVA Module

Compiling: 
1. Modify the **Makefile** in this directory according to your system environments
    * Usually you ned to check ``LDFLAGS`, `CUPATH` and `NVFLAGS`.
    * If you want to compile EVA module without using CUDA kernels, set the `USE_CUDA` to 0. 
    * If you want to compile EVA module without using MKL specific functions, set the `USE_MKL` to 0.
2. Use `make` to compile SPARC with EVA module. All **.o** files (from SPARC source codes and EVA module source codes) will be stored in this directory, and the executable file is at **../../lib/sparc-EVA**

Running:
* Set environment variable `EXT_VEC_ACCEL=1` to activate EVA module functions. If `EXT_VEC_ACCEL=0`, EVA module will only collect and report timing results from the original implementations in SPARC.
* Set environment variable `OMP_NUM_THREADS` to control the number of threads used by each MPI process in EVA module kernel functions. You can also set other OpenMP environment variables. 
* Set environment variable `EXT_VEC_ACCEL_CUDA=1` to activate EVA module CUDA kernels. Setting  `EXT_VEC_ACCRL=0`  will override  `EXT_VEC_ACCEL_CUDA=1` .

**IMPORTANT NOTICE**:

* Currently EVA module only supports orthogonal cell computation.
* Currently you need to run 1 MPI process per computing node to use the CUDA kernels in the EVA module. 
* If the number of MPI process >= the `NSTATE` value in the input file and use more than 1 OpenMP threads per MPI process, using the EVA module may SLOW DOWN the calculation.
* If the number of MPI process > the `NSTATE` value in the input file, EVA module will NOT activate CUDA kernels.



## Some Technical Details for Developers and Advanced Users

Functionality of EVA module:
* Provides timing results of the computational kernels in the original SPARC implementation and the EVA module;
* Accelerates the Chebyshev Filtering process (the major computation kernel) with OpenMP or CUDA kernels;
* Accelerates the AAR procedure with optimized and OpenMP supported Laplacian  matrix-vector multiplication operator.

SPARC functions that calls EVA module functions:
* **electronicGroundState.c**: `scf()`: calls `EVA_buff_init()` and `EVA_buff_finalize()`
* **eigenSolver.c**: 
    * `CheFSI()`: calls `EVA_Chebyshev_Filtering()`
    * `Project_Hamiltonian()`: calls `EVA_Hamil_MatVec()`