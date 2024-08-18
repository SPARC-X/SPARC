## SPARC installation and usage 

### (1) Brief:
SPARC is an open-source software package for the accurate, effcient, and scalable solution of the Kohn-Sham density functional theory (DFT) problem. The main features of SPARC currently include

* Applicable to isolated systems such as molecules as well as extended systems such as crystals, surfaces, and wires.
* Local, semilocal, and nonlocal (including hybrid) exchange-correlation functionals.
* Standard ONCV pseudopotentials, including nonlinear core corrections (NLCCs).
* Calculation of ground state energy, atomic forces, and stress tensor.
* Structural relaxation and ab initio molecular dynamics (NVE, NVT, and NPT).
* Spin polarized and unpolarized calculations.
* Spin-orbit coupling (SOC).
* Noncollinear spin.
* Dispersion interactions through DFT-D3, vdW-DF1, and vdW-DF2.
* Symmetry-adaption for cyclic and/or helical symmetries (Cyclix-DFT).
* O(N) Spectral Quadrature (SQ) method.
* On-the-fly machine-learned force field (MLFF) molecular dynamics (MD) simulations. 

SPARC is straightforward to install, use, and modify, with minimal external library dependencies. It has shown to be an order of magnitude faster than state-of-the-art planewave codes, with a range of exchange-correlation functionals, and with increasing advantages as the number of processors is increased. In particular, SPARC efficiently scales to thousands of processors in regular operation, bringing solution times down to about a minute for systems with O(500-1000) atoms, and a few seconds for O(100-500) atoms. Using the O(N) SQ method, SPARC has been scaled to system sizes of over a million atoms (https://doi.org/10.1088/1361-651X/acdf06). 

### (2) Installation:

Prerequisite: C compiler, MPI.

There are several options to compile SPARC, depending on the available external libraries.

* Option 1: Compile with BLAS and LAPACK.

  * Step 1: Install/Load OpenBLAS/BLAS and LAPACK.

  * Step 2: Change directory to `src/` directory, there is an available `makefile`.

  * Step 3 (optional): Edit `makefile`. If the BLAS library path and LAPACK library path are not in the search path, edit the `BLASROOT` and `LAPACKROOT` variables, and add them to `LDFLAGS`. If you are using BLAS instead of OpenBLAS, replace all `-lopenblas` flags with `-lblas`.

  * Step 4 (optional): To turn on `DEBUG` mode, set `DEBUG_MODE` to 1 in the `makefile`.

  * Step 5: Within the `src/` directory, compile the code by
    ```shell     
    $ make clean; make
    ```
**Remark**: make sure in the makefile `USE_MKL = 0`, `USE_SCALAPACK = 0`, and `USE_DP_SUBEIG = 1` for option 1.

* Option 2 (default): Compile with MKL.
  * Step 1: Install/Load MKL.

  * Step 2: Change directory to `src/`  directory, there is an available `makefile`.

  * Step 3: Edit `makefile` . Set `USE_MKL` to 1 to enable compilation with MKL. If the MKL library path is not in the search path, edit the `MKLROOT` variable to manually set the MKL path.

  * Step 4 (optional): For the projection/subspace rotation step, to use SPARC routines for matrix data distribution rather than ScaLAPACK (through MKL), set `USE_DP_SUBEIG`  to 1. We found on some machines this option is faster.

  * Step 5 (optional): To turn on `DEBUG` mode, set `DEBUG_MODE` to 1 in the `makefile` .

  * Step 6: Within the `src/` directory, compile the code by

    ```shell
    $ make clean; make
    ```
**Remark**: make sure in the makefile `USE_MKL = 1` and `USE_SCALAPACK = 0` for option 2.

* Option 3: Compile with BLAS, LAPACK, and ScaLAPACK.

  * Step 1: Install/Load OpenBLAS/BLAS, LAPACK, and ScaLAPACK.

  * Step 2: Change directory to `src/`  directory, there is an available `makefile`.

  * Step 3 (optional): Edit `makefile`. If the BLAS library path, LAPACK library path, and/or ScaLAPACK library path are not in the search path, edit the `BLASROOT`, `LAPACKROOT` , and/or `SCALAPACKROOT` variables accordingly, and add them to `LDFLAGS`. If you are using BLAS instead of OpenBLAS, replace all `-lopenblas` flags with `-lblas`.

  * Step 4 (optional): For the projection/subspace rotation step, to use SPARC routines for matrix data distribution rather than ScaLAPACK, set `USE_DP_SUBEIG`  to 1. We found on some machines this option is faster.

  * Step 5 (optional): To turn on `DEBUG`  mode, set `DEBUG_MODE` to 1 in the `makefile`.

  * Step 6: Within the `src/` directory, compile the code by

    ```shell
    $ make clean; make
    ```
**Remark**: make sure in the makefile `USE_MKL = 0` and `USE_SCALAPACK = 1` for option 3.

Once compilation is done, a binary named `sparc` will be created in the `lib/` directory.

* Option 4: Install pre-compiled `sparc` binaries distributed by `conda-forge`

Pre-compiled `sparc` package can be installed on x86_64 or aarch64 Linux platforms using `anaconda` or `miniconda`. 
The binary is compiled with OpenBLAS and OpenMPI and flags `USE_MKL=0 USE_SCALAPACK=1 USE_FFTW=1`.

  * Step 1 (optional): create a conda environment (e.g. `sparc-env`)
    ```shell
    conda create -n sparc-env
    conda activate sparc-env
    ```
  * Step 2: install conda package `sparc-x`
    ```shell
    conda install -c conda-forge sparc-x
    echo sparc binary is located at: $(which sparc)
    echo .psp files installed at: $SPARC_PSP_PATH
    echo SPARC doc files installed at: $SPARC_DOC_PATH
    ```


### (3) Input files:
The required input files to run a simulation with SPARC are (with shared names)  

(a) ".inpt" -- User options and parameters.  

(b) ".ion"  -- Atomic information.  

It is required that the ".inpt" and ".ion" files are located in the same directory and share the same name. A detailed description of the input options is provided in the documentation located in doc/. Examples of input files can be found in the `SPARC/tests` directory .

In addition, SPARC requires pseudopotential files of psp8 format which can be generated by D. R. Hamann's open-source pseudopotential code [ONCVPSP](http://www.mat-simresearch.com/). The SPMS table of soft and transferable pseudopotentials is provided within the package. For access to more pseudopotentials, the user is referred to [pseudoDOJO ONCV potentials](http://www.pseudo-dojo.org) and the [SG15 ONCV potentials](http://www.quantum-simulation.org/potentials/sg15_oncv/). Note that using the [ONCVPSP](http://www.mat-simresearch.com/) input files included in the [SG15 ONCV potentials](http://www.quantum-simulation.org/potentials/sg15_oncv/), one can easily convert the [SG15 ONCV potentials](http://www.quantum-simulation.org/potentials/sg15_oncv/) from upf format to psp8 format. Paths to the pseudopotential files are specified in the ".ion" file.

### (4) Execution:
SPARC can be executed in parallel using the `mpirun` command. Sample PBS script files are available in "SPARC/tests" folder. It is required that the ".inpt" and ".ion" files are located in the same directory and share the same name. For example, to run a simulation with 8 processes with input files as "filename.inpt" and "filename.ion" in the root directory (`SPARC/`), use the following command:

```shell
$ mpirun -np 24 ./lib/sparc -name filename
```

As an example, one can run one of the tests located in `SPARC/tests/`. First go to `SPARC/tests/Example_tests/` directory:

```shell
$ cd tests/Example_tests/
```

The input file is available inside the folder. Run a DC silicon system by

```shell
$ mpirun -np 24 ../../lib/sparc -name Si8_kpt
```

The result is printed to output file "Si8_kpt.out", located in the same directory as the input files. If the file "Si8_kpt.out" is already present, the result will be printed to "Si8_kpt.out\_1" instead. The max number of ".out" files allowed with the same name is 100. Once this number is reached, the result will instead overwrite the "Si8_kpt.out" file. One can compare the result with the reference out file named "Si8_kpt.refout".


In the `tests/` directory, we also provide a suite of tests which are arranged in a hierarchy of folders. Each test system has its own directory. A python script is also provided which launches the suite of test systems. To run a set of four quick tests locally on the CPU, simply run: 

```shell
$ python SPARC_testing_script.py quick_run
```

The result is stored in the corresponding directory of the tests. A message is also printed in the terminal showing if the tests passed or failed. The tests can also be launched in parallel on a cluster by using the Python script. Detailed information on using the python script can be found in the 'ReadMe' file in the `tests/` directory.

### (5) Output

Upon successful execution of the `sparc` code, depending on the calculations performed, some output files will be created in the same location as the input files.

#### Single point calculations

- ".out" file

  General information about the test, including input parameters, SCF convergence progress, ground state properties and timing information.

- ".static" file 

  Atomic positions and atomic forces if the user chooses to print these information.

#### Structural relaxation calculations

- ".out" file 

  See above.

- ".geopt" file 

  Atomic positions and atomic forces for atomic relaxation, cell lengths and stress tensor for volume relaxation, and atomic positions, atomic forces, cell lengths , and stress tensor for full relaxation.

- ".restart" file 

  Information necessary to perform a restarted structural relaxation calculation. Only created if atomic relaxation is performed.

**Ab initio molecular dynamics (AIMD) calculations**

- `.out` file  

  See above.

- `.aimd` file  

  Atomic positions, atomic velocities, atomic forces, electronic temperature, ionic temperature and total energy for each QMD step.

- `.restart` file  

  Information necessary to perform a restarted QMD calculation. 


### (6) Citation:

If you publish work using/regarding SPARC, please cite some of the following articles, particularly those that are most relevant to your work:
* **General**: https://doi.org/10.1016/j.simpa.2024.100649 (v2), https://doi.org/10.1016/j.softx.2021.100709 (v1), https://doi.org/10.1016/j.cpc.2016.09.020 (initial developments, isolated systems), https://doi.org/10.1016/j.cpc.2017.02.019 (initial developments, extended systems)
* **Non-orthogonal systems**: https://doi.org/10.1016/j.cplett.2018.04.018
* **Linear solvers**: https://doi.org/10.1016/j.cpc.2018.07.007, https://doi.org/10.1016/j.jcp.2015.11.018
* **Stress tensor/pressure**: https://doi.org/10.1063/1.5057355
* **Atomic forces**: https://doi.org/10.1016/j.cpc.2016.09.020 (isolated systems), https://doi.org/10.1016/j.cpc.2017.02.019 (extended systems)
* **Mixing**: https://doi.org/10.1016/j.cplett.2016.01.033, https://doi.org/10.1016/j.cplett.2015.06.029 (restarting), https://doi.org/10.1016/j.cplett.2019.136983 (preconditioning)
* **SPMS pseudopotentials**: https://doi.org/10.1016/j.cpc.2022.108594
* **Cyclic and/or helical symmetry**: https://doi.org/10.1103/PhysRevB.103.035101, https://doi.org/10.1103/PhysRevB.100.125143 (initial developments, extended systems), https://doi.org/10.1016/j.jmps.2016.08.007 (initial developments, isolated systems)
* **O(N) Spectral Quadrature method**: https://doi.org/10.1016/j.cpc.2015.11.005, https://doi.org/10.1016/j.cpc.2015.11.005 (initial implementation), https://doi.org/10.1016/j.cplett.2013.08.035 (formulation), https://doi.org/10.1007/978-3-031-22340-2_12 (detailed mathematical formulation)
* **On-the-fly MLFF**: https://doi.org/10.1063/5.0180541, https://doi.org/10.1063/5.0204229 (SQ), https://doi.org/10.48550/arXiv.2408.07554 (Cyclix), https://doi.org/10.48550/arXiv.2407.15290 (internal energy) 


### (7) Acknowledgement:
  
* **U.S. Department of Energy (DOE), Office of Science (SC): DE-SC0023445, DE-SC0019410**
* **U.S. Department of Energy (DOE), National Nuclear Security Administration (NNSA): Advanced Simulation and Computing (ASC) Program**
* **U.S. Department of Energy (DOE), National Nuclear Security Administration (NNSA): DE-NA0004128 (highT feature)**
* **U.S. National Science Foundation (NSF): 1553212 (cyclix feature)**
* **U.S. National Science Foundation (NSF): 1663244, and 1333500 (preliminary developments)**
