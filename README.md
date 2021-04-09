## SPARC installation and usage 

### (1) Brief:
SPARC is an open-source software package for the accurate, effcient, and scalable solution of the Kohn-Sham density functional theory (DFT) problem. The main features of SPARC currently include

* Isolated systems such as molecules and clusters as well as extended systems such as crystals, surfaces, and wires.
* Calculation of ground state energy, atomic forces, and stress tensor.
* Unconstrained collinear magnetization via spin polarized calculations.
* Structural relaxation and quantum molecular dynamics (QMD). 
* Local and semilocal exchange correlation functionals.
* ONCV and TM pseudopotentials in psp8 (ABINIT) format. 

SPARC is straightforward to install/use and highly competitive with state-of-the-art planewave codes, demonstrating comparable performance on a small number of processors and order-of-magnitude advantages as the number of processors increases. Notably, the current version of SPARC brings solution times down to a few seconds for systems with O(100-500) atoms on large-scale parallel computers, outperforming planewave counterparts by an order of magnitude and more. Additional details regarding the formulation and implementation of SPARC can be found in the paper referenced below. Future versions will target similar solution times for large-scale systems containing many thousands of atoms, and the efficient solution of systems containing a hundred thousand atoms and more.

**Users of SPARC are expected to cite the following publication: Q. Xu, A. Sharma, B. Comer, H. Huang, E. Chow, A.J. Medford, J.E. Pask, and P. Suryanarayana, 2020. SPARC: Simulation Package for Ab-initio Real-space Calculations. arXiv preprint arXiv:2005.10431.**

### (2) Installation:

Prerequisite: C compiler, MPI.

There are several options to compile SPARC, depending on the available external libraries.

* Option 1 (default): Compile with BLAS and LAPACK.

  * Step 1: Install/Load OpenBLAS/BLAS and LAPACK.

  * Step 2: Change directory to `src/` directory, there is an available `makefile`.

  * Step 3 (optional): Edit `makefile`. If the BLAS library path and LAPACK library path are not in the search path, edit the `BLASROOT` and `LAPACKROOT` variables, and add them to `LDFLAGS`. If you are using BLAS instead of OpenBLAS, replace all `-lopenblas` flags with `-lblas`.

  * Step 4 (optional): To turn on `DEBUG` mode, set `DEBUG_MODE` to 1 in the `makefile`.

  * Step 5: Within the `src/` directory, compile the code by
    ```shell     
    $ make clean; make
    ```
**Remark**: make sure in the makefile `USE_MKL = 0`, `USE_SCALAPACK = 0`, and `USE_DP_SUBEIG = 1` for option 1.

* Option 2 (recommended): Compile with MKL.
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

### (3) Input files:
The required input files to run a simulation with SPARC are (with shared names)  

(a) ".inpt" -- User options and parameters.  

(b) ".ion"  -- Atomic information.  

It is required that the ".inpt" and ".ion" files are located in the same directory and share the same name. A detailed description of the input options is provided in the documentation located in doc/. Examples of input files can be found in the `SPARC/tests` directory .

In addition, SPARC requires pseudopotential files of psp8 format which can be generated by D. R. Hamann's open-source pseudopotential code [ONCVPSP](http://www.mat-simresearch.com/). A large number of accurate and efficient pseudopotentials are already provided within the package. For access to more pseudopotentials, the user is referred to the [SG15 ONCV potentials](http://www.quantum-simulation.org/potentials/sg15_oncv/). Using the [ONCVPSP](http://www.mat-simresearch.com/) input files included in the [SG15 ONCV potentials](http://www.quantum-simulation.org/potentials/sg15_oncv/), one can easily convert the [SG15 ONCV potentials](http://www.quantum-simulation.org/potentials/sg15_oncv/) from upf format to psp8 format. Paths to the pseudopotential files are specified in the ".ion" file.

### (4) Execution:
SPARC can be executed in parallel using the `mpirun` command. Sample PBS script files are available in "SPARC/tests" folder. It is required that the ".inpt" and ".ion" files are located in the same directory and share the same name. For example, to run a simulation with 8 processes with input files as "filename.inpt" and "filename.ion" in the root directory (`SPARC/`), use the following command:

```shell
$ mpirun -np 8 ./lib/sparc -name filename
```

As an example, one can run a test located in `SPARC/tests`. First go to `SPARC/tests/MeshConvergence/Si8` directory:

```shell
$ cd tests/MeshConvergence/Si8
```

There are a few input files available. Run a DC silicon system with mesh = $0.4$ Bohr by

```shell
$ mpirun -np 8 ../../../lib/sparc -name Si8-ONCV-0.4
```

The result is printed to output file "Si8-ONCV-0.4.out", located in the same directory as the input files. If the file "Si8-ONCV-0.4.out" is already present, the result will be printed to "Si8-ONCV-0.4.out\_1" instead. The max number of ".out" files allowed with the same name is 100. Once this number is reached, the result will instead overwrite the "Si8-ONCV-0.4.out" file. One can compare the result with the reference out file named "Si8-ONCV-0.4.refout".

In the `tests/quick/` directory, we also provide a sample script file `quickrun.sh`, which launches four quick tests one by one. To run these quick tests, simply change directory to `tests/quick/` directory, and run: 

```shell
$ chmod +x ./quickrun.sh
$ ./quickrun.sh
```

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

**Quantum molecular dynamics (QMD) calculations**

- `.out` file  

  See above.

- `.aimd` file  

  Atomic positions, atomic velocities, atomic forces, electronic temperature, ionic temperature and total energy for each QMD step.

- `.restart` file  

  Information necessary to perform a restarted QMD calculation. 
