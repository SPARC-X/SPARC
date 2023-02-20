## SPARC testing suite 

### (1) Brief:
The test suite consists of a number of systems that are chosen to check different functionalities/features of SPARC. Each test system has its own directory inside of which the input files and the reference output files are stored. For each test system, two sets of reference output files are stored in folders named 'Standard' and 'High_accuracy' respectively: 

 * Standard case with an accuracy of around 1.0E-03 Ha/atom or better in energy. 
 * High accuracy case with an accuracy of around 1.0E-04 Ha/atom or better in energy.

The accuracy is with respect to well established plane wave codes ABINIT/Quantum Espresso.  A python script named `SPARC_testing_script.py` is provided which can launch the test systems and compare the results against the stored reference output files.

### (2) Running the script: 
Prerequisite: python3

The SPARC executable named `sparc` need to be placed in the `lib` folder or in the `tests` folder. Then, the tests can be run on a cluster by using the following command:
```shell
$ python SPARC_testing_script.py
```
Each system in the testing suite is given a set of tags based on the funtionality/feature that the given test system is testing. All systems with a set of common tags can be executed by using the following command:
```shell
$ python SPARC_testing_script.py -tags <tag1> <tag2> <tag3> ...
```
Individual or a group of systems can be executed by using the following command:
```shell
$ python SPARC_testing_script.py -systems <system1> <system2> <system3> ...
```

Once the script is executed, a progress bar of completion is also shown in the terminal and after all the test systems have been run, the result of passed and failed tests is printed in the terminal and a file named 'Report.txt' is generated which contains details of the difference in energy, force, stress etc. from the reference files. 

Output files generated during the execution of the tests can be deleted by the following commnad:
```shell
$ python SPARC_testing_script.py clean_temp
```

Comparison of the output files with the reference files once the tests have finished running can be done with the following command:

```shell
$ python SPARC_testing_script.py only_compare
```

The tag `only_compare` can be added along with other option to make the comparison for only the test systems corresponding to those tags.

### (3) Tags:

The systems in the testing suites are classified with a set of tags which describe the features which are being tested. The list of tags (highlighted) are given below:

 * Boundary conditions: `bulk`, `surface`, `wire`, `molecule`.
 * Cell type: `orth`, `nonorth`.
 * Exchange correlation: `lda`, `gga`,`scan`,`pbe0`,`hse`,`soc`,`vdWDF`,`d3`.
 * SCF Mixing and preconditioner: `potmix`,`denmix`,`kerker`.
 * Calculation type: `scf`,`relax_atom`,`relax_cell`,`relax_full`,`md`.
 * Relaxation type: `nlcg`,`lbfgs`,`fire`.
 * MD type: `nvtnh`,`nvkg`,`nve`,`npt`.
 * K-point sampling: `gamma`,`kpt`.
 * Spin polarization: `spin`,`SOC`.
 * Methods: `highT`,`SQ3`.
 * Smearing: `smear_fd`,`smear_gauss`.
 * Bandgap: `bandgap`.
 * Others: `nlcc`,`memcheck`,`fast`.

In addtion to the tags listed above, there are some tags which can be used to run every test with extra features. These tags are described below:

 * `VHQ`: run with high accuracy.
 * `serial`: run in serial.
 * `valgrind_all`: run with valgrind.
 * `update_reference`: update the reference files. 

### (4) Add new test system:

A new test system can be added to the test suite. The input and reference output files need to be generated and the python script `SPARC_testing_script.py` needs to be updated by following the steps as below:

 * Step 1: Create a new directory in the `tests` folder with the same name as the system to be added 
 * Step 2: Create two subdirectories named `standard` and `high_accuracy` inside the system directory created above
 * Step 3: Generate the input files and the corresponding output files for the given system
 * Step 4: The output files should be named as: `.refout`,`refstatic`,`refaimd`,`refgeopt`,`refeigen`
 * Step 5: Place the input and reference output files inside the `standard` and `high_accuracy` folders
 * Step 6: Update the `SPARC_testing_script.py` by adding the new system to the dictionary variable named `SYSTEMS` (initialized at line 40) in the end (at line 268)

### (5) Running on the cluster:

The python script is capable of launching the tests on a cluster. First, the `samplescript_cluster` file inside the `tests` folder needs to be replaced with the appropriate job submission script for the given cluster. Then, the lines 15-20 of the file `SPARC_testing_script.py` need to be chnaged for the given cluster. 
 
