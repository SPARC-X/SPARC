#!/bin/bash

# mpirun -np 6 gdb -batch -ex=r -ex=bt -ex=q --args ./sparc -name BaTiO3
mpirun -np 1 valgrind ./sparc -name BaTiO3
