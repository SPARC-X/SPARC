#!/bin/bash

# mpirun -np 6 gdb -batch -ex=r -ex=bt -ex=q --args ./sparc -name BaTiO3
#LIBPCE_DEBUG_LEVEL=5 mpirun -np 1 gdb -batch -ex=r -ex=bt -ex=q --args ./sparc -name BaTiO3
#LIBPCE_DEBUG_LEVEL=5 mpirun -np 1 valgrind ./sparc -name BaTiO3
#LIBPCE_DEBUG_LEVEL=25 mpirun -hostfile hostfile --np 16 gdb -batch -ex=r -ex=bt -ex=q --args ./sparc -name BaTiO3 | tee serial_no_them.txt
LIBPCE_DEBUG_LEVEL=25 mpirun -hostfile hostfile --np 1 ./sparc -name BaTiO3 | tee serial_no_them.txt
