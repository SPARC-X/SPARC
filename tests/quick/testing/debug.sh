#!/bin/bash

mpirun -np 1 gdb -batch -ex=r -ex=bt -ex=q --args ./sparc -name BaTiO3
