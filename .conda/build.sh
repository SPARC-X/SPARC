#!/bin/bash
echo Compiling sparc using ${CPU_COUNT} cores
cd ./src
make clean
make -j ${CPU_COUNT} USE_MKL=0 USE_SCALAPACK=1
# ls -al lib
echo "Installing sparc into $PREFIX/bin"
cp ../lib/sparc $PREFIX/bin
echo "Moving sparc psp into $PREFIX/share/sparc/psps"
mkdir -p $PREFIX/share/sparc/psps
cp ../psps/* $PREFIX/share/sparc/psps/
echo "Finish compiling sparc!"
