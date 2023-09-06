#!/bin/bash
# The build script is intended to be used on the root level
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
mkdir -p $PREFIX/doc/sparc
cp -r ../doc/ $PREFIX/doc/sparc/
echo "Finish compiling sparc!"

# Copy activate and deactivate scripts
cd ../.conda/
cp activate.sh $PREFIX/etc/conda/activate.d/activate-sparc.sh
cp deactivate.sh $PREFIX/etc/conda/deactivate.d/deactivate-sparc.sh
echo "Finish setting up activate / deactivate scripts"
