module load gcc openblas cuda mvapich2-gdr  
git checkout integration 
git submodule update --init --recursive 
cd Hamiltonian 
# Edit Makefile.pace.gcc to include USE_GPU=1 TRY_CAM=1 
sed -i -e 's/\(USE_GPU\s*\):= [01]/\1:= 1/' Makefile.pace.gcc 
sed -i -e 's/\(TRY_CAM\s*\):= [01]/\1:= 1/' Makefile.pace.gcc
cd CA3DMM/src
# Edit makefile to include USE_GPU=1 TRY_CAM=1
sed -i -e 's/\(USE_GPU\s*\):= [01]/\1:= 1/' gcc-openblas-anympi.make
cd ../..
module load gcc openblas cuda mvapich2-gdr && make -f Makefile.pace.gcc
cd ../src
# Edit makefile to include USE_GPU=1
sed -i -e 's/\(USE_GPU\s*\):= [01]/\1:= 1/' makefile
module load gcc openblas cuda mvapich2-gdr && make -j
