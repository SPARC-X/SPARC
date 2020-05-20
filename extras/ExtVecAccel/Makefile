# Makefile for External Vectorization Acceleration (EVA) module
#    Copyright (c) 2018-2019 Edmond Group at Georgia Tech
# This makefile will compile:
# (1) .c files in ../, the .o file are stored in this directory 
# (2) EVA module files in this directory 
# (3) SPARC executable file with EVA module, stored to ../../lib/sparc-EVA

# Specify the path MKLROOT if necessary, e.g.
# MKLROOT = /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl

# If you are using Intel compiler for non-Intel CPUs, you may need to remove -xHost in this file

# If you need to run with DEBUG mode, add -DDEBUG
CFLAGS     = -m64 -I${MKLROOT}/include -I../include/ -I./ -O3 -qopenmp -DUSE_EVA_MODULE

EVA_CFLAGS = -m64 -I../include/ -I./ -O3 -qopenmp -g -xHost -Wall -std=gnu99

# Set USE_CUDA / USE_MKL = 0 to disable using CUDA kernels / MKL specific functions
USE_CUDA   = 1
USE_MKL    = 1

CUDA_PATH  ?= /usr/local/pacerepov1/cuda/9.0

# Set USE_GTM = 0 to disable using GTMatrix
USE_GTM  = 1
GTM_PATH = ../GTMatrix

# Adjust the -arch=sm_xx according to your GPU architecture
# See: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
NVFLAGS   = -I../include/ -I./ -m64 -Wno-deprecated-gpu-targets -O3 -g -arch=sm_60 

LDFLAGS   = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -ldl -liomp5 -lrt -lpthread -lmkl_core \
            -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_blacs_intelmpi_lp64 \
            -m64 -qopenmp -g -O3 -xHost

COBJS     = main.o initialization.o readfiles.o atomdata.o parallelization.o relax.o tools.o md.o    \
            electrostatics.o electronicGroundState.o electronDensity.o orbitalElecDensInit.o         \
            occupation.o lapVecRoutines.o gradVecRoutines.o gradVecRoutinesKpt.o nlocVecRoutines.o   \
            hamiltonianVecRoutines.o lapVecOrth.o lapVecOrthKpt.o lapVecNonOrth.o lapVecNonOrthKpt.o \
            linearSolver.o mixing.o exchangeCorrelation.o eigenSolver.o eigenSolverKpt.o energy.o    \
            forces.o stress.o pressure.o finalization.o                                              \
            EVA_buff.o EVA_Lap_MV_common.o EVA_Lap_MV_orth.o EVA_Lap_MV_nonorth.o EVA_Vnl_MV.o       \
            EVA_Hamil_MV.o EVA_CheFSI.o CSRPlus_utils.o CSRPlus_kernels.o
 
ifeq ($(USE_CUDA), 1)
EVA_CFLAGS += -DUSE_EVA_CUDA_MODULE
LDFLAGS    += -L$(CUDA_PATH)/lib64 -lcudart -lcusparse 
NVFLAGS    += -DUSE_EVA_CUDA_MODULE
COBJS      += EVA_CheFSI_CUDA.o EVA_CheFSI_CUDA_drivers.o
endif

ifeq ($(USE_MKL), 1)
EVA_CFLAGS += -I${MKLROOT}/include -mkl -DUSE_EVA_MKL_MODULE 
endif

ifeq ($(USE_GTM), 1)
CFLAGS   += -I$(GTM_PATH) -qopenmp -DUSE_GTMATRIX
LDFLAGS  += -qopenmp
COBJS    += $(GTM_PATH)/libGTMatrix.a #Jacobi_dsyev.o 
endif   

EXEC = ../../lib/sparc-EVA

# Use CC=mpicc if you are not using Intel compiler + Intel MPI
CC   = mpiicc
NVCC = nvcc

all: sparc

sparc: $(COBJS)
	$(CC) $(LDFLAGS) -o $(EXEC) $^

EVA_buff.o: Makefile EVA_buff.h EVA_buff.c
	$(CC) $(EVA_CFLAGS) -c EVA_buff.c    

EVA_Lap_MV_common.o: Makefile EVA_buff.h EVA_Lap_MV_common.c
	$(CC) $(EVA_CFLAGS) -c EVA_Lap_MV_common.c  
    
EVA_Lap_MV_orth.o: Makefile EVA_buff.h EVA_Lap_MV_common.h EVA_Lap_MV_orth.c
	$(CC) $(EVA_CFLAGS) -c EVA_Lap_MV_orth.c  

EVA_Lap_MV_nonorth.o: Makefile EVA_buff.h EVA_Lap_MV_common.h EVA_Lap_MV_nonorth.c
	$(CC) $(EVA_CFLAGS) -c EVA_Lap_MV_nonorth.c  
    
EVA_Vnl_MV.o: Makefile EVA_buff.h EVA_Vnl_MV.c
	$(CC) $(EVA_CFLAGS) -c EVA_Vnl_MV.c  
    
EVA_Hamil_MV.o: Makefile EVA_buff.h EVA_Lap_MV_orth.h EVA_Lap_MV_nonorth.h EVA_Vnl_MV.h CSRPlus.h EVA_Hamil_MV.c
	$(CC) $(EVA_CFLAGS) -c EVA_Hamil_MV.c

EVA_CheFSI.o: Makefile EVA_buff.h EVA_Hamil_MV.h EVA_CheFSI.h CSRPlus.h EVA_CheFSI.c
	$(CC) $(EVA_CFLAGS) -c EVA_CheFSI.c

EVA_CheFSI_CUDA.o: Makefile EVA_buff.h EVA_Hamil_MV.h EVA_CheFSI_CUDA_drivers.h EVA_CheFSI_CUDA.c
	$(CC) $(EVA_CFLAGS) -c EVA_CheFSI_CUDA.c

EVA_CheFSI_CUDA_drivers.o: Makefile EVA_CheFSI_CUDA_buff.h EVA_CheFSI_CUDA_drivers.cu EVA_CheFSI_CUDA_kernels.cuh
	$(NVCC) $(NVFLAGS)  -c EVA_CheFSI_CUDA_drivers.cu

CSRPlus_utils.o: Makefile CSRPlus.h
	$(CC) $(EVA_CFLAGS) -c CSRPlus_utils.c

CSRPlus_kernels.o: Makefile CSRPlus.h
	$(CC) $(EVA_CFLAGS) -c CSRPlus_kernels.c

main.o: ../main.c
	$(CC) $(CFLAGS) -c ../main.c  
    
initialization.o: ../initialization.c
	$(CC) $(CFLAGS) -c ../initialization.c 
    
readfiles.o: ../readfiles.c
	$(CC) $(CFLAGS) -c ../readfiles.c 

atomdata.o: ../atomdata.c
	$(CC) $(CFLAGS) -c ../atomdata.c  

parallelization.o: ../parallelization.c
	$(CC) $(CFLAGS) -c ../parallelization.c  

relax.o: ../relax.c
	$(CC) $(CFLAGS) -c ../relax.c  

tools.o: ../tools.c
	$(CC) $(CFLAGS) -c ../tools.c  

md.o: ../md.c
	$(CC) $(CFLAGS) -c ../md.c  

electrostatics.o: ../electrostatics.c
	$(CC) $(CFLAGS) -c ../electrostatics.c  

electronicGroundState.o: ../electronicGroundState.c
	$(CC) $(CFLAGS) -c ../electronicGroundState.c  

electronDensity.o: ../electronDensity.c
	$(CC) $(CFLAGS) -c ../electronDensity.c  

orbitalElecDensInit.o: ../orbitalElecDensInit.c
	$(CC) $(CFLAGS) -c ../orbitalElecDensInit.c  

occupation.o: ../occupation.c
	$(CC) $(CFLAGS) -c ../occupation.c  

lapVecRoutines.o: ../lapVecRoutines.c
	$(CC) $(CFLAGS) -c ../lapVecRoutines.c  

gradVecRoutines.o: ../gradVecRoutines.c
	$(CC) $(CFLAGS) -c ../gradVecRoutines.c  

gradVecRoutinesKpt.o: ../gradVecRoutinesKpt.c
	$(CC) $(CFLAGS) -c ../gradVecRoutinesKpt.c 

nlocVecRoutines.o: ../nlocVecRoutines.c
	$(CC) $(CFLAGS) -c ../nlocVecRoutines.c  

hamiltonianVecRoutines.o: ../hamiltonianVecRoutines.c
	$(CC) $(CFLAGS) -c ../hamiltonianVecRoutines.c  

lapVecOrth.o: ../lapVecOrth.c
	$(CC) $(CFLAGS) -c ../lapVecOrth.c 

lapVecOrthKpt.o: ../lapVecOrthKpt.c
	$(CC) $(CFLAGS) -c ../lapVecOrthKpt.c 

lapVecNonOrth.o: ../lapVecNonOrth.c
	$(CC) $(CFLAGS) -c ../lapVecNonOrth.c  
    
lapVecNonOrthKpt.o: ../lapVecNonOrthKpt.c
	$(CC) $(CFLAGS) -c ../lapVecNonOrthKpt.c  

linearSolver.o: ../linearSolver.c
	$(CC) $(CFLAGS) -c ../linearSolver.c  

mixing.o: ../mixing.c
	$(CC) $(CFLAGS) -c ../mixing.c 

exchangeCorrelation.o: ../exchangeCorrelation.c
	$(CC) $(CFLAGS) -c ../exchangeCorrelation.c  

eigenSolver.o: ../eigenSolver.c
	$(CC) $(CFLAGS) -c ../eigenSolver.c  

eigenSolverKpt.o: ../eigenSolverKpt.c
	$(CC) $(CFLAGS) -c ../eigenSolverKpt.c  

energy.o: ../energy.c
	$(CC) $(CFLAGS) -c ../energy.c  

forces.o: ../forces.c
	$(CC) $(CFLAGS) -c ../forces.c  
    
stress.o: ../stress.c
	$(CC) $(CFLAGS) -c ../stress.c
    
pressure.o: ../pressure.c
	$(CC) $(CFLAGS) -c ../pressure.c

finalization.o: ../finalization.c
	$(CC) $(CFLAGS) -c ../finalization.c  

$(GTM_PATH)/libGTMatrix.a: 
	cd $(GTM_PATH) && MPICC=$(CC) make

.PHONY: clean
clean:
	rm -f  $(COBJS) $(EXEC) 
