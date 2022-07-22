# Choose whether to use MKL or ScaLAPACK or neither
# Warning: don't turn on USE_MKL and USE_SCALAPACK simultaneously
# Set USE_MKL = 1 to compile with MKL BLAS, LAPACK, and ScaLAPACK
# Set USE_MKL = 0 otherwise
USE_MKL       = 1
# Set USE_SCALAPACK = 1 to compile with non-MKL BLAS, LAPACK, and ScaLAPACK
# Set USE_SCALAPACK = 0 to compile with non-MKL BLAS and LAPACK only
USE_SCALAPACK = 0
# Set USE_DP_SUBEIG = 1 to use SPARC rather than ScaLAPACK routines for matrix data distribution
# (USE_DP_SUBEIG = 1 is required if both USE_MKL = 0 and USE_SCALAPACK = 0)
# Set USE_DP_SUBEIG = 0 to use ScaLAPACK rather than SPARC routines
USE_DP_SUBEIG = 1
# Set USE_FFTW = 1 to use FFTW for fast Fourier transform in vdWDF. Don't open it together with USE_MKL
USE_FFTW      = 0

# Set DEBUG_MODE = 1 to run with debug mode and print debug output
DEBUG_MODE    = 1

# Set USE_PCE=1 to use PCE, or =0 to not use PCE
USE_PCE := 1

ifeq ($(USE_PCE), 1)
# Run some sanity checks to ensure multiple accelerators aren't enabled
ifeq ($(USE_DP_SUBEIG), 0)
$(error USE_PCE=1 is currently only tested with USE_DP_SUBEIG=1)
endif
ifeq ($(ACCEL), 1)
$(error USE_PCE=1 is currently only tested with ACCEL=0)
endif
ifeq ($(USE_EVA_MODULE), 1)
$(error USE_PCE=1 is currently only tested with USE_EVA_MODULE=0)
endif

# Set USE_GPU =1 to enable PCE GPU support. 
# If set, you can use the environment variable LIBPCE_USE_GPU to control at 
# runtime if it is ran on the CPU or GPU
USE_GPU    := 1

# Set USE_EPLA=1 to enable ELPA support for PCE Eigensolves
USE_ELPA := 1

else
USE_GPU    := 0
USE_ELPA := 0
endif
DEBUG_MODE    = 1

# Enable SIMD vectorization for complex stencil routines
# CAUTION: for some compilers this results in wrong results! Use for intel/19.0.3 or later versions
ENABLE_SIMD_COMPLEX = 1

# Specify the path MKLROOT if it's not already set to compile with MKL, e.g,
# MKLROOT = /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl

# Specify the path to ScaLAPACK, LAPACK and BLAS if necessary, and 
# add to LDFLAGS. Note that sometimes LDFLAGS already contains the default 
# path to these libraries, or the libraries are located in the default search
# path. In those cases, the following is not needed.
# SCALAPACKROOT = /nv/hp27/qxu78/data/scalapack-2.0.2
# LDFLAGS += -L$(SCALAPACKROOT)
# LAPACKROOT = /usr/local/pacerepov1/lapack/3.6.0
# LDFLAGS += -L$(LAPACKROOT)
# BLASROOT = /usr/lib64
# LDFLAGS += -L$(BLASROOT)

CPPFLAGS = -Iinclude/ -Ixc/exx/include -Ixc/vdW/d3/include -Ixc/vdW/vdWDF/include -Ixc/mgga/include -IorderN/sq/include
LDLIBS   = -lrt

LIBPCE_DEP = main.o # Dummy value for dependency
ifeq ($(USE_PCE), 1)
CPPFLAGS += -DUSE_PCE=1
LIBPCE_DIR=../Hamiltonian/
CA3DMM_DIR=$(LIBPCE_DIR)/CA3DMM

# Make sure you have ELPA_ROOT set in the environment variable
# This is set by the elpa modulefile by PACE

#ELPA_ROOT =/usr/lib/elpa/
ELPA_DIR = $(ELPA_ROOT)
#ELPA_DIR =/usr/include/elpa_openmp-2020.11.001/elpa/

# This may need to be changed depending on the installed elpa
# The one used by the elpa module is currently elpa_openmp
ELPA_LIB = elpa
#ELPA_LIB = elpa_openmp

# Select which makefile should be used for LIBPCE
# On PACE with MKL you cane use Makefile.pace.icc, the most well tested option
LIBPCE_MAKEFILE=Makefile.pace.icc

# PCE uses CA3DMM for matrix-matrix products, so we must make sure to compile with CA3DMM support
# SPARC_COMPILATION indicates that PCE is being compiled within SPARC, and so doesn't need to redefine certain structs
CPPFLAGS += -I $(LIBPCE_DIR)/include -I ../Hamiltonian/CA3DMM/include -DSPARC_COMPILATION
LDLIBS += -L $(LIBPCE_DIR)/lib -L $(CA3DMM_DIR)/lib  -lpce -lca3dmm

#If we are to use GPUs setup the flags for GPU
ifeq ($(USE_GPU), 1)
LFLAGS_GPU_HOST = -std=gnu++98,-MMD,-MP,-O3,-g,-march=native,-fopenmp
CPPFLAGS += -I $(CUDA_ROOT)/include
LFLAGS_GPU:=-lcudart -ccbin=mpicxx  -O3 -L$(CUDA_ROOT)/lib64 -lcurand -lcusparse -lcublas -lcusolver -arch=sm_70 -gencode=arch=compute_70,code=sm_70 -G $(LDLIBS)

ifeq ($(USE_MKL), 1)
LFLAGS_GPU_HOST:=$(LFLAGS_GPU_HOST)
LFLAGS_GPU   += -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
else
LFLAGS_GPU_HOST:=$(LFALGS_GPU_HOST),-lopenblas
endif
LFLAGS_GPU+=-Xcompiler $(LFLAGS_GPU_HOST) 

endif
LIBPCE_DEP = $(LIBPCE_DIR)/lib/libpce.a

endif


		


ifeq ($(USE_MKL), 1)
CPPFLAGS += -m64 -I${MKLROOT}/include -DUSE_MKL
LDFLAGS   = -L${MKLROOT}/lib/intel64
LDLIBS   += -Wl,-rpath=${MKLROOT}/lib/intel64,--no-as-needed -lmkl_scalapack_lp64 -lmkl_cdft_core -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
else ifeq ($(USE_SCALAPACK), 1)
CPPFLAGS += -DUSE_SCALAPACK
LDLIBS   += -lscalapack
endif

ifeq ($(USE_GPU), 1)
CPPFLAGS += -DUSE_GPU=1
endif

ifeq ($(USE_ELPA), 1)
LDLIBS += -L $(ELPA_ROOT)/lib -l$(ELPA_LIB)
LFLAGS_GPU += -L $(ELPA_ROOT)/lib -l$(ELPA_LIB)
LDFLAGS += -I $(ELPA_DIR)
endif

# if MKL is not used, link BLAS and LAPACK directly
ifeq ($(USE_MKL), 0)
# if you are using BLAS instead of OpenBLAS, change -lopenblas to -lblas
# and move it to after -llapack
LDLIBS += -lopenblas -lm
#LDLIBS += -lopenblas -llapack -llapacke -lm
#LDLIBS += -llapacke -llapack -lblas -lm
endif

# if FFTW is used, link fftw
ifeq ($(USE_FFTW), 1)
CPPFLAGS += -DUSE_FFTW
LDLIBS += -lfftw3_mpi -lfftw3 # fftw needs to be loaded before compile
endif

# To use domain parallelization + LAPACK for solving sobspace eigen problem
ifeq ($(USE_DP_SUBEIG), 1)
CPPFLAGS += -DUSE_DP_SUBEIG
endif

# to compile with DEBUG mode
ifeq ($(DEBUG_MODE), 1)
CPPFLAGS += -Wall -g -DDEBUG -Werror-implicit-function-declaration
endif

# to enable SIMD for complex stencil routines
ifeq ($(ENABLE_SIMD_COMPLEX), 1)
CPPFLAGS += -DENABLE_SIMD_COMPLEX
endif

# for old Intel compiler, use -qopenmp instead of -fopenmp. ICC 17 and later also accepts -fopenmp. 
CFLAGS = -std=gnu99 -O3 -fopenmp

OBJSC = main.o initialization.o readfiles.o atomdata.o parallelization.o relax.o tools.o md.o    \
        electrostatics.o electronicGroundState.o electronDensity.o orbitalElecDensInit.o         \
        occupation.o lapVecRoutines.o gradVecRoutines.o gradVecRoutinesKpt.o nlocVecRoutines.o   \
        hamiltonianVecRoutines.o lapVecOrth.o lapVecOrthKpt.o lapVecNonOrth.o lapVecNonOrthKpt.o \
        linearSolver.o mixing.o exchangeCorrelation.o eigenSolver.o eigenSolverKpt.o energy.o    \
        forces.o stress.o pressure.o finalization.o spinOrbitCoupling.o                          \
        xc/vdW/d3/d3correction.o xc/vdW/d3/d3findR0ab.o xc/vdW/d3/d3copyC6.o                     \
        xc/vdW/d3/d3initialization.o xc/vdW/d3/d3finalization.o xc/vdW/d3/d3forceStress.o        \
        xc/vdW/vdWDF/vdWDFinitialization.o xc/vdW/vdWDF/vdWDFfinalization.o                      \
        xc/vdW/vdWDF/vdWDFexchangeLinearCorre.o xc/vdW/vdWDF/vdWDFnonlinearCorre.o               \
        xc/vdW/vdWDF/vdWDFstress.o xc/vdW/vdWDF/vdWDFgenerateKernelSpline.o                      \
        xc/mgga/energyDensity.o xc/mgga/MGGAscan.o xc/mgga/MGGASscan.o                           \
        xc/mgga/MGGAhamiltonianTerm.o xc/mgga/MGGAinitialization.o xc/mgga/MGGAstress.o          \
        xc/mgga/MGGAfinalization.o xc/mgga/MGGAexchangeCorrelation.o                             \
        xc/exx/exactExchange.o xc/exx/exactExchangeKpt.o                                         \
        xc/exx/exactExchangeInitialization.o xc/exx/exactExchangeFinalization.o                  \
        xc/exx/exactExchangeStress.o xc/exx/exactExchangePressure.o orderN/sq/sq.o               \
        orderN/sq/sqInitialization.o orderN/sq/sqFinalization.o orderN/sq/sqEnergy.o             \
        orderN/sq/sqDensity.o orderN/sq/sqNlocVecRoutines.o orderN/sq/sqParallelization.o        \
        orderN/sq/sqProperties.o orderN/sq/sqtool.o   

ifeq ($(USE_PCE), 1)
OBJSC += hamstruct.o pce_interface.o
endif

LIBBASE = ../lib/sparc
TESTBASE = ../.ci

override CC=mpicc

all: sparc

# Note the implicit rule to compile '.c' files into '.o' files is
# %.o : %.c
# 	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

# Use either nvcc or $(CC) depending on if PCE GPU support is requested
sparc: $(OBJSC) $(LIBPCE_DEP) 
	if [ "$(USE_GPU)" == "1" ]; then \
		nvcc -DUSE_GPU $(LFLAGS_GPU) $(LDFLAGS) -o $(LIBBASE) $^; \
	else \
		$(CC) $(CFLAGS) $(LDFLAGS) -o $(LIBBASE) $^ $(LDLIBS); \
	fi

.PHONY: clean
clean:
	rm -f  $(OBJSC) $(LIBBASE)
test: ../tests/test.py
	cd ../tests; python test.py 
