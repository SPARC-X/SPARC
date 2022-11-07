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
USE_DP_SUBEIG = 0
# Set USE_FFTW = 1 to use FFTW for fast Fourier transform in vdWDF. Don't open it together with USE_MKL
USE_FFTW      = 0
# Set DEBUG_MODE = 1 to run with debug mode and print debug output
DEBUG_MODE    = 0

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

ifeq ($(USE_MKL), 1)
CPPFLAGS += -m64 -I${MKLROOT}/include -DUSE_MKL
LDFLAGS   = -L${MKLROOT}/lib/intel64
LDLIBS   += -Wl,-rpath=${MKLROOT}/lib/intel64,--no-as-needed -lmkl_scalapack_lp64 -lmkl_cdft_core -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
else ifeq ($(USE_SCALAPACK), 1)
CPPFLAGS += -DUSE_SCALAPACK
LDLIBS   += -lscalapack
endif

# if MKL is not used, link BLAS and LAPACK directly
ifeq ($(USE_MKL), 0)
# if you are using BLAS instead of OpenBLAS, change -lopenblas to -lblas
# and move it to after -llapack
LDLIBS += -lopenblas -lm
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
CPPFLAGS += -Wall -g -DDEBUG # -Werror-implicit-function-declaration
endif

# to enable SIMD for complex stencil routines
ifeq ($(ENABLE_SIMD_COMPLEX), 1)
CPPFLAGS += -DENABLE_SIMD_COMPLEX
endif

# for old Intel compiler, use -qopenmp instead of -fopenmp. ICC 17 and later also accepts -fopenmp. 
CFLAGS = -std=gnu99 -O3 -fopenmp

OBJSC = main.o initialization.o readfiles.o atomdata.o parallelization.o relax.o tools.o md.o      \
        electrostatics.o electronicGroundState.o electronDensity.o orbitalElecDensInit.o           \
        occupation.o gradVecRoutines.o gradVecRoutinesKpt.o nlocVecRoutines.o     \
        hamiltonianVecRoutines.o lapVecRoutines.o lapVecRoutinesKpt.o \
        linearSolver.o mixing.o exchangeCorrelation.o eigenSolver.o eigenSolverKpt.o energy.o      \
        forces.o stress.o pressure.o finalization.o spinOrbitCoupling.o printing.o \
        linearAlgebra.o \
        xc/vdW/d3/d3correction.o xc/vdW/d3/d3findR0ab.o xc/vdW/d3/d3copyC6.o                       \
        xc/vdW/d3/d3initialization.o xc/vdW/d3/d3finalization.o xc/vdW/d3/d3forceStress.o          \
        xc/vdW/vdWDF/vdWDFinitialization.o xc/vdW/vdWDF/vdWDFfinalization.o                        \
        xc/vdW/vdWDF/vdWDFexchangeLinearCorre.o xc/vdW/vdWDF/vdWDFnonlinearCorre.o                 \
        xc/vdW/vdWDF/vdWDFstress.o xc/vdW/vdWDF/vdWDFgenerateKernelSpline.o                        \
        xc/mgga/MGGAscan.o xc/mgga/MGGASscan.o xc/mgga/MGGAhamiltonianTerm.o xc/mgga/MGGAstress.o  \
        xc/mgga/MGGAinitialization.o xc/mgga/MGGAfinalization.o xc/mgga/MGGAexchangeCorrelation.o  \
        xc/exx/exactExchange.o xc/exx/exactExchangeKpt.o xc/exx/exactExchangeInitialization.o      \
        xc/exx/exactExchangeFinalization.o xc/exx/exactExchangeStress.o                            \
        xc/exx/exactExchangePressure.o xc/exx/exactExchangeEnergyDensity.o \
        orderN/sq/sq.o orderN/sq/sqInitialization.o orderN/sq/sqFinalization.o orderN/sq/sqEnergy.o\
        orderN/sq/sqDensity.o orderN/sq/sqNlocVecRoutines.o orderN/sq/sqParallelization.o          \
        orderN/sq/sqProperties.o orderN/sq/sqtool.o   

LIBBASE = ../lib/sparc
TESTBASE = ../.ci

override CC=mpicc

all: sparc

# Note the implicit rule to compile '.c' files into '.o' files is
# %.o : %.c
# 	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

sparc: $(OBJSC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(LIBBASE) $^ $(LDLIBS)

.PHONY: clean
clean:
	rm -f  $(OBJSC) $(LIBBASE)
test: ../tests/SPARC_testing_script.py
	cd ../tests; python SPARC_testing_script.py
