# Choose whether to use MKL or ScaLAPACK or neither
# Warning: don't turn on USE_MKL and USE_SCALAPACK simultaneously
# Set USE_MKL = 1 to compile with MKL BLAS, LAPACK, and ScaLAPACK
# Set USE_MKL = 0 otherwise
USE_MKL       = 0
# Set USE_SCALAPACK = 1 to compile with non-MKL BLAS, LAPACK, and ScaLAPACK
# Set USE_SCALAPACK = 0 to compile with non-MKL BLAS and LAPACK only
USE_SCALAPACK = 0
# Set USE_DP_SUBEIG = 1 to use SPARC rather than ScaLAPACK routines for matrix data distribution
# (USE_DP_SUBEIG = 1 is required if both USE_MKL = 0 and USE_SCALAPACK = 0)
# Set USE_DP_SUBEIG = 0 to use ScaLAPACK rather than SPARC routines
USE_DP_SUBEIG = 1
# Set DEBUG_MODE = 1 to run with debug mode and print debug output
DEBUG_MODE    = 0

# Enable SIMD vectorization for complex stencil routines
# CAUTION: for some compilers this results in wrong results! Use for intel/19.0.3 or later versions
ENABLE_SIMD_COMPLEX = 0

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

CPPFLAGS = -Iinclude/
LDLIBS   = -lrt

ifeq ($(USE_MKL), 1)
CPPFLAGS += -m64 -I${MKLROOT}/include -DUSE_MKL
LDFLAGS   = -L${MKLROOT}/lib/intel64
LDLIBS   += -Wl,-rpath=${MKLROOT}/lib/intel64,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl
else ifeq ($(USE_SCALAPACK), 1)
CPPFLAGS += -DUSE_SCALAPACK
LDLIBS   += -lscalapack
endif

# if MKL is not used, link BLAS and LAPACK directly
ifeq ($(USE_MKL), 0)
# if you are using BLAS instead of OpenBLAS, change -lopenblas to -lblas
# and move it to after -llapack
LDLIBS += -lopenblas -llapacke -llapack -lm
#LDLIBS += -llapacke -llapack -lblas -lm
endif

# To use domain parallelization + LAPACK for solving sobspace eigen problem
ifeq ($(USE_DP_SUBEIG), 1)
CPPFLAGS += -DUSE_DP_SUBEIG
endif

# to compile with DEBUG mode
ifeq ($(DEBUG_MODE), 1)
CPPFLAGS += -Wall -g -DDEBUG
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
        forces.o stress.o pressure.o finalization.o

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
test1: ../tests/test.py
	cd ../tests; python test.py 
