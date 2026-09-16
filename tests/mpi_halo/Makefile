# Regression test for the Cartesian halo-exchange block convention.
# See halo_convention_test.c for what it pins and why.

CC      ?= mpicc
CFLAGS  ?= -std=gnu11 -O2 -Wall -Wextra
MPIRUN  ?= mpirun

# This directory carries a copy of the two SPARC files it links against, so it
# can be copied to another machine and built on its own. Inside the repo the
# canonical files in src/ are used instead, and the bundled copies are checked
# against them -- a stale copy is a build error, not a silent divergence.
# Refresh them with 'make sync' after editing src/mpi_nbr_shim.{c,h}.
CANON_SRC := ../../src/mpi_nbr_shim.c
CANON_INC := ../../src/include/mpi_nbr_shim.h

ifneq ($(wildcard $(CANON_SRC)),)
  SHIM_SRC   := $(CANON_SRC)
  SHIM_INCD  := ../../src/include
  SYNC_CHECK := check-sync
else
  SHIM_SRC   := mpi_nbr_shim.c
  SHIM_INCD  := .
  SYNC_CHECK :=
endif

all: halo_convention_test

halo_convention_test: halo_convention_test.c $(SHIM_SRC) $(SYNC_CHECK)
	$(CC) $(CFLAGS) -I$(SHIM_INCD) -o $@ halo_convention_test.c $(SHIM_SRC)

# Guard against the bundled copy drifting away from src/.
check-sync:
	@cmp -s $(CANON_SRC) mpi_nbr_shim.c && cmp -s $(CANON_INC) mpi_nbr_shim.h || { \
	  echo "ERROR: tests/mpi_halo/mpi_nbr_shim.{c,h} differ from src/. Run 'make sync'."; \
	  exit 1; }

sync:
	cp $(CANON_SRC) mpi_nbr_shim.c
	cp $(CANON_INC) mpi_nbr_shim.h

# Shapes chosen to cover the three regimes:
#   3 3 3  no duplicate edge          -- control, convention is invisible
#   2 3 3  one duplicate pair         -- discriminates the packing convention
#   1 3 3  one duplicate pair, self   -- same, via a unit dimension
#   1 1 2  two periodic unit dims     -- also exercises the MPI library defect
test: halo_convention_test
	$(MPIRUN) -n 27 ./halo_convention_test 3 3 3
	$(MPIRUN) -n 18 ./halo_convention_test 2 3 3
	$(MPIRUN) -n 9  ./halo_convention_test 1 3 3
	$(MPIRUN) -n 2  ./halo_convention_test 1 1 2

# Subset that runs on a 4-slot CI runner.
test-quick: halo_convention_test
	$(MPIRUN) -n 4 ./halo_convention_test 1 2 2
	$(MPIRUN) -n 2 ./halo_convention_test 1 1 2
	$(MPIRUN) -n 1 ./halo_convention_test 1 1 1

clean:
	rm -f halo_convention_test

.PHONY: all test test-quick clean check-sync sync
