# Halo exchange convention test

`halo_convention_test.c` is a regression test for SPARC's Cartesian halo
exchange, and a probe for how the local MPI pairs duplicate neighbour edges.

## What it checks

A periodic global field with a distinct value per cell is distributed over the
process grid, the halo exchange is run, and every ghost cell is compared with
the field value at its wrapped global coordinate. That invariant does not depend
on how the exchange is implemented, so the test does not drift with the code it
guards.

Three exchanges are run per invocation:

| row | path | meaning |
|---|---|---|
| `SPARC halo path` | `SPARC_Ineighbor_alltoallv` | what SPARC actually does — **this is the gate** |
| `unmirrored packing` | bare `MPI_Ineighbor_alltoallv` | diagnostic: does this library apply the swap? |
| `mirrored packing` | bare call, historical packing | diagnostic: does it use list order? |

The exit status depends only on the first row, which must pass on every machine.
The other two report what the library does and are expected to disagree between
machines — that disagreement is the reason the first row exists.

## Why the shape matters

Where every `dims[d] >= 3`, the `-1` and `+1` neighbours along each direction
are different ranks, the pairing is forced by rank matching, and every library
agrees. The convention only becomes observable when `dims[d] <= 2` and direction
`d` is periodic, because both neighbours along `d` are then the same rank.

`make test` runs `3 3 3` (control), `2 3 3` and `1 3 3` (discriminating), and
`1 1 2`. `make test-quick` is the subset that fits a 4-slot CI runner.

## Measured library behaviour

Same source, same `2 3 3` decomposition, opposite results:

| MPI | unmirrored | mirrored | convention |
|---|---|---|---|
| Cray MPICH 9.0.1.498 | pass | fail (31104) | swap |
| MVAPICH2 2.3.7 | fail (31104) | pass | list order |
| Open MPI 4.1.2 | fail (31104) | pass | list order |

At two or more periodic unit dimensions Cray MPICH fails *both* rows, matching
no self-consistent convention. The `SPARC halo path` row passes on all three
libraries, on every shape.

Cray MPICH applies the pairwise swap of MPI-4.0 §8.6 Example 8.10, whose advice
to implementors names the `periods[d]==1 && dims[d]==1-or-2` case; the rule was
clarified as errata against MPI-3.1 in MPI-4.0 Annex B.1.1 item 1. MVAPICH2
2.3.7 carries the pre-errata list order verbatim. No packing inside SPARC is
correct on both, which is why `SPARC_Ineighbor_alltoallv` performs the pairing
itself and is always active.

If **both** diagnostic rows fail on a shape with two or more periodic unit
dimensions, the library matches no self-consistent convention there. That is
expected of MPICH 4.x/5.x and derivatives; see `pmodels/mpich#7945` and
`utils/mpicheck/nbr_match_probe`.
