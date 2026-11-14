# Fe_spin example

bcc Fe in the conventional cubic cell (2 atoms), collinear spin (`SPIN_TYP` 1), one k-point (1/4, 1/4, 1/4), 24 bands; a metal, with a non-zero Drude term.

This example only demonstrates how to use the code. It is not converged with respect to the mesh, k-points or number of bands (`NSTATES`), so its spectra are not physical results.

## Run

```bash
python run.py                       # serial
python run.py --n_threads 2         # 2 k-point-spin threads (one per spin)
OMP_NUM_THREADS=4 python run.py     # 4 BLAS threads
```
Any other input of `kubo_greenwood.py` can be added the same way, e.g. `--plot` (`python run.py -h` lists them). Results are written to `KG_output/`; a new run writes numbered files (`Fe_spin.kgout_01`, ...).

## Files

| File | Contents |
|---|---|
| `Fe_spin.inpt`, `Fe_spin.ion` | SPARC input (pseudopotential from SPARC's `psps/` folder) |
| `Fe_spin.out`, `Fe_spin.eigen`, `Fe_spin.psi`, `Fe_spin.static` | SPARC output, the input of the Kubo–Greenwood calculation |
| `run.py` | Kubo–Greenwood run script |
| `KG_output/` | Kubo–Greenwood output: `Fe_spin.kgout`, `Fe_spin.kglog`, `Fe_spin.sigma`, `Fe_spin.epsilon` |

The SPARC output was computed in this folder with `mpirun -np 4 ../../../../lib/sparc -name Fe_spin`.
