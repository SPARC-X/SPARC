# Si2 example

Diamond Si in the primitive cell (non-orthogonal), spin-unpolarized (`SPIN_TYP` 0), 2×2×2 k-points (4 k-points), 12 bands.

This example only demonstrates how to use the code. It is not converged with respect to the mesh, k-points or number of bands (`NSTATES`), so its spectra are not physical results.

## Run

```bash
python run.py                       # serial
python run.py --n_threads 4         # 4 k-point-spin threads (one per k-point)
OMP_NUM_THREADS=4 python run.py     # 4 BLAS threads
```
Any other input of `kubo_greenwood.py` can be added the same way, e.g. `--plot` (`python run.py -h` lists them). Results are written to `KG_output/`; a new run writes numbered files (`Si2.kgout_01`, ...).

## Files

| File | Contents |
|---|---|
| `Si2.inpt`, `Si2.ion` | SPARC input (pseudopotential from SPARC's `psps/` folder) |
| `Si2.out`, `Si2.eigen`, `Si2.psi`, `Si2.static` | SPARC output, the input of the Kubo–Greenwood calculation |
| `run.py` | Kubo–Greenwood run script |
| `KG_output/` | Kubo–Greenwood output: `Si2.kgout`, `Si2.kglog`, `Si2.sigma`, `Si2.epsilon` |

The SPARC output was computed in this folder with `mpirun -np 4 ../../../../lib/sparc -name Si2`.
