# PLUMED API tests (SPARC socket + Python PLUMED)

SPARC is a socket **DFT** (density functional theory) client (`MD_FLAG: 0`).
A Python driver integrates the nuclei and calls **PLUMED** (PLUgin for
MolEcular Dynamics) through `plumed.Plumed()`. SPARC is **not** rebuilt and
does **not** use `PLUMED_FLAG`.

Same 4-atom **FCC** (face-centered cubic) Al cell as `ipi_sparc_tests`.

| ID | Folder | DFT | Bias / CV | Integrator |
|----|--------|-----|-----------|------------|
| A0 | `a0_socket_smoke/` | SPARC socket | none | one single-point |
| A1 | `a1_energy/` | SPARC socket | PLUMED `ENERGY` | 3 rattled single-points |
| B1 | `b1_cv/` | SPARC socket | PLUMED `DISTANCE` | Langevin, 8 steps |
| B2 | `b2_umbrella/` | SPARC socket | harmonic restraint | Verlet, 15+15 steps |

`_common/` holds `Al.inpt` / `Al.ion` / `init.xyz`. Tutorial:
[`../tutorial.md`](../tutorial.md) (Part I).

```bash
# python3, numpy, py-plumed, and an MPI launcher must already be on PATH.
# Optional: SPARC, NP, MPIEXEC, PLUMED_ROOT, PLUMED_KERNEL
cd utils/interface/plumed/examples
./check_env.sh
./run_all.sh
```

Or one test:

```bash
source _common/env.sh
cd a0_socket_smoke && ./run_all.sh
```
