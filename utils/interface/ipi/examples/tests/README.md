# i-PI + SPARC tests

Each subdirectory is one validation item from `notes/ipi_sparc_test_todo.md`.
Naming: `<id>_<short_name>`.

i-PI owns nuclear MD. SPARC is a socket force client (`MD_FLAG: 0`).

Shared launcher: `_common/run_md.sh` (source from a test dir after setting `PORT`).

| ID | Folder | What it proves | Status |
|----|--------|----------------|--------|
| A1 | `a1_energy_force/` | i-PI energy/forces match independent SPARC single-points | **PASS** |
| A2 | `a2_units/` | Length / energy / time conversion (Å↔Bohr, Ha↔eV, fs↔ps) | **PASS** |
| A3 | `a3_exit/` | Clean `EXIT` shutdown (rebuild existing source; no C edit) | **PASS** |
| B1 | `b1_nve/` | NVE conserved quantity for 0.5 / 1 / 2 fs | **PASS** |
| B2 | `b2_nvt/` | Langevin NVT holds ~300 K | **PASS** |
| B3 | `b3_npt/` | Isotropic barostat uses SPARC virial | **PASS** (P→V sign deferred) |
| C1 | `c1_umbrella/` | `ffplumed` restraint vs unbiased CV | **PASS** |
| D1 | `d1_mts/` | PLUMED inner / SPARC outer MTS (~4×) | **PASS** |
| E1 | `e1_multiclient/` | Two SPARC clients, `nbeads=2` | **PASS** |
| E2 | `e2_geometry/` | i-PI xyz vs SPARC socket geometries | **PASS** |
| E3 | `e3_restart/` | Continue from i-PI `RESTART` | **PASS** |

Skipped / deferred (no SPARC C change in this pass):

- **B3 directional P→V**: SPARC `.static` stress ≈ +10 GPa, i-PI `pressure_md` ≈ −10 GPa at t=0. Review `stress_to_virial` later.
- **C3** two-window WHAM
- **D3** DFT–DFT MTS
- **UNIX socket**: already covered by `tests/Socket/Al_singlepoint_unix`
- **i-PI geop**: optional, not run

How to run: put `i-pi` and `python3` on `PATH`, `cd` into a folder, `./run_all.sh`.
Optional overrides: `SPARC`, `NP`, `MPIEXEC`, `PLUMED_ROOT`, `PLUMED_KERNEL`.
