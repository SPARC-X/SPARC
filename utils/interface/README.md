# External interfaces

SPARC can be a socket force client for an external molecular-dynamics
program. This directory holds those interfaces. It is not linked into the
SPARC binary.

| Path | Role |
|------|------|
| `ipi/tutorial.md` | How to run the i-PI interface. |
| `plumed/tutorial.md` | How to run the Python PLUMED interface (Part I). |
| `plumed/driver/` | Python package: socket client, `plumed.Plumed()`, and the integrator. |
| `ipi/examples/Al_FCC/` | Minimal 5-step NVE example. |
| `ipi/examples/tests/` | NVE, NVT, NPT, umbrella, multiple clients, and restart checks. |
| `plumed/examples/` | A0–B2 runs of the Python driver. |

Build the socket client with `USE_SOCKET=1`. The binary is `lib/sparc`.
Pseudopotentials are read from `psps/`.

The launch scripts do not activate a module or a conda environment. Put
`i-pi` (i-PI examples), `python3`, and an MPI launcher on `PATH` first.
These variables override the defaults:

| Variable | Default | Use |
|----------|---------|-----|
| `SPARC` | `<repo>/lib/sparc` | Socket binary |
| `NP` | `4` | MPI ranks |
| `MPIEXEC` | `mpirun -np` | Words before the rank count. On Slurm, `MPIEXEC="srun -n"`. |
| `PLUMED_ROOT` | unset | PLUMED prefix, used when `lib/` exists |
| `PLUMED_KERNEL` | unset | `libplumedKernel.so`. A value already in the environment is kept. |

If none of the PLUMED variables is set, `$HOME/opt/plumed` is used only when
that directory exists. A cluster module that already set `PLUMED_KERNEL` or
`LD_LIBRARY_PATH` is left as it is. `plumed/examples/check_env.sh` prints
what was resolved.

PLUMED compiled into SPARC molecular dynamics is separate: `src/plumed`,
enabled with `USE_PLUMED=1` and `PLUMED_FLAG`. That path does not use this
directory.
