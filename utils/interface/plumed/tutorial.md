# Coupling SPARC to PLUMED: API vs built-in

This tutorial is for someone who already has (or can build) **SPARC**
(Simulation Package for Ab-initio Real-space Calculations) and wants
**PLUMED** (PLUgin for MolEcular Dynamics) to evaluate **CVs** (collective
variables) and optional bias forces.

There are **two independent ways** to do that. They share a `plumed.dat` and
the same 4-atom **FCC** (face-centered cubic) aluminium cell. They do **not**
share who integrates the nuclei.

## Quick start

**Part I — Python calls PLUMED.** SPARC is a socket force client: `USE_SOCKET=1`, `MD_FLAG: 0`, and no `PLUMED_FLAG`. Do not start the `i-pi` program. `python3`, `numpy`, `py-plumed`, and an MPI launcher must be on `PATH`. From the repository root:

```bash
cd utils/interface/plumed/examples
./check_env.sh
./run_all.sh
```

A0 is one DFT single point (energy near −254 eV). A1 checks that PLUMED `ENERGY` matches SPARC. B1 and B2 are short trajectories with a distance collective variable and an umbrella bias. Set `PLUMED_ROOT` or `PLUMED_KERNEL` if PLUMED is not in `$HOME/opt/plumed`. `SPARC`, `NP` (default 4), and `MPIEXEC` (`srun -n` on Slurm) override the SPARC launch.

**Part II — SPARC calls PLUMED.** Rebuild with `USE_PLUMED=1`. `PLUMED_ROOT` must contain `lib/plumed/src/lib/Plumed.inc`. In the `.inpt` set `MD_FLAG: 1`, `PLUMED_FLAG: 1`, and `PLUMED_FILE`. Then one command, with no `-socket`:

```bash
mpirun -np 4 /path/to/lib/sparc -name Al
```

`Al.out` should show `PLUMED_FLAG: 1`, and a `COLVAR` file should appear. Rank 0 is the only rank that calls PLUMED.

i-PI can also host PLUMED. That path is [`../ipi/tutorial.md`](../ipi/tutorial.md).

| | Part I — API | Part II — built-in |
|--|--|--|
| Who moves the ions | Python driver (Velocity Verlet / Langevin) | SPARC (`MD_FLAG: 1`) |
| SPARC’s job | Socket **DFT** (density functional theory) client | MD driver + DFT |
| How PLUMED is called | Python `plumed.Plumed()` | SPARC C API (`plumed_sparc.c`) |
| SPARC rebuild | Not required (`USE_SOCKET=1` is enough) | Required (`USE_PLUMED=1`) |
| SPARC input | `MD_FLAG: 0`, no `PLUMED_FLAG` | `MD_FLAG: 1`, `PLUMED_FLAG: 1` |
| Examples | `utils/interface/plumed/examples/` | `examples/plumed_builtin/` |

i-PI (a Python interface for *ab initio* path integral molecular dynamics)
can also host PLUMED (`ffplumed`). That is a third path and is documented in
[`../ipi/tutorial.md`](../ipi/tutorial.md), not here.

Official references:

- PLUMED: <https://www.plumed.org/>
- i-PI socket protocol (Part I only): <https://docs.ipi-code.org/distributed.html>
- SPARC-X-API (optional Python wrapper): <https://sparc-x.github.io/SPARC-X-API/>

---

## 0. Choose a path

Pick **Part I** if you already have the i-PI socket binary and do not want to
link PLUMED into SPARC. Pick **Part II** if SPARC should own the whole MD
run from `mpirun … sparc -name Al` with no extra process.

The rest of this file is two self-contained tutorials. Units, `plumed.dat`
pitfalls, and the FCC Al cell are repeated where they matter so you can
follow only one part.

---

# Part I — API: Python driver + SPARC socket

You do **not** write a new protocol and you do **not** turn on SPARC’s own
MD. You call SPARC’s existing i-PI socket **API** (application programming
interface). PLUMED runs in the same Python process that integrates the ions.

---

## I.1. Who does what

On the wire everything is **atomic units** (Bohr, Hartree, Hartree/Bohr).
The Python driver converts to **eV / Å / fs** for MD and for PLUMED
(`setMD*Units`). SPARC’s `.inpt` / `.ion` stay in SPARC conventions (Bohr
cell, fractional or Cartesian ions).

```bash
mpirun -np 4 /path/to/lib/sparc -socket localhost:32410 -name Al
```

The Python side binds that port first (it is the server), then launches the
`mpirun` line above. You do not start the `i-pi` executable.

---

## I.2. Build SPARC (socket only)

Linux or **WSL** (Windows Subsystem for Linux). **MPI** (Message Passing
Interface; `mpicc`, `mpirun`) and **BLAS**/**LAPACK** as in the SPARC
`makefile`.

Part I does **not** need `USE_PLUMED=1`. The binary used for i-PI is enough:

```bash
cd src
grep -E '^USE_SOCKET|^USE_PLUMED' makefile
make -j$(nproc) USE_SOCKET=1
```

The binary is `lib/sparc`. You do **not** need to edit `src/socket/` or
`src/plumed/`.

A binary that also has `USE_PLUMED=1` still works for Part I, as long as the
`.inpt` keeps `MD_FLAG: 0` and omits `PLUMED_FLAG`.

---

## I.3. Python environment

The examples need `numpy`, `py-plumed`, and an MPI launcher on `PATH`.
**ASE** (Atomic Simulation Environment) is not required. They do not
activate a named environment.

```bash
# Point at the PLUMED you loaded, or omit both if $HOME/opt/plumed exists.
export PLUMED_ROOT=/path/to/plumed
export PLUMED_KERNEL="$PLUMED_ROOT/lib/libplumedKernel.so"
python -c "import numpy, plumed; print('ok')"
```

`PLUMED_KERNEL` is the PLUMED runtime library that `py-plumed` loads. It is
independent of whether SPARC itself was linked with PLUMED.

---

## I.4. SPARC input: electronic structure only

Nuclear dynamics live in Python. In the SPARC `.inpt`:

```text
MD_FLAG: 0
RELAX_FLAG: 0
CALC_STRESS: 1
PRINT_ATOMS: 1
PRINT_FORCES: 1
```

Keep mesh, **XC** (exchange–correlation), k-points, smearing, and the
pseudopotential as you would for a single-point. Do **not** set
`PLUMED_FLAG`.

The cell in SPARC Bohr must match the Python xyz cell (Å in the example
below: 4.05 Å ≈ 7.653391 Bohr).

`.ion` example (fractional), plus the **ONCV** (Optimized Norm-Conserving
Vanderbilt) **psp** (pseudopotential) copied into the run directory:

```text
ATOM_TYPE: Al
N_TYPE_ATOM: 4
COORD_FRAC:
  0.0  0.0  0.0
  0.5  0.5  0.0
  0.5  0.0  0.5
  0.0  0.5  0.5

PSEUDO_POT: 13_Al_3_1.9_1.9_pbe_n_v1.0.psp8
```

The driver overwrites ion positions and the cell from the socket every step.

---

## I.5. How the Python driver calls PLUMED

`utils/interface/plumed/driver/plumed_api.py` issues the same
`plumed_cmd` sequence as SPARC’s `src/plumed/plumed_sparc.c`, but with
**MD-side** units eV / Å / fs:

```text
setRealPrecision, setMDEnergyUnits, setMDLengthUnits, setMDTimeUnits,
setNatoms, setPlumedDat, setTimestep, setKbT, init
```

Then each step:

```text
setStep, setPositions, setMasses, setCharges, setBox, setEnergy, setForces,
setVirial, calc
```

The DFT energy and forces come from the socket. PLUMED’s force buffer is
**added** to the SPARC forces (after the buffer is zeroed). The driver then
takes a Velocity-Verlet or Langevin step.

A minimal call:

```python
from driver import PlumedHandle, SparcPlumedEngine, SparcSocketSession, load_al_fcc

atoms = load_al_fcc("init.xyz")          # positions Å, cell Å
with SparcSocketSession(workdir, port=32411, np=4) as sparc:
    with PlumedHandle(
        natoms=len(atoms.symbols),
        timestep_fs=1.0,
        plumed_dat="plumed.dat",
    ) as plumed:
        energy_eV, forces_eV_A = SparcPlumedEngine(sparc, plumed).evaluate(atoms)
```

`SparcSocketSession` binds the port, launches

```text
mpirun -np N  $SPARC  -socket localhost:PORT  -name Al
```

and converts Bohr/Hartree on the wire to eV/Å for the caller.

---

## I.6. `plumed.dat` (API units)

Literals and `PRINT` columns follow the `UNITS` line. The driver already
told PLUMED that MD arrays are eV / Å / fs via `setMD*Units`, so **do not**
convert coordinates yourself.

```text
UNITS LENGTH=A ENERGY=eV TIME=ps

d: DISTANCE ATOMS=1,2 NOPBC
r: RESTRAINT ARG=d AT=3.20 KAPPA=20.0
PRINT ARG=d,r.bias FILE=COLVAR STRIDE=1
```

`NOPBC` matters on this 4-atom FCC cell: the nearest-neighbour distance is
≈ 2.86 Å while half the box is 2.025 Å, so a periodic minimum-image
`DISTANCE` wraps.

---

## I.7. How to launch (API tests)

**Always start the Python server first** (it binds the port), which then
starts SPARC. From the SPARC repo root:

```bash
# python3 and an MPI launcher must already be on PATH.
# Optional: SPARC, NP, MPIEXEC, PLUMED_ROOT, PLUMED_KERNEL
cd utils/interface/plumed/examples
./check_env.sh
./run_all.sh
```

One test:

```bash
cd utils/interface/plumed/examples/a0_socket_smoke
./run_all.sh
```

Ports start at 32410 so they do not collide with the i-PI suite (31415…).
`NP` (default 4), `SPARC`, `PORT`, and `PLUMED_KERNEL` can be overridden.

### Success checks

- A0: finite energy (around −254 eV for this cell) and forces
- SPARC `Al.out` shows SCF cycles and `MD_FLAG: 0`
- A1 / B1 / B2 write `COLVAR` and a `*_table.txt` that prints `PASS`

---

## I.8. Validation suite (API)

| ID | Folder | What it checks |
|----|--------|----------------|
| A0 | `utils/interface/plumed/examples/a0_socket_smoke/` | Socket DFT works without i-PI or PLUMED |
| A1 | `utils/interface/plumed/examples/a1_energy/` | PLUMED `ENERGY` matches SPARC socket energy (eV) |
| B1 | `utils/interface/plumed/examples/b1_cv/` | Distance CV is printed every MD step, near the FCC nn 2.86 Å |
| B2 | `utils/interface/plumed/examples/b2_umbrella/` | Harmonic restraint shifts Al–Al distance vs unbiased |

B2 uses zero-velocity Verlet (not Langevin) so thermal noise does not hide
the bias. `KAPPA=20` eV/Å² is gentler than the i-PI C1 value (200) so a 1 fs
step stays stable on this 4-atom cell.

---

# Part II — Built-in: SPARC MD + `PLUMED_FLAG`

SPARC drives MD. PLUMED is compiled into the SPARC binary and called from
`src/plumed/plumed_sparc.c` every ionic step. There is no socket and no
Python driver.

---

## II.1. Who does what

Launch is a single command:

```bash
mpirun -np 4 /path/to/lib/sparc -name Al
```

No `-socket`. Rank 0 owns the PLUMED object (`MPI_COMM_SELF`); other ranks
do not call `plumed_cmd`.

---

## II.2. Build SPARC with PLUMED

```bash
cd src
grep -E '^USE_SOCKET|^USE_PLUMED' makefile
```

Set:

```text
USE_SOCKET    = 1   # optional for this path; keep 1 if you also run i-PI
USE_PLUMED    = 1
```

Set `PLUMED_ROOT` to the prefix that contains
`lib/plumed/src/lib/Plumed.inc`. The makefile default is `$HOME/opt/plumed`.

```bash
export PLUMED_ROOT=/path/to/plumed
export LD_LIBRARY_PATH="$PLUMED_ROOT/lib:${LD_LIBRARY_PATH:-}"
make clean
make -j$(nproc) PLUMED_ROOT="$PLUMED_ROOT"
```

The binary is `lib/sparc`. Confirm the link:

```bash
ldd ../lib/sparc | grep -i plumed
```

If `USE_PLUMED=0`, `PLUMED_FLAG: 1` is rejected at input time.

---

## II.3. SPARC input: MD + PLUMED tags

Nuclear dynamics live in SPARC. In the `.inpt`:

```text
MD_FLAG: 1
MD_METHOD: NVT_NH
MD_TIMESTEP: 1.0
MD_NSTEP: 5
ION_TEMP: 300.0
ION_TEMP_END: 300.0

PLUMED_FLAG: 1
PLUMED_FILE: plumed.dat
```

`PLUMED_FLAG: 1` requires `MD_FLAG: 1` and a readable `PLUMED_FILE`. Mesh,
XC, k-points, smearing, and the psp are the same as Part I. Do **not** pass
`-socket`.

`.ion` is the same 4-atom FCC Al cell as Part I (7.653391 Bohr).

---

## II.4. How SPARC calls PLUMED

`src/plumed/plumed_sparc.c` (`Plumed_Init`, `Plumed_Evaluate`) talks to
PLUMED in **SPARC MD units**: Hartree, Bohr, **atu** (atomic time unit).
`setMD*Units` converts those into PLUMED internals (kJ/mol, nm, ps):

```text
setMDEnergyUnits   Hartree -> kJ/mol
setMDLengthUnits   Bohr    -> nm
setMDTimeUnits     atu     -> ps
```

Each MD step SPARC passes positions, masses, charges, box, potential, and
the force array; PLUMED adds bias forces in place; SPARC then integrates.

You do not call `plumed.Plumed()` from Python on this path.

---

## II.5. `plumed.dat` (built-in units)

The `UNITS` line still controls **literals and PRINT columns in the file**.
It does **not** double-convert MD coordinates (those already went through
`setMD*Units`). The examples use the same Å / eV numbers as Part I so the
two tutorials can be compared:

```text
UNITS LENGTH=A ENERGY=eV TIME=ps

d: DISTANCE ATOMS=1,2 NOPBC
r: RESTRAINT ARG=d AT=3.20 KAPPA=20.0
PRINT ARG=d,r.bias FILE=COLVAR STRIDE=1
```

`NOPBC` is required for the same reason as Part I (FCC nn > L/2).

Older in-source experiments used `UNITS LENGTH=nm ENERGY=kj/mol`. That is
also valid; then `AT` and `KAPPA` must be written in nm and kJ/mol.

---

## II.6. How to launch (built-in tests)

From the SPARC repo root, with `PLUMED_ROOT` set to the prefix SPARC was
linked against:

```bash
export PLUMED_ROOT=/path/to/plumed
export LD_LIBRARY_PATH="$PLUMED_ROOT/lib:${LD_LIBRARY_PATH:-}"
cd examples/plumed_builtin
./run_all.sh
```

One test:

```bash
cd examples/plumed_builtin/a1_energy
./run_all.sh
```

Each `run_all.sh` copies `_common/Al.inpt` and `Al.ion`, copies the psp from
`psps/`, and runs `mpirun -np $NP $SPARC -name Al`.

### Success checks

- `Al.out` contains `PLUMED_FLAG: 1` and `MD_FLAG: 1`
- A `COLVAR` (or `colvar`) file appears with one row per printed stride
- SPARC also writes `Al.plumed` (PLUMED log) and `Al.aimd`

---

## II.7. Validation suite (built-in)

| ID | Folder | What it checks |
|----|--------|----------------|
| A1 | `a1_energy/` | SPARC MD with PLUMED `ENERGY` printed |
| B1 | `b1_cv/` | Distance CV printed every MD step |
| B2 | `b2_umbrella/` | Unbiased vs `RESTRAINT` on the same CV |

These runs are 5 NVT steps: enough to prove the in-process hook is on. A
clear umbrella shift of the CV needs a longer trajectory (see the API B2
test, 15+15 Verlet steps, or lengthen `MD_NSTEP`).

---

## 3. What you need to watch

**Unit conversion is automatic.** Keep SPARC `.inpt` / `.ion` in SPARC
conventions (`CELL` in Bohr). The socket driver (Part I) and
`setMD*Units` (both parts) convert MD arrays into PLUMED internals. Do
**not** convert coordinates or energies yourself before calling PLUMED.

What you **do** write:

- Numbers in `plumed.dat` in the units of that file’s `UNITS` line. The
  examples use `UNITS LENGTH=A ENERGY=eV TIME=ps`, so `AT=3.20` is 3.20 Å.
  Matching SPARC Bohr by hand would double-convert.
- On this 4-atom FCC Al cell, nearest neighbour ≈ 2.86 Å and half the box
  is 2.025 Å. Use `DISTANCE … NOPBC` unless you enlarge the cell; otherwise
  the periodic minimum image wraps.
- Short NVT tests only check the coupling. A 4-atom cell has large
  temperature noise and is not a thermodynamics benchmark.

---

## 4. Checklist for a new system

### API (Part I)

1. Socket-enabled SPARC (`USE_SOCKET=1` → `lib/sparc`).
2. `numpy` + `py-plumed`; `PLUMED_KERNEL` set; port free.
3. SPARC `.inpt`: DFT settings, `MD_FLAG: 0`, no `PLUMED_FLAG`.
4. Matching geometry: SPARC cell (Bohr) ↔ Python `init.xyz` (Å). The
   driver converts the rest.
5. `plumed.dat`: write literals in the `UNITS` of that file (examples: Å, eV).
6. Start the Python session; it launches `mpirun … sparc -socket …`.
7. Confirm `COLVAR` and SPARC SCF output, then lengthen the MD.

### Built-in (Part II)

1. SPARC built with `USE_PLUMED=1` → `lib/sparc`; `ldd` shows PLUMED.
2. `LD_LIBRARY_PATH` includes the PLUMED `lib` directory.
3. SPARC `.inpt`: DFT + MD settings, `PLUMED_FLAG: 1`, `PLUMED_FILE`.
4. `plumed.dat` next to the `.inpt`; write literals in that file’s `UNITS`.
5. `mpirun … sparc -name JOB` (no `-socket`).
6. Confirm `PLUMED_FLAG: 1` in `Al.out` and a `COLVAR`, then lengthen
   `MD_NSTEP`.
