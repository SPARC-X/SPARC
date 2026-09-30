# Coupling SPARC to i-PI through the socket API

This tutorial is for someone who already has (or can build) **SPARC**
(Simulation Package for Ab-initio Real-space Calculations) and wants
**i-PI** (a Python interface for *ab initio* path integral molecular
dynamics) to integrate the nuclei while SPARC only evaluates **DFT**
(density functional theory) energy, forces, and (when needed) the virial.

You do **not** write a new protocol and you do **not** turn on SPARC’s own
**MD** (molecular dynamics). You call SPARC’s existing i-PI socket **API**
(application programming interface).

## Quick start

i-PI moves the ions. SPARC only returns the DFT energy, forces, and, when NPT needs it, the virial. Build SPARC with `USE_SOCKET=1`. In `Al.inpt` keep `MD_FLAG: 0`.

`i-pi` must already be on `PATH`. Start it first, so the port exists, then start SPARC. From the repository root, the 5-step FCC aluminium example is:

```bash
cd utils/interface/ipi/examples/Al_FCC
cp ../../../../../psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8 .

# Terminal A
PYTHONUNBUFFERED=1 i-pi input.xml 2>&1 | tee i-pi.log

# Terminal B, after i-PI is listening on port 31415
./B_run_sparc.sh
```

`simulation.out` should gain one row per step, and `Al.out` should show SCF cycles. On Slurm set `MPIEXEC="srun -n"`. `SPARC` and `NP` override the binary and the rank count (`NP` defaults to 4). NVT, NPT, umbrella sampling, and PIMD use the same SPARC command; only the i-PI XML changes. Those files are in the sections below.

PLUMED without i-PI is [`../plumed/tutorial.md`](../plumed/tutorial.md).

Official references:

- i-PI: <https://docs.ipi-code.org/>
- SPARC-X-API (optional Python wrapper): <https://sparc-x.github.io/SPARC-X-API/>
- Socket protocol: <https://docs.ipi-code.org/distributed.html>

Worked examples use a small **FCC** (face-centered cubic) aluminium cell and
cover the **NVE** (microcanonical; constant number, volume, and energy),
**NVT** (canonical; constant number, volume, and temperature), and **NPT**
(isothermal–isobaric; constant number, pressure, and temperature) ensembles,
plus **PLUMED** (PLUgin for MolEcular Dynamics) through i-PI `ffplumed`,
including umbrella sampling.

Companion tutorial for calling PLUMED without i-PI (Python API **or** SPARC
built-in): [`../plumed/tutorial.md`](../plumed/tutorial.md).

| Path | What it is |
|------|------------|
| `utils/interface/plumed/` | Python driver (`plumed.Plumed()`) |
| `utils/interface/ipi/examples/Al_FCC/` | Minimal 5-step NVE (FCC Al) |
| `utils/interface/ipi/examples/tests/` | Validation suite (NVE/NVT/NPT, PLUMED, multi-client, restart) |
| `utils/interface/plumed/examples/` | Part I examples (`../plumed/tutorial.md`) |
| `examples/plumed_builtin/` | PLUMED inside SPARC MD (`../plumed/tutorial.md` Part II) |

---

## 1. Who does what

```
i-PI  (MD server)                         SPARC  (DFT client, USE_SOCKET=1)
  thermostats, barostats, PIMD, …            SCF only  (MD_FLAG: 0)
        positions, cell      ------------>
        energy, forces, virial  <---------
```

i-PI can run **PIMD** (path integral molecular dynamics) as well as classical
MD. SPARC’s job in this setup is a single **SCF** (self-consistent field)
evaluation per force request.

On the wire everything is **atomic units** (Bohr, Hartree, Hartree/Bohr).
i-PI converts your **XML** (Extensible Markup Language) / xyz units; SPARC’s
`.inpt` / `.ion` stay in SPARC conventions (Bohr cell, fractional or
Cartesian ions).

This is the opposite of SPARC–PLUMED linked into the binary, where SPARC
drives MD (`MD_FLAG: 1`) and PLUMED is an in-process library. That workflow,
and the Python-API alternative, are in [`../plumed/tutorial.md`](../plumed/tutorial.md).

| | SPARC + i-PI (this tutorial) | SPARC + PLUMED (see `../plumed/tutorial.md`) |
|--|--|--|
| Who moves the ions | **i-PI** | SPARC (built-in) or Python (API) |
| SPARC’s job | Socket force client | MD driver + DFT, or socket client |
| How you “call the API” | `sparc -socket host:port` | `PLUMED_FLAG` / `plumed.Plumed()` |
| Typical use | PIMD, NPT, multi-client, `ffplumed` | Enhanced sampling without i-PI |

The same SPARC binary can have `USE_SOCKET=1` and `USE_PLUMED=1`. Runtime
usage still differs: for i-PI you leave `MD_FLAG: 0` and connect as a client.

---

## 2. The API you actually call

There is one wire protocol (i-PI `STATUS` / `POSDATA` / `GETFORCE` / `EXIT`).
You reach it in three ways; **for SPARC + i-PI use path A**.

### A. SPARC command-line socket client (required for i-PI)

```bash
mpirun -np 4 /path/to/lib/sparc -socket localhost:31415 -name Al
```

Equivalent tags in the `.inpt` (optional if you pass `-socket`):

```text
SOCKET_FLAG: 1
SOCKET_HOST: localhost
SOCKET_PORT: 31415
SOCKET_INET: 1
```

`SOCKET_INET: 1` selects an **INET** (Internet, TCP/IP) socket. A **UNIX**
domain socket (a socket file on the local filesystem) is also supported:

```bash
mpirun -np 4 /path/to/lib/sparc -socket /tmp/sparc.socket:unix -name Al
```

`-name Al` is the SPARC job stem (`Al.inpt`, `Al.ion`, `Al.out`, …).

### B. i-PI `ffsocket` (the MD-side API)

i-PI opens the port. `ffsocket` is i-PI’s **force-field socket** client
driver. The forcefield **name** in XML must match the `<force>` block that
uses it:

```xml
<ffsocket mode='inet' name='sparc'>
  <address>localhost</address>
  <port>31415</port>
  <latency>0.01</latency>
  <timeout>600</timeout>
</ffsocket>

<forces>
  <force forcefield='sparc'></force>
</forces>
```

### C. Python, same protocol, no i-PI process

If you want SPARC forces from Python **without** starting the `i-pi`
executable, use the small server in
`utils/interface/plumed/driver/sparc_socket.py`. That is the same socket
API with Python as the server instead of i-PI. See
[§8](#8-optional-python-driver-same-socket-not-i-pi) and
[`../plumed/tutorial.md`](../plumed/tutorial.md) Part I.

[SPARC-X-API](https://sparc-x.github.io/SPARC-X-API/advanced_socket.html)
(`use_socket=True`) is the upstream **ASE** (Atomic Simulation Environment)
wrapper around the same idea.

---

## 3. Build SPARC with the socket API

Linux or **WSL** (Windows Subsystem for Linux). **MPI** (Message Passing
Interface; `mpicc`, `mpirun`) and **BLAS**/**LAPACK** (Basic Linear Algebra
Subprograms / Linear Algebra PACKage) as in the SPARC `makefile`.

```bash
cd src
grep -E '^USE_SOCKET|^USE_PLUMED' makefile
```

Set:

```text
USE_SOCKET    = 1
USE_PLUMED    = 1   # optional; keep 1 if you also use PLUMED / i-PI ffplumed
```

If `USE_PLUMED = 1`, set `PLUMED_ROOT` to the prefix that contains
`lib/plumed/src/lib/Plumed.inc`. The makefile default is `$HOME/opt/plumed`.

```bash
export PLUMED_ROOT=/path/to/plumed
export LD_LIBRARY_PATH="$PLUMED_ROOT/lib:${LD_LIBRARY_PATH:-}"
make clean
make -j$(nproc) PLUMED_ROOT="$PLUMED_ROOT"
```

Socket-only:

```bash
make clean
make -j$(nproc) USE_PLUMED=0
```

The binary is `lib/sparc`. Check that it links PLUMED when you asked for it:

```bash
ldd ../lib/sparc | grep -i plumed
```

You do **not** need to edit `src/socket/` for a standard client.

---

## 4. Install i-PI

```bash
pip install -U ipi
i-pi --help          # or: python -m ipi --help
```

Any environment is fine as long as `i-pi` and `python3` are on `PATH`. The example scripts do not activate one.

For i-PI `ffplumed` (the **force-field PLUMED** driver: **umbrella sampling**,
or **MTS**, multiple time stepping, with a cheap inner force), also install
Python PLUMED and point at the kernel:

```bash
# Point at the PLUMED you loaded, or omit both if $HOME/opt/plumed exists.
export PLUMED_ROOT=/path/to/plumed
export PLUMED_KERNEL="$PLUMED_ROOT/lib/libplumedKernel.so"
python -c "import plumed; print(plumed)"
```

---

## 5. SPARC input: electronic structure only

Nuclear dynamics live in i-PI. In the SPARC `.inpt`:

```text
MD_FLAG: 0
RELAX_FLAG: 0
CALC_STRESS: 1          # needed if i-PI will use the virial (NPT)
PRINT_ATOMS: 1
PRINT_FORCES: 1
```

Keep mesh, **XC** (exchange–correlation), k-points, smearing, and the
pseudopotential as you would for a single-point. Do **not** set `PLUMED_FLAG`
for this workflow.

The cell in SPARC Bohr must match the i-PI xyz cell (Å in the example below:
4.05 Å ≈ 7.653391 Bohr).

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

i-PI **overwrites** ion positions and the cell from the socket every step.
The `.ion` geometry is only the SPARC startup structure; it should still be
consistent with `init.xyz`.

---

## 6. i-PI input: `input.xml` + `init.xyz`

You do **not** need to write the XML from scratch. Copy
`utils/interface/ipi/examples/Al_FCC/input.xml` (NVE) or a file from
`utils/interface/ipi/examples/tests/` and change only the fields in the table below.

i-PI uses nested XML tags, not SPARC’s `KEY: value` lines. Angle brackets
mark blocks (`<dynamics>…</dynamics>`). Curly braces after a quantity set
**print or input units**, for example `time{picosecond}` or
`units='kelvin'`. The `[ step, time{…}, … ]` list is i-PI’s output-column
syntax, not a programming array.

What the blocks mean, and what a new user typically edits:

| Block | Role | Usually change? |
|-------|------|-----------------|
| `<output>` | What to write (`simulation.out`, xyz) | Optional: `stride`, extra properties |
| `<total_steps>` | Number of MD steps | Yes |
| `<ffsocket>` | How i-PI waits for SPARC | Yes: `<port>` must match `-socket` |
| `<initialize>` | Starting xyz and velocities | Yes: `init.xyz`; keep `nbeads='1'` unless PIMD |
| `<forces>` | Which forcefield computes DFT | Keep `forcefield='sparc'` (same as `ffsocket` **name**) |
| `<dynamics>` | Ensemble and timestep | Yes: `nve` / `nvt` / `npt`, `<timestep>` |
| `<ensemble>` | Target T (and P for NPT) | Yes |

Full tag list: [i-PI input tags](https://docs.ipi-code.org/input-tags.html).

`init.xyz` (Å; i-PI reads units from the comment):

```text
4
# CELL(abcABC): 4.050000 4.050000 4.050000 90.0 90.0 90.0 cell{angstrom} positions{angstrom}
Al    0.0000000000    0.0000000000    0.0000000000
Al    2.0250000000    2.0250000000    0.0000000000
Al    2.0250000000    0.0000000000    2.0250000000
Al    0.0000000000    2.0250000000    2.0250000000
```

Copied NVE template (`utils/interface/ipi/examples/Al_FCC/input.xml`):

```xml
<simulation verbosity='high'>
  <output prefix='simulation'>
    <properties stride='1' filename='out'>
      [ step, time{picosecond}, conserved, temperature{kelvin}, potential{electronvolt} ]
    </properties>
    <trajectory filename='pos' stride='1' format='xyz'> positions </trajectory>
  </output>

  <total_steps>5</total_steps>

  <ffsocket mode='inet' name='sparc'>
    <address>localhost</address>
    <port>31415</port>
    <latency>0.01</latency>
    <timeout>600</timeout>
  </ffsocket>

  <system>
    <initialize nbeads='1'>
      <file mode='xyz'> init.xyz </file>
      <velocities mode='thermal' units='kelvin'> 300 </velocities>
    </initialize>
    <forces>
      <force forcefield='sparc'> </force>
    </forces>
    <motion mode='dynamics'>
      <dynamics mode='nve'>
        <timestep units='femtosecond'> 1.0 </timestep>
      </dynamics>
    </motion>
    <ensemble>
      <temperature units='kelvin'> 300 </temperature>
    </ensemble>
  </system>
</simulation>
```

Ready-made variants (copy the XML; the SPARC client command stays the same):

| Want | Copy this file |
|------|----------------|
| NVE | `utils/interface/ipi/examples/Al_FCC/input.xml` |
| NVT (Langevin) | `utils/interface/ipi/examples/tests/b2_nvt/input.xml` |
| NPT (isotropic barostat) | `utils/interface/ipi/examples/tests/b3_npt/input.xml` |
| Umbrella sampling | `utils/interface/ipi/examples/tests/c1_umbrella/input_biased.xml` |
| PIMD, two beads | `utils/interface/ipi/examples/tests/e1_multiclient/input.xml` |

NPT also needs `CALC_STRESS: 1` in the SPARC `.inpt`. PIMD with `nbeads='2'`
needs **two** SPARC clients on the same port. Official walkthrough:
[i-PI tutorials](https://docs.ipi-code.org/tutorials.html).

---

## 7. How to launch

**Always start i-PI first** so the port exists, then start SPARC.

Two terminals, from `utils/interface/ipi/examples/Al_FCC/` after copying the psp:

```bash
# i-pi and python3 must already be on PATH
cp ../../../../../psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8 .

# Terminal A — MD server (unbuffered so the banner appears immediately)
PYTHONUNBUFFERED=1 i-pi input.xml 2>&1 | tee i-pi.log

# Terminal B — DFT client, after i-PI is listening on 31415
# MPIEXEC="srun -n" on Slurm. Default is "mpirun -np".
./B_run_sparc.sh
```

One-shot pattern used by the tests (`utils/interface/ipi/examples/tests/_common/run_md.sh`):

1. `i-pi input.xml &`
2. Poll `127.0.0.1:PORT` until it accepts a **TCP** (Transmission Control
   Protocol) connection.
3. `mpirun -np $NP $SPARC -socket localhost:$PORT -name Al`

The port in XML, in `-socket`, and in any firewall rules must match. Kill a
leftover `i-pi` if bind fails.

### Success checks

- `i-pi.log` shows a handshake and then steps, then a clean finish
- `simulation.out` gains one row per MD step
- SPARC `Al.out` / `Al.static` show SCF cycles
- Both processes exit after `total_steps`

For NVT, NPT, or PIMD, copy the matching XML from the table in §6. The SPARC
`-socket` command does not change (except the port, and a second client for
PIMD).

---

## 8. Optional: Python driver (same socket, not i-PI)

To request SPARC forces from Python, this repo provides a server that
speaks the same protocol (`utils/interface/plumed/driver/`):

```python
from driver import SparcSocketSession, load_al_fcc

atoms = load_al_fcc("init.xyz")          # positions Å, cell Å
with SparcSocketSession(workdir, port=31415, np=4) as sparc:
    energy_eV, forces_eV_A = sparc.get_energy_forces(atoms)
```

`SparcSocketSession` binds the port, launches

```text
mpirun -np N  $SPARC  -socket localhost:PORT  -name Al
```

and converts Bohr/Hartree on the wire to eV/Å for the caller.

That path is **SPARC + Python**, not SPARC + the `i-pi` program. Use it for
custom integrators or PLUMED-from-Python
([`../plumed/tutorial.md`](../plumed/tutorial.md) Part I,
`utils/interface/plumed/examples/`). Use §7 when you want i-PI’s MD (thermostats,
PIMD, `ffplumed`, restart).

---

## 9. Optional: PLUMED through i-PI (`ffplumed`) — umbrella sampling

Bias and **CVs** (collective variables) can sit in i-PI instead of inside
SPARC. SPARC stays a pure DFT client.

The worked example is **umbrella sampling**: a harmonic restraint on the Al–Al
distance, compared with an unbiased run. See
`utils/interface/ipi/examples/tests/c1_umbrella/`.

```xml
<ffsocket mode='inet' name='sparc'> ... </ffsocket>
<ffplumed name='plumed'>
  <file mode='xyz'> init.xyz </file>
  <plumed_dat> plumed.dat </plumed_dat>
</ffplumed>
<forces>
  <force forcefield='sparc'></force>
  <force forcefield='plumed'></force>
</forces>
```

`plumed.dat` for one umbrella window (Å, eV, ps; `RESTRAINT` is the harmonic
umbrella on the CV):

```text
UNITS LENGTH=A ENERGY=eV TIME=ps
d: DISTANCE ATOMS=1,2
RESTRAINT ARG=d AT=3.20 KAPPA=200.0
PRINT ARG=d FILE=COLVAR STRIDE=1
```

Set `PLUMED_KERNEL` as in §4. Multiple time stepping (cheap PLUMED inner
loop, SPARC outer loop) is `utils/interface/ipi/examples/tests/d1_mts/`.

---

## 10. Validation suite

From `utils/interface/ipi/examples/tests/`, with `i-pi` on `PATH`:

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/a1_energy_force
./run_all.sh
```

| ID | What it checks |
|----|----------------|
| A1 | **DFT energy and forces**: i-PI socket vs independent SPARC single-points |
| A2 | **Length units on the wire**: Å ↔ Bohr |
| A3 | **Clean shutdown**: i-PI `EXIT` |
| B1 | **Microcanonical (NVE) dynamics**: conserved energy at 0.5 / 1 / 2 fs |
| B2 | **Canonical (NVT) dynamics**: Langevin thermostat near 300 K |
| B3 | **Isothermal–isobaric (NPT) dynamics**: isotropic barostat and SPARC virial |
| C1 | **Umbrella sampling**: `ffplumed` harmonic restraint vs unbiased CV |
| D1 | **Multiple time stepping (MTS)**: PLUMED inner loop / SPARC outer loop |
| E1 | **Path-integral MD (PIMD)**: two beads, two SPARC clients (`nbeads=2`) |
| E2 | **Atomic geometry**: i-PI xyz vs SPARC socket coordinates |
| E3 | **MD restart**: continue from i-PI `RESTART` |

If SPARC finds an existing `Al.static`, it writes `Al.static_01` (then `_02`,
…). The test parsers pick the **newest** `Al.static*`. Cleaning old SPARC
outputs before a rerun avoids confusion.

---

## 11. Units and known caveats

- Wire: Bohr, Hartree, Hartree/Bohr, virial in Hartree.
- SPARC `CELL` is Bohr; i-PI xyz in the examples is Å.
- For NPT, SPARC must return a virial (`CALC_STRESS: 1`). i-PI
  `pressure_md` matches SPARC `pres = -trace(stress)/3` (`W = -σV` in
  `stress_to_virial`). The printed stress tensor has the other sign.
- 4-atom cells have huge temperature noise; short NVT tests use a wide
  window around the target T.

---

## 12. Checklist for a new system

1. Socket-enabled SPARC (`USE_SOCKET=1` → `lib/sparc`).
2. i-PI installed; port free.
3. SPARC `.inpt`: DFT settings, `MD_FLAG: 0`, `RELAX_FLAG: 0`.
4. Matching geometry: SPARC cell (Bohr) ↔ i-PI `init.xyz` (declared units).
5. i-PI `input.xml`: `ffsocket` name = `<force forcefield='…'>`, same port.
6. Start i-PI, then `mpirun … sparc -socket localhost:PORT -name JOB`.
7. Confirm `simulation.out` and SPARC SCF output, then lengthen `total_steps`.
