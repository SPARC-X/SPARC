# A1 — Energy / force consistency

Compare what i-PI records from the SPARC socket against **independent**
SPARC single-points on the same geometries (no socket). That is the
strongest check that the driver protocol is passing energy and forces
correctly.

System: 4-atom FCC Al, same settings as `utils/interface/ipi/examples/Al_FCC/`.
Short NVE (5 steps) is only a way to generate a few distinct frames.

## Pass criteria

| Quantity | Tolerance |
|----------|-----------|
| \|E_iPI − E_SP\| | < 5×10⁻⁵ Ha |
| max \|F_iPI − F_SP\| | < 1×10⁻⁴ Ha/Bohr |

`TOL_SCF` is 10⁻⁵, so those numbers are “same SCF, same geometry”,
not a physics accuracy claim vs a plane-wave code.

A three-way table is printed when the socket `Al.static` is present:

1. **i-PI** — `simulation.out` potential and `simulation.frc_0.xyz`
2. **SPARC socket** — energies/forces SPARC printed during the i-PI run
3. **SPARC SP** — fresh `mpirun … sparc` (no `-socket`) on each frame

If 1 ≈ 2 but 1 ≠ 3, the protocol is fine and the single-point inputs differ.
If 1 ≠ 2, units or the socket payload are wrong.

i-PI step `n` is SPARC “socket step `n+1`” (SPARC labels from 1).

## How to run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/a1_energy_force
cp ../../../../../../psps/13_Al_3_1.9_1.9_pbe_n_v1.0.psp8 .
```

**Option 1 — one script** (starts i-PI, waits for the port, then SPARC):

```bash
./run_all.sh
```

**Option 2 — two terminals**, same pattern as `ipi_Al_FCC`:

```bash
# Terminal A
./A_run_ipi.sh

# Terminal B, after i-PI is listening on 31415
./B_run_sparc.sh
```

Then independent single-points and the comparison:

```bash
python3 C_run_singlepoints.py
python3 D_compare.py
```

`D_compare.py` exits 0 on pass and 1 on fail. It writes `compare_table.txt`.

## Files

| File | Role |
|------|------|
| `input.xml` | i-PI NVE; prints potential in Ha and a force trajectory |
| `Al.inpt` / `Al.ion` | SPARC client: DFT only, `MD_FLAG: 0` |
| `init.xyz` | 4.05 Å FCC Al |
| `A_run_ipi.sh` / `B_run_sparc.sh` | two-process launch |
| `C_run_singlepoints.py` | write `sp_frameXX/` and run native SPARC |
| `D_compare.py` | parse outputs, print table, apply tolerances |
| `run_all.sh` | A → B → C → D |

## Notes

- Port `31415` must be free. Kill a leftover `i-pi` if bind fails.
- SPARC may print `Getting an unknown message from server` on i-PI `EXIT`.
  Ignore that for this test; it is item A3.
- Units: everything compared here is Ha and Ha/Bohr. eV conversion is A2.

## Example PASS (local run)

```
max |E_ipi - E_sp|          = 6.96e-08 Ha
max |F_ipi - F_sp|          = 4.04e-06 Ha/Bohr
max |E_ipi - E_socket|      = 4.71e-09 Ha
energy vs independent SPARC SP : PASS
force  vs independent SPARC SP : PASS
```

i-PI and SPARC-socket print the same energy to ~10⁻⁹ Ha. Independent SPARC single-points
match within SCF noise. Exact numbers change with the random NVE velocities.

