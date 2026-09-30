# A2 — Units sanity

i-PI uses **atomic units internally** and on the socket. File I/O units are
whatever you put on the XML tag or in the xyz comment.

This folder prints the same quantities in two units and checks that the
ratio matches `ipi.utils.units`. It also checks Å (init.xyz) ↔ Bohr
(SPARC `CELL` / i-PI `positions{atomic_unit}`).

## Conversion table (i-PI 3.3 factors)

| Quantity | Atomic unit | Common I/O | i-PI factor |
|----------|-------------|------------|-------------|
| Length | Bohr | Å | 1 Å = 1.8897261 Bohr |
| Energy | Hartree | eV | 1 Ha = 27.2113834 eV |
| Time | \(\hbar/E_h\) | fs, ps | 1 fs = 41.341373 atu |
| Temperature | Hartree / \(k_B\) | K | 1 K = 3.1668152×10⁻⁶ Ha |
| Pressure | Ha/Bohr³ | bar, GPa | 1 GPa = 3.398827×10⁻⁵ a.u. |

SPARC `driver.h` uses `HARTREE_TO_EV = 27.21138602`, which differs from i-PI
in the 8th significant figure (~10⁻⁷ relative). **A1 compared both sides in
Ha** and avoided this. If you convert i-PI `potential{electronvolt}` with
SPARC’s constant you pick up ~10⁻⁶ Ha of fake error (seen on the old
`ipi_Al_FCC` eV file).

If you omit `cell_units`, i-PI still writes `cell{atomic_unit}` even when
`positions{angstrom}`. Always read the xyz comment; do not assume the cell
and the coordinates share a unit.

## Pass criteria

- Dual-unit columns in `simulation.out` match i-PI’s conversion to ~10⁻¹⁰ relative
- `positions{angstrom}` × 1.8897261 = `positions{atomic_unit}` cell
- SPARC printed lattice (Bohr) matches the i-PI a.u. cell to ~10⁻⁵ Bohr
- 1 fs timestep → `time{picosecond}` increases by 0.001 each step

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/a2_units
./run_all.sh
```

Port: `31422`.
