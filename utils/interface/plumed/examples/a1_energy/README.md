# A1 — PLUMED ENERGY vs SPARC socket energy

Python driver:
1. SPARC socket returns DFT energy (eV)
2. PLUMED is initialized with the same structure and `ENERGY`
3. Printed COLVAR energy must match SPARC after unit conversion

This is the API analogue of the in-source PLUMED energy overlay
(report_plumed Figure 1), without linking PLUMED into SPARC.

## Pass

| Quantity | Tolerance |
|----------|-----------|
| max \|E_PLUMED − E_SPARC\| | < 1×10⁻⁴ eV |

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/plumed/examples/a1_energy
./run_all.sh
```

Port: `32411`.
