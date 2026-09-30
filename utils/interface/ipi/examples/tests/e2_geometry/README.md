# E2 — Geometry / cell communication

Uses existing A1 outputs: i-PI `positions{atomic_unit}` vs SPARC
`Al.static` fractional coordinates × lattice (Bohr).

No extra DFT if `a1_energy_force` has already been run.

## Pass criteria

- Max Cartesian difference (after wrapping) < 10⁻⁴ Bohr on sampled frames
- Cell lengths match to 10⁻⁵ Bohr
