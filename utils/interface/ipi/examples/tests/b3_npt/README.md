# B3 — Isotropic NPT (SPARC virial)

i-PI `dynamics mode='npt'` + isotropic barostat. SPARC returns the virial
because `CALC_STRESS: 1`.

This DFT Al cell (coarse mesh) prints a stress tensor of about +10 GPa at
4.05 Å. SPARC's scalar pressure is `pres = -trace(stress)/3`, about −10 GPa.
i-PI defines `pressure_md = trace(virial)/(3V)` at zero kinetic stress, so
`stress_to_virial` keeps `W = -σV`. Those two pressures agree at t=0; the
printed stress tensor is the other sign on purpose.

## Pass criteria

- Run completes; `volume` stays finite and positive
- Volume **fluctuates** (barostat is using the SPARC virial)
- Temperature stays finite
- SPARC `.static` still prints `Stress (GPa)` each socket step
- i-PI `pressure_md` at step 0 matches SPARC `pres` within 2 GPa

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/b3_npt
./run_all.sh
```

Port: `31433`.
