# B3 — Isotropic NPT (SPARC virial)

i-PI `dynamics mode='npt'` + isotropic barostat. SPARC returns the virial
because `CALC_STRESS: 1`.

This DFT Al cell (coarse mesh) prints ~+10 GPa stress in SPARC `.static` at
4.05 Å. i-PI `pressure_md` starts near **−10 GPa** (opposite sign). The
barostat still moves the cell. A directional “raise P → shrink V” check is
**deferred** until the virial sign in `driver.c` `stress_to_virial` is
reviewed (no source change in this test).

## Pass criteria

- Run completes; `volume` stays finite and positive
- Volume **fluctuates** (barostat is using the SPARC virial)
- Temperature stays finite
- SPARC `.static` still prints `Stress (GPa)` each socket step

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/b3_npt
./run_all.sh
```

Port: `31433`.
