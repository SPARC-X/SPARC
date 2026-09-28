# C1 / C2 — Umbrella restraint via i-PI `ffplumed`

Physical forces: SPARC `ffsocket`. Bias: PLUMED harmonic restraint on the
Al(1)–Al(2) distance (FCC nearest neighbour ≈ 2.86 Å). SPARC does **not**
use `PLUMED_FLAG`.

Two short runs from rest (40 steps, 1 fs) with a Langevin thermostat at
1 K (`tau = 2` fs). Zero initial velocity and strong damping keep thermal
noise from hiding the restraint, and stop the biased bond from ringing
through the window. `DISTANCE … NOPBC` is required: the neighbour length
2.86 Å is larger than half the 4.05 Å box. `KAPPA=20` eV/Å² (same as the
other PLUMED paths) stays stable at 1 fs; 200 was too stiff on this cell.

| Run | Bias | Expected CV |
|-----|------|-------------|
| unbiased | none | stays at ~2.86 Å (DFT forces ≈ 0 by symmetry) |
| biased | `RESTRAINT AT=3.20` | \(d_{12}\) increases toward 3.20 Å |

CV is measured from the i-PI xyz (does not rely on PLUMED `COLVAR`, which
may not flush for a constant umbrella unless a `<metad>` SMotion is present).

## Pass criteria

- Both runs complete
- Unbiased mean \(d_{12}\) is closer to 2.86 Å than to 3.20 Å
- Biased mean \(d_{12}\) is larger than the unbiased mean (restraint pulls out)

Short trajectories only need a clear CV shift to prove the bias force is on.
Two-window WHAM (C3) is skipped.

## Run

Needs `PLUMED_KERNEL` (taken from the environment, or from `PLUMED_ROOT`)
and the Python `plumed` module on `PYTHONPATH`.

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/c1_umbrella
./run_all.sh
```

Port: `31441`.
