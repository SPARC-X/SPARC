# B2 — Umbrella restraint via the PLUMED API (not SPARC `PLUMED_FLAG`)

Physical forces: SPARC socket. Bias: PLUMED harmonic restraint on the
Al(1)–Al(2) distance (`AT=3.20`, `KAPPA=20` eV/Å²). Same CV as
`utils/interface/ipi/examples/tests/c1_umbrella`, but i-PI is not used and SPARC
has no `PLUMED_FLAG`. `KAPPA` is gentler than the i-PI C1 value (200)
so a 1 fs Velocity-Verlet step stays stable on this 4-atom cell.

Two short Velocity-Verlet runs with **zero initial velocity** (so the
restraint is not hidden by Langevin noise). Same DFT client settings:

| Run | Bias | Expected CV |
|-----|------|-------------|
| unbiased | none | stays at ~2.86 Å (DFT forces ~ 0) |
| biased | `RESTRAINT AT=3.20` | \(d_{12}\) increases toward 3.20 Å |

Short trajectories only need a small shift to prove the bias force is on.

## Pass

- Both runs complete
- Unbiased mean \(d_{12}\) closer to 2.86 Å than to 3.20 Å
- Biased mean \(d_{12}\) larger than the unbiased mean

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/plumed/examples/b2_umbrella
./run_all.sh
```

Ports: `32431` (unbiased), `32432` (biased).
