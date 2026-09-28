# B1 — NVE conserved quantity

Short classical NVE on the same 4-atom Al cell, three timesteps.
i-PI prints `conserved`, `potential`, `kinetic_md`, `temperature` every step.

## Pass criteria

- Run completes for dt = 0.5, 1.0, 2.0 fs
- Conserved quantity does not explode (\(|\Delta E_{\mathrm{cons}}| < 0.05\) Ha over the run)
- Local run: 0.5–1.0 fs is usable; **2 fs heats the tiny cell strongly** (T ~ 1000 K).
  Prefer 0.5–1 fs for later tests.

This is a smoke check of the integrator + SPARC forces, not a production
timestep study. 25 steps per dt.

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/b1_nve
./run_all.sh
```

Port: `31431`.
