# B2 — NVT temperature control

Same 4-atom Al cell. i-PI `dynamics mode='nvt'` with a Langevin thermostat
(`tau = 10 fs`). Timestep 0.5 fs (B1 showed 2 fs overheats this cell).
Target T = 300 K.

## Pass criteria

- Mean temperature after discarding the first 10 steps is within **±150 K** of 300 K
  (4 atoms: large fluctuations; this is a smoke demo)
- Conserved (extended) quantity does not explode (\(|\Delta| < 0.1\) Ha)
- Instantaneous T fluctuates (std > 1 K)

This is a smoke demo, not a converged canonical average.

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/ipi/examples/tests/b2_nvt
./run_all.sh
```

Port: `31432`.
