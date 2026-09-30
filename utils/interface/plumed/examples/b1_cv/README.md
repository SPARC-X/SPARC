# B1 — Unbiased distance CV through the PLUMED API

Short Langevin NVT. SPARC supplies DFT forces over the socket.
PLUMED only monitors the Al(1)–Al(2) distance (FCC nearest neighbour
≈ 2.86 Å). No restraint.

## Pass

- COLVAR has one row per MD evaluation
- Mean \(d_{12}\) is closer to 2.86 Å than to 3.20 Å

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/plumed/examples/b1_cv
./run_all.sh
```

Port: `32421`.
