# A0 — Socket smoke test (no PLUMED, no i-PI)

One single-point on 4-atom FCC Al through the Python i-PI socket server.
SPARC is the already-compiled socket client. This is the same DFT path
i-PI uses, without starting i-PI.

## Pass

- Finite energy and forces
- SPARC log contains a completed SCF (or `Socket server requested EXIT`)

## Run

```bash
# i-pi and python3 must already be on PATH
cd utils/interface/plumed/examples/a0_socket_smoke
./run_all.sh
```

Port: `32410`.
