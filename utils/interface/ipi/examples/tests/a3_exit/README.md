# A3 — Clean shutdown (`EXIT`)

When i-PI finishes it sends a 12-byte `EXIT` header. SPARC should close
without treating that as an error.

## What happens with the current binary (Aug 3 2026)

`sparc.log` ends with:

```
Getting an unknown message from server, exiting...
```

and `mpirun` returns 1. Physics is already finished; this is shutdown noise.

## Source vs binary

`src/socket/driver.c` **already maps `EXIT` → `IPI_MSG_EXIT`** and prints
`Socket server requested EXIT; closing SPARC client.` then `break`s.

`strings lib/sparc` does **not** contain that message — only the old
“unknown message” string. The running binary is older than the source.

## This test (no source edit)

1. Rebuild SPARC from the existing tree (`make`, no C changes).
2. Run a 2-step NVE.
3. Pass if `sparc.log` contains `Socket server requested EXIT` and SPARC
   exits 0 (or at least does not print `unknown message`).

If a rebuild is not possible in this environment, skip and rebuild later.
Do **not** patch `driver.c` again unless a rebuild still fails.

Port: `31423`.
