# experiments/

Where `scripts/fly.py run` writes by default when no `--out` is given, and the
default `out_root` of `vssim.bench.run.start_run`. Each run is a dated directory
`YYYY-MM-DD_<name>/` carrying its **provenance in git** — `run.json` (git hash,
seed, machine, GPU, the numeric environment), `resolved_config.yaml` (the config
exactly as the code saw it), `MD5SUMS` (a checksum of every file the run wrote) —
and its **payloads out of it** (`baseline_check.jsonl`, `telemetry.jsonl.gz`; see
`.gitignore`). A run refuses to start on a dirty working tree unless told otherwise,
and the runs already here never count as dirt (`dirty_paths(ignore_prefix=)`).
