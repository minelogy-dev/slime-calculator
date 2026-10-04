# Slime Calculator

**English** | [简体中文](README.zh.md)

Minecraft slime chunk finder with CUDA acceleration. Supports Java Edition only.

## Quick Start

```bash
git clone <repo_url>
cd slime-calculator
./build.sh
./build/slime_main 12345 -10 -10 10 10 5 5 3 output.csv
```

This scans seed `12345` from chunk `(-10,-10)` to `(10,10)` for 5×5 chunk areas with ≥3 slime chunks.

## Build

```bash
./build.sh
```

Outputs three binaries to `build/`.

## Tools

### slime_main

Primary rectangular-area slime chunk scanner with CUDA.

```bash
./build/slime_main <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <output.csv> [--sort=on|off]
```

Scans the area `(startX, startZ)`–`(endX, endZ)` for all rectangles of size `sizeX` × `sizeZ` containing at least `threshold` slime chunks. Coordinates are in chunk coordinates (integers). `sizeX` must be ≤ 255 and `sizeZ` ≤ 32: these are hard limits derived from the `uint8_t` prefix sum and the sliding-window ring buffer, and the program validates them at startup. `--sort` (default `on`) makes the program emit rows in `(x,z)` order. The result pool allocates one big buffer with a single `malloc` (50% of available RAM, capped at 8 GB, floor 4 blocks) and never frees or reallocates it until the run is over. A pool of sorter threads (up to 10) turns each full block into a sorted run **while the GPU is still scanning** — the sort is CPU/GPU-parallel and the runs are independent, so it scales across cores. When the big buffer is exhausted the sorted runs spill to `<output>.runN` and the block is recycled, so memory stays bounded; at the end one k-way merge of the in-memory and on-disk runs writes the CSV (those run files are removed). `--sort=off` skips the sorting entirely and streams the CSV while scanning, which is useful for very long runs and when you intend to sort externally. Cost: about +1.4% wall time on the largest benchmark (4.35M rows) — only the final merge and the (now serial) CSV write remain, the sorting itself is hidden under GPU time — and it removes the external `LC_ALL=C sort` pass from test loops entirely.

Coordinates are also limited to |chunk coord| ≤ 2,000,000 (far inside the vanilla world border of ±1,874,999 and far below `INT_MAX`): that bound is what makes the result-buffer geometry provably safe (see the design notes). The fallback path (`sizeX > 32`) still tiles Z to fit GPU memory; the fused path keeps the whole Z range in one block on a single device and instead streams Z in adaptive chunks.

The four vertices of the specified rectangle are strictly guaranteed to be within the search range. However, the actual region used internally is rounded up to multiples of 256 for GPU memory alignment, which may slightly expand the scanned area beyond the specified bounds.

### Performance

On an RTX 4060 Ti 16 GB in the standard test state — GPU starting at 35 °C, SM 2340 MHz, memory
9001 MHz locked — the following command takes **37.1 s** measured over three repetitions
(the fastest run; 35.6 s of that is GPU time), producing 40,700,773 rows:

```bash
./build/slime_main 114514 -1875000 -1875000 1875000 1875000 17 17 55 output.csv
```

Absolute seconds depend on the host and on the machine state (clock lock, start temperature), so
quote them together with the host they came from. The state-independent metric is `inst/candidate`
(dynamic instructions divided by candidate count); see `docs/zh-cn/optimization-design.md` for the
current value and for the design that produces it.

The locked numbers above are the conservative, reproducible ones. On a native Linux host that lets
the card boost instead of holding it at the test clock, the same binary does this:

![quickbench.sh --full -n 1: every workload det=ok, and the full-map run at 419.7 B/s end to end](docs/img/fullmap-benchmark.jpg)

*`quickbench.sh --full -n 1` on a headless bare-metal host with the same card and driver
595.91.07, clocks unlocked: the single full-map run checks 1.40625e13 candidate centres in
**33.36 s of GPU time / 33.50 s wall = 419.7 B/s end to end**, and every workload reports `det=ok`.
The program's own report for that run: 74,117 hits over 58 chunks, 33.48 s GPU / 33.66 s wall.

The absolute seconds depend on the host: an unlocked headless machine, a graphical desktop and a
virtualised environment all give different numbers for the same binary, so seconds are only
meaningful together with the host they were measured on. Within one workload, B/s and the
`det=ok` flag are the parts worth comparing.*

Output (world/block coordinates):

```
x,z,slime_count
```

### Benchmark

```bash
./bench.sh --quick     # two small workloads
./bench.sh             # quick + X-dominant + Z-dominant
./bench.sh --full      # also the 3.75M x 3.75M workload above (~37 s per run at the test clock)
```

Three small helpers cover measuring and self-checking. `gpuinfo.sh` prints the GPU state;
`quickbench.sh` runs a speed table over several workload shapes;
`quickcheck.sh` checks the output against an independent Java-spec oracle in about ten seconds.
They read live values at runtime rather than baking in this host's clocks, so they are part of the
repository and work on any machine with a CUDA GPU.

The scripts that *are* machine-specific — GPU arbitration, start-temperature alignment and the
cooldown convention — stay in the working tree and out of the repository (see `.gitignore`). Every
script that would call them checks for their presence and reports the stage as skipped instead of
failing silently.

`tools/collect.sh` gathers a complete performance and correctness snapshot (GPU state, full-map
timing, ncu metrics and the raw report, the bit-exactness matrix, the JVM cross-check, the oracle
check) into `.work/collect/<timestamp>/` in one command and tars it up next to the directory
(which is kept); `SUMMARY.md` inside records what each file is and how each stage ended.

![gpuinfo.sh reporting NOT READY while a benchmark is in flight, next to nvidia-smi](docs/img/gpu-state-check.jpg)

*`gpuinfo.sh` run while a benchmark is in flight: 59 °C, 2745 MHz SM, 8751 MHz memory, 163.51 W
of a 165 W cap, 100% utilization — and it correctly calls that **NOT READY** for the standard test
state (35 °C or cooler, 2340 MHz SM, 9001 MHz memory). `nvidia-smi` below it shows the same card
from the driver's side, including the `slime_main` process holding 946 MiB. Absolute times are
only comparable between runs that started in that state.*

Each workload is warmed up once, then measured `--reps` times (default 3); the table reports min/median/mean wall time, the summed per-block GPU time, the hit count, throughput in **B/s** (billions of candidate centres checked per second, where a candidate centre is a window top-left position inside the requested region; `cells_M / ms == B/s`, `B/s_min` uses the fastest run and `B/s_avg` the mean), and a `det` column that compares the md5 of the output CSV sorted with `LC_ALL=C sort` (the output order is decided by `atomicAdd` and is not stable, and the sort must use the C locale: under e.g. zh_CN.UTF-8 the `,`/`-` characters are ignored at the primary collation level, which makes `sort` lose its total order and produced false DIFF reports). `--save results.csv` records every run and `--baseline results.csv` prints the change against a previous run, which is the intended way to measure an optimization. A `det=DIFF` row means the output content changed for identical input, so the speed number is meaningless. The fallback path (`sizeX > 32`) sizes `H_max` from free VRAM (`d_row_ps = width * H_max`) and the scanned Z range expands with it, so runs made under different free-memory states are not strictly comparable there; the fused path (`sizeX <= 32`) allocates no `d_row_ps`, sets `H_max` to the whole Z range on a single device and therefore scans exactly the requested region, and its result buffer is sized from free VRAM (`cap`), which only changes how often a full buffer is drained, not the output; B/s also depends strongly on workload shape, so compare it only within one workload. Results are handed to a writer thread through a hand-written pinned FIFO pool (`ResultPool`), so CSV formatting overlaps GPU compute; the summed GPU time therefore covers the scan and the device-to-host copy but *not* the formatting, which means `gpu_min`/`B/s_avg` are not directly comparable with releases from before that change — use wall-clock `B/s_min` when comparing across it.

Before a release, run `tools/releasecheck.sh`. It builds, then runs the correctness battery
(`tools/quickcheck.sh` against an independent Java-spec oracle, `tools/mccrosscheck.sh` against
the original Minecraft Java running on a real JVM, and `tools/verify.sh`), and writes
a report (by default `.work/release-verification.md`) recording the git commit, the source and
binary hashes, every stage's result, and
an explicit list of **what the report does not establish**. Add `--soak 6` to fold in the 5-8 hour
exhaustive sweep.

`slime_main.cu` has two diagnostic build switches: `-DSLIME_PROFILE` splits the row-prefix kernel from the sliding-window kernel using CUDA events and prints one `PROFILE: row-prefix kernel ... sliding-window + D2H/launch ...` line; `-DSLIME_SCAN_ONLY` keeps the hit test but performs no registration (no `atomicAdd`, no `d_results` stores, no D2H copies, no CSV rows), which isolates the cost of the result-registration path. The compile-time knobs `-DSLIME_Z_SUB_ROWS=<N>` and `-DSLIME_Z_SPLIT_MIN_TILES=<N>` control the fused kernel's Z split (used to force the un-split path in tests). The environment variable `SLIME_CAP_SLOTS=<slots>` overrides the result-buffer size (it is clamped to at least one full row of candidates); setting it small forces the "buffer full -> discard and retry with a smaller chunk" path on every chunk, which is how that path is tested. One further invariant: `cap >= rangeX` always holds, and because hits never exceed candidates, a chunk of `cap / rangeX` output rows can never overflow.

```bash
nvcc -o build/slime_main_prof src/slime_main.cu -O3 -use_fast_math -arch=sm_89 -DSLIME_PROFILE
nvcc -o build/slime_main_scan src/slime_main.cu -O3 -use_fast_math -arch=sm_89 -DSLIME_SCAN_ONLY
```

### slime_circle

Circle-based filter using `slime_main` output.

```bash
./build/slime_circle <input_csv> <radius> <sizeX> <sizeZ> <seed> <output_csv> <threshold>
```

Reads candidates from `slime_main` output and finds sub-rectangles whose circular region of `radius` contains more than `threshold` slime chunks.

### slime_cmp

Distance-based post-filter.

```bash
./build/slime_cmp <input_csv> <distance> <threshold> <output_csv>
```

Filters records whose chunk distance from `(0,0)` ≤ `distance` and slime count ≥ `threshold`.

Output:

```
x,z,slime_count,distance
```

### test

A validation tool that proves Java's `nextInt` rejection sampling requires at most one iteration. Enumerates all bad seeds (8 possible `u` values × 2¹⁷ low bits) and performs DFS on the seed graph to compute the longest chain of consecutive bad seeds. The result is 1, confirming that no bad seed produces another bad seed, so a `do-while` loop is unnecessary and can be safely replaced with an `if` in the CUDA kernel.

```bash
./build/test
```

## Pipeline

```
slime_main  <seed> <area> <rect> <threshold>  → candidates.csv
slime_circle candidates.csv <radius> <rect> <seed> <threshold>  → circles.csv
slime_cmp   circles.csv <distance> <threshold>  → result.csv
```
