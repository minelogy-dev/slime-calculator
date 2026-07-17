# Slime Calculator

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
./build/slime_main <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <output.csv>
```

Scans the area `(startX, startZ)`–`(endX, endZ)` for all rectangles of size `sizeX` × `sizeZ` containing more than `threshold` slime chunks. Coordinates are in chunk coordinates (integers). Both `sizeX` and `sizeZ` must be ≤ 32. Dynamically tiles the search along Z to fit GPU memory.

The four vertices of the specified rectangle are strictly guaranteed to be within the search range. However, the actual region used internally is rounded up to multiples of 256 for GPU memory alignment, which may slightly expand the scanned area beyond the specified bounds.

### Performance

On an RTX 4060 Ti 16 GB, the following command takes ~5 minutes:

```bash
./build/slime_main 114514 -1875000 -1875000 1875000 1875000 17 17 55 output.csv
```

Output (world/block coordinates):

```
x,z,slime_count
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
