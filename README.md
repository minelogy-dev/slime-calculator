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

Scans the area `(startX, startZ)`–`(endX, endZ)` for all rectangles of size `sizeX` × `sizeZ` containing more than `threshold` slime chunks. Coordinates are in chunk coordinates (integers). Dynamically tiles the search along Z to fit GPU memory.

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

## Pipeline

```
slime_main  <seed> <area> <rect> <threshold>  → candidates.csv
slime_circle candidates.csv <radius> <rect> <seed> <threshold>  → circles.csv
slime_cmp   circles.csv <distance> <threshold>  → result.csv
```
