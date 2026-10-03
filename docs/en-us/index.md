# Slime Chunk Calculator

Enumerates, over a batch of candidate positions, every position whose
`sizeX x sizeZ` chunk window contains at least `threshold` slime chunks.
Coordinates are block coordinates (`x` is the chunk x multiplied by 16);
the output is a CSV with the columns `x,z,slime_count`.

## 1. The rule

Minecraft is written in Java, and the wiki gives the check as:

```java
import java.util.Random;

public class CheckSlimeChunk {

    public static boolean isSlimeChunk(long worldSeed,     // world seed, a 64-bit integer, obtainable via /seed
                                       int  chunkX,        // chunk X coordinate, 32-bit integer
                                       int  chunkZ) {      // chunk Z coordinate, 32-bit integer
        Random rng = new Random(
            worldSeed +
            (long)(chunkX * chunkX * 4987142) +
            (long)(chunkX * 5947611) +
            (long)(chunkZ * chunkZ) * 4392871L +
            (long)(chunkZ * 389711) ^ 987234911L
        );
        return rng.nextInt(10) == 0;
    }
}
```

The CUDA side has to replicate `java.util.Random` and that expression in C++. The implementation
lives in `src/slime_hash.cuh` and is shared by `slime_main` and `slime_circle` (the cross-check
tool `tools/oracle.c` is written independently from the Java specification and deliberately
shares no code with them):

```cpp
// src/slime_hash.cuh -- bit-for-bit equivalent to the Java above; hot path 18 -> 16 SASS instructions
__device__ __forceinline__ bool isSlimeChunk(int64_t baseX, int64_t baseZ) {
    constexpr uint32_t L2   = 0xE66D0000u;  // low 32 bits of (MULT<<16); the 16 trailing zeros are what makes dirty bits harmless
    constexpr uint32_t H2   = 0x0005DEECu;  // (MULT<<16) >> 32
    constexpr uint32_t ADD2 = 0x000B0000u;  // 0xB << 16
    constexpr uint32_t C_LO = 0xE434E432u;  // low 32 bits of C = 987234911 ^ 0x5DEECE66D
    constexpr uint32_t C_HI = 0x00000005u;
    constexpr uint32_t REJECT = 2147483640U;
    constexpr uint32_t MAGIC = 0x4CCCCCCDU; // 10*MAGIC == 2 (mod 2^32)
    constexpr uint32_t DIV10_MAX = 0x19999999U;

    uint64_t s = (uint64_t)baseX + (uint64_t)baseZ;
    uint32_t X_lo = (uint32_t)s ^ C_LO;
    uint32_t X_hi = (uint32_t)(s >> 32) ^ C_HI;
    // 48x35 multiply: the low half goes to a 64-bit multiply, the high half plus the cross term
    // goes to the high 32 bits; 0xB<<16 and the carry are folded into one 64-bit addend.
    uint32_t z = X_hi * L2 + X_lo * H2;
    uint64_t Q = (uint64_t)X_lo * L2 + (((uint64_t)z << 32) | (uint64_t)ADD2);
    uint32_t V = (uint32_t)(Q >> 32);
    uint32_t bits = V >> 1;                 // = W >> 17

    if (__builtin_expect(bits < REJECT, 1)) {
        return (bits * MAGIC) <= DIV10_MAX; // one IMAD + one ISETP, no rotate needed
    }
    // Java's rejection path (probability about 3.7e-9): advance the state once and re-sample,
    // at most once.
    uint32_t s1_lo = __funnelshift_r((uint32_t)Q, V, 16);
    uint32_t s1_hi = (V >> 16) & 0xFFFFu;
    uint32_t lo = s1_lo * L2, hi = __umulhi(s1_lo, L2);
    uint32_t z2 = s1_hi * L2 + s1_lo * H2;
    uint32_t lo2 = lo + ADD2;
    uint32_t V2 = hi + z2 + (uint32_t)(lo2 < ADD2);
    return ((V2 >> 1) * MAGIC) <= DIV10_MAX;
}
```

The parameters are not `seed`, `chunkX` and `chunkZ`: the seed is a constant term and is folded
into the x part, so `computeBases` precomputes `baseX = slimeQuadX(chunkX) + seed` and
`baseZ = slimeQuadZ(chunkZ)`; each candidate then needs a single addition to obtain the seed of
its `new Random`.

There are five transformations relative to the Java form, each with its derivation and exhaustive
evidence in the comments of `src/slime_hash.cuh`: the two XORs are merged into one constant, the
48-bit mask disappears entirely (the 16 trailing zeros of `L2` absorb the dirty bits), `+0xB` is
folded into the addend of an IMAD.WIDE, `bits % 10 == 0` becomes one 32-bit multiply and compare
(`0x4CCCCCCD`, verified exhaustively over all 2^31 values, mismatch = 0), and the rejection path
(probability about 3.7e-9) is written as a real branch rather than being predicated. The result of
the check is one 32-bit bitmap per warp per row (`__ballot_sync`); everything downstream works on
those bitmaps.

## 2. Counting the hits in a window

Window counting uses a fused bitmap scan, and the X extent of a window never touches memory: one
warp handles one row and packs 32 checks into a 32-bit bitmap, the X window sum comes from a
single `__funnelshift_r` that takes a `sizeX`-wide bit field out of the 64 bits formed by two
adjacent words, and `__popc` counts it; rolling along Z adds the popcount of the new row and
subtracts the popcount of the row `sizeZ` rows back. The bitmap ring lives in shared memory
(`RING = 64` slots, which must exceed the maximum `sizeZ` of 32), so a block covers `sizeX-1` more
input columns than it emits candidates (the halo travels through the ring and registers, and a
window never crosses a block). The older implementation first wrote a per-row prefix sum to memory
and then looked up each candidate, costing one byte written and one byte read per candidate
centre; with bitmaps that DRAM traffic is zero.

The per-column bookkeeping along X can be amortised further: each lane owns **four adjacent
candidate columns**, and the hot path maintains only one bundle-level loose upper bound `U` (the
number of slime chunks inside the union of `sizeZ` rows and `sizeX+3` columns). When `U` is below
`threshold`, none of the four candidates can hit, so the per-column rolling
`SHF+LOP3+POPC+LDS.U8+STS.U8` is spread over four candidates; a bundle that does pass the gate
recomputes the exact value on demand from the shared row-bitmap ring. The union bit field is
`sizeX+3` wide and has to fit inside two adjacent words, which is why `sizeX <= 29` takes this
path while `30..32` uses the counting ring.

For `sizeX > 32` the window is too wide to fit into two adjacent words, so the code falls back to
two stages: `wideRowPrefixSumKernel` builds a prefix sum per row (one warp cooperatively scans a
whole row) and `wideWindowScoreKernel` then reads two table entries per candidate to get the
interval sum. The prefix sums are stored as `uint8_t` and are allowed to wrap, so a row needs no
segmentation at all: each warp accumulates from the start of the row with `col += WARP_SIZE` all
the way to the end, and overflow simply overflows. The consumer only ever subtracts the two window
endpoints as `(uint8_t)(a - b)`, i.e. modulo 256 -- a window spans exactly `sizeX` columns within
one row, so as long as `sizeX < 256` the difference is always exact, whether the wrap happens
before the window's left endpoint or inside it (for example `(uint8_t)(1 - 255) == 2`). This relies
on the C guarantee that unsigned arithmetic is reduced modulo 2^N, handled for free by 8-bit
truncation with no explicit modulo or carry fixup; `wideWindowScoreKernel` therefore has no "which
segment does the window cross" test and no two-halves stitching, and the inner loop is a single
byte subtraction (the only exception is the leftmost column, which has no left endpoint and is
handled by a mask computed once before the row loop). The limit `sizeX <= 255` is validated at
startup in `main`.

## 3. Path selection and runtime adaptivity

The path is chosen from `sizeX`, `sizeZ` and `K`, and all four paths are bit-for-bit equivalent:

| Path | Taken when | What the shared ring holds |
|---|---|---|
| `scanRectFusedMerge4Kernel<K>` | `K != 1` and `sizeX <= 29` | one bitmap per row; X bookkeeping via the 4-column union bound |
| `scanRectFusedKernelCount<K>` | `sizeX` in `[30,32]` and `sizeZ` within the runtime limit | the `sizeX` window count of every candidate column (`uint8`) |
| `scanRectFusedKernelBallot<K>` | `K == 1`, or `sizeZ` above that limit | one bitmap per row |
| `wideRowPrefixSumKernel` + `wideWindowScoreKernel` | `sizeX > 32` | no shared ring (prefix sums) |

The shared footprint of the counting ring grows linearly with `sizeZ`, and past some `sizeZ` the
number of resident blocks per SM collapses, so that path's `sizeZ` limit is derived by the program
from the real occupancy crossover of the **current device** instead of being hard-coded. In the
same way, `K = 4` (four strided columns per lane) is only worth using when there are enough X tiles,
or when the tile count times the number of Z sub-blocks is enough to fill the GPU; both thresholds
are computed after `cudaSetDevice` from that card's SM count and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`. The derived values can be printed with
`SLIME_PRINT_TUNING=1`, and `SLIME_NO_AUTOTUNE=1` falls back to the compile-time constants. The
device code embeds both SASS and PTX for the local compute capability, so a card with a newer
architecture is handled by the driver's JIT.

## 4. No duplicates, no omissions, no out-of-bounds writes

* Every candidate position is covered exactly once by the `(block, warp, k, lane, row)` partition.
* A result slot is claimed with `atomicAdd` before the bounds check, and `pos` increases
  monotonically, so every slot in `[0, cap)` is written exactly once and an out-of-bounds write is
  structurally impossible.
* When the result buffer fills up, the block stops immediately, the host discards the whole round
  and retries with a smaller chunk, so a discarded hit never turns into a missing row.

The formal proof (14 lemmas covering the partitioning and the window invariants of both compute
kernels) is in `docs/zh-cn/correctness-proof.md`, and the external cross-checks are described in
`docs/zh-cn/correctness.md`.

## 5. Building and running

```bash
./build.sh
./build/slime_main <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <out.csv> [--sort=on|off]
```

`--sort=on` (the default) produces a CSV ordered by `(x,z)`, which makes byte-for-byte comparison
possible; `--sort=off` skips the sort. The scan range is rounded up to a multiple of 256 columns
and padded along Z, and the program prints the range it actually used as `Global range:`; the
requested range is always contained in it. A progress bar appears when stderr is a terminal and can
be forced on or off with `SLIME_PROGRESS=1` / `SLIME_NO_PROGRESS=1`. Hard limits, runtime knobs and
the remaining tools are documented in `docs/zh-cn/tools.md` and
`docs/zh-cn/optimization-design.md`.
