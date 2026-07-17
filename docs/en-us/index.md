# Slime Chunk Calculator

## 1. Implementation Approach

Minecraft is written in Java. According to the wiki, the slime chunk check uses:

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

This project uses CUDA, so Java's `Random` and Minecraft's logic must be replicated in C++, as implemented in `slime_main.cu`:

```cpp
__device__ __forceinline__ int isSlimeChunk(int64_t baseX, int64_t baseZ) {
    // 1. Compute initial state
    uint64_t acc = (uint64_t)baseX + (uint64_t)baseZ;
    acc ^= 987234911ULL;
    uint64_t seed48 = (acc ^ 0x5DEECE66DULL) & 0xFFFFFFFFFFFFULL;

    const uint64_t MULT = 0x5DEECE66DULL;
    const uint64_t ADD = 0xBULL;
    const uint32_t REJECT = 2147483640;
    const uint32_t MAGIC = 0xCCCCCCCDULL;  // magic number for fast division by 10

    // 2. First sample (hits in the vast majority of cases)
    seed48 = (seed48 * MULT + ADD) & 0xFFFFFFFFFFFFULL;
    uint32_t u1 = (uint32_t)(seed48 >> 17);
    uint32_t q = (uint32_t)(((uint64_t)u1 * MAGIC) >> 35);
    uint32_t r1 = u1 - q * 10;

    // 3. Second sample (fallback)
    seed48 = (seed48 * MULT + ADD) & 0xFFFFFFFFFFFFULL;
    uint32_t u2 = (uint32_t)(seed48 >> 17);
    q = (uint32_t)(((uint64_t)u2 * MAGIC) >> 35);
    uint32_t r2 = u2 - q * 10;

    return (((u1 >= REJECT) ? r2 : r1) == 0) ? 1 : 0;
}
```

Note that the function does not take separate seed, chunkX, and chunkZ parameters. Since chunkX and chunkZ vary over a relatively small range during large-area scans, `computeBases` pre-processes them and integrates the seed into baseX, computing the `new Random` seed with a single addition.

The remainder of the kernel implements `rng.nextInt(10)`. Java uses a do-while rejection loop, but branches hurt GPU pipeline performance. Analysis with `src/test.cpp` shows that at most one resample is ever needed, so an if-based approach (using a hardware select instruction) replaces the branch.

The initial version used a 2D prefix sum to accelerate rectangle queries, but this still involved significant redundant computation, and parallelizing a 2D prefix sum is relatively difficult. The current version switches to a 1D prefix sum combined with a sliding window, improving parallelism in `computeRowPrefixSumTiledWide` and reducing total computation. To further increase parallelism, the X dimension is also split into tiles of length 256. Since it is nearly impossible to have 256 consecutive slime chunks in a straight line in Minecraft, `uint8_t` is used to drastically reduce VRAM bandwidth (one quarter of the old `uint32_t` bandwidth) at the cost of minimal branching in `slidingWindowOutputKernel`, most of which is flattened into hardware selects by the compiler. This also saves 75% of VRAM and bandwidth. See `result.ncu-rep` for the detailed assembly.
