# 史莱姆区块计算器

## 1.实现思路

首先Minecraft使用Java编写，查看wiki可知，使用
```java
import java.util.Random;

public class CheckSlimeChunk {

    public static boolean isSlimeChunk(long worldSeed,     // 世界种子，一个64位整数，可以通过/seed获取
                                       int  chunkX,        // 区块X轴坐标，32位整数
                                       int  chunkZ) {      // 区块Z轴坐标，32位整数
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
来判定史莱姆区块，现在使用Cuda，需要在C++中复刻Java的Random与Minecraft的逻辑，也就是`slime_main.cu`中的
```cpp
__device__ __forceinline__ int isSlimeChunk(int64_t baseX, int64_t baseZ) {
    // 1. 计算初始状态
    uint64_t acc = (uint64_t)baseX + (uint64_t)baseZ;
    acc ^= 987234911ULL;
    uint64_t seed48 = (acc ^ 0x5DEECE66DULL) & 0xFFFFFFFFFFFFULL;

    const uint64_t MULT = 0x5DEECE66DULL;
    const uint64_t ADD = 0xBULL;
    const uint32_t REJECT = 2147483640;
    const uint32_t MAGIC = 0xCCCCCCCDULL;  // 用于快速除以10的魔数

    // 2. 第一次采样（绝大多数情况会命中这里）
    seed48 = (seed48 * MULT + ADD) & 0xFFFFFFFFFFFFULL;
    uint32_t u1 = (uint32_t)(seed48 >> 17);
    uint32_t q = (uint32_t)(((uint64_t)u1 * MAGIC) >> 35);
    uint32_t r1 = u1 - q * 10;

    // 3.第二次采样
    seed48 = (seed48 * MULT + ADD) & 0xFFFFFFFFFFFFULL;
    uint32_t u2 = (uint32_t)(seed48 >> 17);
    q = (uint32_t)(((uint64_t)u2 * MAGIC) >> 35);
    uint32_t r2 = u2 - q * 10;

    return (((u1 >= REJECT) ? r2 : r1) == 0) ? 1 : 0;
}
```
可以注意到传入的不是seed与chunkX与chunkZ，因为在大范围扫描中，chunkX与chunkZ的取值范围相对而言极小，所以在computeBases进行了预处理，并将种子集成在baseX中，通过一步加法算出`new Random`时的种子。

随后的内容用于实现`rng.nextInt(10)`的逻辑，在Java中有一个do-while的采样，但是分支会破坏GPU的流水线，利用`src/test.cpp`进行检查后发现最多需要一次重采样就可以拿到解，于是利用一次硬件选择指令代替if分支。

最初的版本使用二维前缀和加速取矩形选区，但是仍然会产生大量重复计算，同时二位前缀和并行难度相对较大，现版本已经修改为一维前缀和配合滑动窗口，提升了`computeRowPrefixSumTiledWide`的并行度，同时简化了总计算量，为了进一步提升并行，在X上也进行了分段，长度为256，由于Minecraft中几乎不可能出现连续在一条直线上的256个史莱姆区块，这里使用了uint8_t极大压缩了显存带宽需求（相比旧的uint32_t，带宽需求是四分之一）代价仅是在`slidingWindowOutputKernel`中引入了极少的分支结构，大部分被编译器压平成硬件选择，而且节约了75%的显存与带宽。具体汇编见`result.ncu-rep`
