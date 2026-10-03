# 史莱姆区块计算器

枚举一批候选位置，输出其中「`sizeX × sizeZ` 区块窗口内的史莱姆区块数 `>= threshold`」的全部
位置。坐标按方块计（`x` 为该区块 x 乘 16），输出 CSV：`x,z,slime_count`。

## 1. 判定规则

Minecraft 用 Java 编写，判定规则见 wiki：

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

CUDA 侧要在 C++ 中复刻 `java.util.Random` 与这个表达式，实现收在 `src/slime_hash.cuh`，
由 `slime_main` 与 `slime_circle` 共用（用于对拍的 `tools/oracle.c` 是从 Java 规范独立
重写的，刻意不与它们共享任何代码）：

```cpp
// src/slime_hash.cuh —— 与上面那段 Java 逐位等价，热路径 SASS 18 -> 16 条
__device__ __forceinline__ bool isSlimeChunk(int64_t baseX, int64_t baseZ) {
    constexpr uint32_t L2   = 0xE66D0000u;  // (MULT<<16) 的低 32 位，16 个尾零是「脏位免疫」的关键
    constexpr uint32_t H2   = 0x0005DEECu;  // (MULT<<16) >> 32
    constexpr uint32_t ADD2 = 0x000B0000u;  // 0xB << 16
    constexpr uint32_t C_LO = 0xE434E432u;  // C = 987234911 ^ 0x5DEECE66D 的低 32 位
    constexpr uint32_t C_HI = 0x00000005u;
    constexpr uint32_t REJECT = 2147483640U;
    constexpr uint32_t MAGIC = 0x4CCCCCCDU; // 10*MAGIC ≡ 2 (mod 2^32)
    constexpr uint32_t DIV10_MAX = 0x19999999U;

    uint64_t s = (uint64_t)baseX + (uint64_t)baseZ;
    uint32_t X_lo = (uint32_t)s ^ C_LO;
    uint32_t X_hi = (uint32_t)(s >> 32) ^ C_HI;
    // 48x35 乘法：低半给 64 位乘，高半 + 交叉项给高 32 位；0xB<<16 与进位由 64 位加数一次做完
    uint32_t z = X_hi * L2 + X_lo * H2;
    uint64_t Q = (uint64_t)X_lo * L2 + (((uint64_t)z << 32) | (uint64_t)ADD2);
    uint32_t V = (uint32_t)(Q >> 32);
    uint32_t bits = V >> 1;                 // = W >> 17

    if (__builtin_expect(bits < REJECT, 1)) {
        return (bits * MAGIC) <= DIV10_MAX; // 单条 IMAD + 单条 ISETP，无需循环右移
    }
    // Java 拒绝路径（概率约 3.7e-9）：推进一步状态后重采，至多一次
    uint32_t s1_lo = __funnelshift_r((uint32_t)Q, V, 16);
    uint32_t s1_hi = (V >> 16) & 0xFFFFu;
    uint32_t lo = s1_lo * L2, hi = __umulhi(s1_lo, L2);
    uint32_t z2 = s1_hi * L2 + s1_lo * H2;
    uint32_t lo2 = lo + ADD2;
    uint32_t V2 = hi + z2 + (uint32_t)(lo2 < ADD2);
    return ((V2 >> 1) * MAGIC) <= DIV10_MAX;
}
```

传入的不是 seed 与 chunkX/chunkZ：`seed` 是常数项，折进 x 部分；`computeBases` 预先算出
`baseX = slimeQuadX(chunkX) + seed` 与 `baseZ = slimeQuadZ(chunkZ)`，于是每个候选只需要
一次加法就能得到 `new Random` 的种子。

相对 Java 的写法有五处变形，逐条说明与穷举证据见 `src/slime_hash.cuh` 的注释：把两次 XOR
合成一个常量、48 位掩码整体消掉（靠 `L2` 的 16 个尾零吸收脏位）、把 `+0xB` 折进 IMAD.WIDE
的加数、把 `bits % 10 == 0` 换成一次 32 位乘加比较（`0x4CCCCCCD`，已对全部 2^31 个取值穷举
验证，mismatch = 0）、以及把概率约 3.7e-9 的拒绝路径写成真分支而不是谓词化。判定结果是每行
每 warp 一个 32 位位图（`__ballot_sync`），后续统计全部在位图上做。

## 2. 统计窗口命中数

窗口计数走**位图融合扫描**，候选的 X 窗口完全不落显存：一个 warp 处理一行，把 32 个判定
压成一个 32 位位图，X 方向的窗口和由 `__funnelshift_r` 从「相邻两个字拼成的 64 位」里
一次取出 `sizeX` 宽的位段再 `__popc` 得到；Z 方向滚动 = 加本行的 popcount、减 `sizeZ` 行
之前那一行的同一个 popcount。位图环放在 shared（`RING = 64` 槽，必须大于 `sizeZ` 上限 32），
所以每个 block 覆盖的输入列比它输出的候选列多 `sizeX-1` 个（halo 走环与寄存器，窗口永不跨
block）。旧实现是「每行先做前缀和写进显存、再对每个候选查表」，每个候选中心要写读各 1 字节；
改成位图后这部分 DRAM 流量降为 0。

X 方向的逐列记账还有更省的写法：每 lane 负责 **4 个相邻候选列**，热路径只维护一个
bundle 级松上界 `U`（`sizeZ` 行 × `sizeX+3` 列并集内的史莱姆数），`U < threshold` 时整组
4 个候选都不可能命中，于是逐列滚动的 `SHF+LOP3+POPC+LDS.U8+STS.U8` 被摊到 4 个候选上；
过闸的 bundle 再从 shared 行位图环按需重算精确值。并集位段宽 `sizeX+3` 要落在相邻两字里，
这就是 `sizeX <= 29` 走这条路径、`30..32` 另走计数环的原因。

`sizeX > 32` 时窗口宽到放不进相邻两字，退回两段式：`wideRowPrefixSumKernel` 先逐行做前缀和
（一行一个 warp 协作扫描），`wideWindowScoreKernel` 再对每个候选用两次查表求区间和。前缀和
以 `uint8_t` 存储、允许自然回绕，于是同一行完全不需要分段：每个 warp 从行首一路
`col += WARP_SIZE` 累加到底，溢出就让它溢出。消费者只对窗口两端做 `(uint8_t)(a - b)` 的
模 256 减法 —— 窗口在单行上的跨度恰为 `sizeX`，所以只要 `sizeX < 256`，这个差值就一定等于
真值，无论回绕发生在窗口左端点之前还是之内（例如 `(uint8_t)(1 - 255) == 2`）。这里依赖的是
C 对无符号整数「结果按 2^N 取模」的保证，由硬件的 8 bit 截断完成，不需要任何显式取模或进位
补偿；`wideWindowScoreKernel` 里因此既没有「窗口跨了哪一段」的判断，也没有两半拼接，循环内
只是一次字节减法（唯一例外是最左列没有左端点，用一个预先算好的掩码按 0 处理）。边界条件
`sizeX <= 255` 由 `main` 启动时校验。

## 3. 路径选择与运行期自适应

按 `sizeX` / `sizeZ` / `K` 选路，四条路径逐位等价：

| 路径 | 何时走 | shared 环里放什么 |
|---|---|---|
| `scanRectFusedMerge4Kernel<K>` | `K != 1` 且 `sizeX <= 29` | 每行的位图；X 向按 4 邻列共享松上界记账 |
| `scanRectFusedKernelCount<K>` | `sizeX ∈ [30,32]` 且 `sizeZ <=` 运行期上限 | 每个候选列的 `sizeX` 窗口计数（uint8） |
| `scanRectFusedKernelBallot<K>` | `K == 1`，或 `sizeZ` 超过上面的上限 | 每行的位图 |
| `wideRowPrefixSumKernel` + `wideWindowScoreKernel` | `sizeX > 32` | 不用 shared 环（走前缀和） |

计数环的 shared 占用随 `sizeZ` 线性增长，超过某个 `sizeZ` 后每 SM 常驻块数会塌陷，所以那条
路径的 `sizeZ` 上限由程序按**当前设备**的真实占用率交叉点派生，而不是写死。同理，`K=4`
（每 lane 4 个跨步列）只在 X 向 tile 足够多、或 tile 数乘 Z 子块数足够填满 GPU 时才值得用；
这两个阈值也在 `cudaSetDevice` 之后按该卡的 SM 数与 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
算出来。派生出的值可用 `SLIME_PRINT_TUNING=1` 打印，`SLIME_NO_AUTOTUNE=1` 退回编译期常量。
设备代码同时内嵌 SASS 与本机 compute capability 的 PTX，架构更新的卡交给驱动 JIT。

## 4. 不重不漏与不越界

* 候选位置被 `(block, warp, k, lane, row)` 唯一划分，每个候选恰好统计一次；
* 结果登记先 `atomicAdd` 领槽再判界，`pos` 单调递增 ⇒ `[0, cap)` 每槽恰好写一次，
  越界写在结构上不可能发生；
* 结果缓冲区满时本 block 立即停手、主机丢弃整轮并缩小 chunk 重试，所以被丢弃的命中不会
  变成漏报。

形式化证明（两个计算核的切分与窗口不变式，共 14 条引理）见 `docs/zh-cn/correctness-proof.md`，
外部权威对拍见 `docs/zh-cn/correctness.md`。

## 5. 构建与运行

```bash
./build.sh
./build/slime_main <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <out.csv> [--sort=on|off]
```

`--sort=on`（默认）产出按 `(x,z)` 字典序的 CSV，便于逐字节比对；`--sort=off` 跳过排序。
扫描区间按 256 列对齐并在 Z 向补齐，实际区间由程序打印的 `Global range:` 给出，请求区间
一定被它包含。进度条在 stderr 是终端时自动显示，`SLIME_PROGRESS=1` / `SLIME_NO_PROGRESS=1`
可强制开关。参数硬界、运行期旋钮与其余工具见 `docs/zh-cn/tools.md` 与
`docs/zh-cn/optimization-design.md`。
