/*
 * slime_hash.cuh —— 史莱姆区块判定的唯一实现（slime_main / slime_circle 共用）。
 *
 * 语义 = Minecraft Java 版的区块种子表达式 + java.util.Random(seed).nextInt(10) == 0，
 * 但全部用整数运算逐位等价地重写。tools/mccrosscheck.sh 会把 docs/en-us/index.md 里的
 * 原始 Java 抽出来交给真 JVM 跑，再与本实现逐区块对拍（1×1 窗口的输出即区块集合）。
 *
 * 用法：
 *     int64_t baseX = slimeQuadX(chunkX) + seed;   // seed 是常数项，折进 x 部分
 *     int64_t baseZ = slimeQuadZ(chunkZ);
 *     bool slime = isSlimeChunk(baseX, baseZ);
 * 热路径想自己控制拒绝分支时，用 slimeBitsFast + slimeTestBits（见下）。
 */
#ifndef SLIME_HASH_CUH
#define SLIME_HASH_CUH

#include <cstdint>
#include <cuda_runtime.h>

// warp 宽度：两个使用者（slime_main / slime_circle）都按 32 假设
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Java 表达式拆成「只依赖 x」与「只依赖 z」两半。两半都是先按 int 溢出算、
// 再符号扩展成 64 位相加；调用点把它们相加（加法在 2^64 下可结合，顺序无关）。
__host__ __device__ __forceinline__ int64_t slimeQuadX(int32_t chunkX) {
    const uint32_t ux = (uint32_t)chunkX;
    return (int64_t)(int32_t)((ux * ux) * 4987142U) + (int64_t)(int32_t)(ux * 5947611U);
}
__host__ __device__ __forceinline__ int64_t slimeQuadZ(int32_t chunkZ) {
    const uint32_t uz = (uint32_t)chunkZ;
    return (int64_t)(int32_t)(uz * uz) * 4392871LL + (int64_t)(int32_t)(uz * 389711U);
}

/* 史莱姆区块判定（worldSeed 已预加至 baseX）
 *
 * 与 Java 规范逐位等价（tools/mccrosscheck.sh 拿真 JVM 对拍），热路径 SASS 从 18 条
 * 降到 16 条。五处变形：
 *  1) (s ^ 987234911) ^ 0x5DEECE66D 合并成一个常量 XOR，省一次 64 位 XOR；
 *  2) 48x35 的 LCG 乘法整体左移 16 位再做（"中段抽取"）：
 *       Q = (X * (MULT<<16) + (0xB<<16)) mod 2^64 = W << 16,  W = (X*MULT+0xB) mod 2^48
 *     Q 的高 32 位 = W>>16，再 >>1 就是 bits = W>>17（31 位，天然无脏位）。
 *     于是老实现的「& MASK48」掩码和 64 位漏斗移位都不需要了，乘法链少一条 IMAD；
 *     +0xB 折进 IMAD.WIDE 的 64 位加数（0xB<<16 : z），不额外花指令。
 *  3) 求和后不需要任何 48 位掩码：X 高 32 位里 bit16 以上的脏数据乘的是
 *     L2 = (MULT<<16) & 0xffffffff = 0xE66D0000（16 个尾零），乘出来的贡献
 *     恰好落在 2^32 之外，被 32 位截断丢掉。
 *  4) nextInt(10) 的 "bits % 10 == 0" 用「一次 32 位乘 + 一次比较」，**不需要循环右移**：
 *     取 MAGIC = 0x4CCCCCCD（10*MAGIC ≡ 2 mod 2^32）时，在真实定义域 bits ∈ [0, 2^31) 上
 *     `(bits*MAGIC) <= 0x19999999 ⟺ bits % 10 == 0`，已对全部 2^31 个取值穷举验证。
 *     注意这与「0xCCCCCCCD + 右移 1 位」是**两条不同的常量路线**：在 0xCCCCCCCD 下旋转不可省
 *     （r=5 时 5*MAGIC mod 2^32 = 1，去掉旋转会把 %10==5 的 bits 大量误判为真，
 *       bits=10130365 即反例）；换到 0x4CCCCCCD 后旋转才成为冗余。
 *  5) 拒绝采样走真分支（bits >= 2147483640 的概率约 3.7e-9），热路径只做一次 LCG。
 *     冷路径刻意写成「__umulhi + 显式进位 + 显式旋转」并保留一个语义冗余的 & 0xFFFF：
 *     CUDA 13.4 的 ptxas 在 sm_75..sm_121 全部 12 个架构上都因此选择真分支
 *     （BSSY/@!P0 BRA/BSYNC），冷块整块被跳过；写成更短的 IMAD.WIDE 版本时，
 *     sm_100/110/120/121 会被谓词化、每次判定多发射 7 条（实测 sm_120 上差 3 条/次）。
 *     冷块长度不影响热路径，所以「写长一点」是纯赚。src/test.cpp 已证明坏种子的
 *     后继永远不是坏种子，所以分支内只需再采一次。 */
static constexpr uint32_t SLIME_REJECT = 2147483640U;

// bits % 10 == 0（单条 32 位乘 + 单条比较；**不需要**循环右移）
//
// 取 MAGIC = 0x4CCCCCCD（注意它是 0xCCCCCCCD 的「带旋转」版本：0x4CCCCCCD = ror1(0xCCCCCCCD)
// 的等价常量，满足 10*MAGIC == 2 (mod 2^32)），则在**真实定义域 bits ∈ [0, 2^31)** 上
//     (bits * 0x4CCCCCCD mod 2^32) <= 0x19999999   ⟺   bits % 10 == 0
// 成立，于是省掉原来那条 SHF.R.W（循环右移 1 位）。
// 依据：旧形式 `q = bits*0xCCCCCCCD; q = ror1(q); return q <= 0x19999999` 与本形式在
// 定义域上逐位等价 —— 已对**全部 2^31 个 bits** 穷举验证，mismatch = 0
// （对照：旧形式同样 0 mismatch）。定义域保证来自 bits = (Q>>32)>>1，最高位恒 0。
__device__ __forceinline__ bool slimeTestBits(uint32_t bits) {
    constexpr uint32_t MAGIC = 0x4CCCCCCDU;  // 10*MAGIC ≡ 2 (mod 2^32)
    constexpr uint32_t DIV10_MAX = 0x19999999U;
    return (bits * MAGIC) <= DIV10_MAX;  // 单条 IMAD + 单条 ISETP
}

// 快路径：只算到 31 位的 bits，**刻意不在这里判拒绝**。拒绝路径被提到「行」这一级，
// 由调用点的 warp 均匀门控统一处理（slime_main 的两个融合核都是这么用的：
// rejAny = warp 内是否有 lane 落到拒绝区 -> 整个 warp 一起走 slimeBallotSlow）。
__device__ __forceinline__ uint32_t slimeBitsFast(int64_t baseX, int64_t baseZ) {
    constexpr uint32_t L2 = 0xE66D0000u;    // (MULT<<16) 的低 32 位；16 个尾零是「脏位免疫」的关键
    constexpr uint32_t H2 = 0x0005DEECu;    // (MULT<<16) >> 32
    constexpr uint32_t ADD2 = 0x000B0000u;  // 0xB << 16
    constexpr uint32_t C_LO = 0xE434E432u;  // C 的 bit0..31（C = 987234911 ^ 0x5DEECE66D）
    constexpr uint32_t C_HI = 0x00000005u;  // C 的 bit32..47（= C>>32；注意 C mod 2^16 = 0xE432，不是这个）

    // 求和 + 异或；bit48..63 的脏数据由 L2 的 16 个尾零吸收，无需掩码
    uint64_t s = (uint64_t)baseX + (uint64_t)baseZ;
    uint32_t X_lo = (uint32_t)s ^ C_LO;
    uint32_t X_hi = (uint32_t)(s >> 32) ^ C_HI;
    // 48x35 乘法：低半给 64 位乘，高半 + 交叉项给高 32 位；0xB<<16 与进位由 64 位加数一次做完
    uint32_t z = X_hi * L2 + X_lo * H2;
    uint64_t Q = (uint64_t)X_lo * L2 + (((uint64_t)z << 32) | (uint64_t)ADD2);
    return (uint32_t)(Q >> 32) >> 1;  // = (W >> 16) >> 1 = W >> 17
}

/*
 * 拒绝路径的出线慢函数：只被行级 warp 均匀门控里那条「概率 ~4e-9」的分支调用。
 *
 * 为什么必须是 __noinline__ 的真 CALL：把这段写在行循环的内联位置时，无论怎么排列，
 * ptxas 要么把它 if-convert 成每次都发射的谓词指令（慢路径 20+ 条全进热路径），要么把
 * 快路径的 4 个 48 位中间值 CSE 复用、钉在寄存器里，寄存器 40 -> 48 —— 越过 41 这条线
 * 就掉到 5 blocks/SM（实测 warps_active 98.5% -> 82.3%）。真 CALL 让 ABI 在调用点处理
 * 活跃值，慢路径因此完全不占热路径寄存器（实测 40 -> 40）。
 *
 * 这里刻意按「老写法」（先掩 48 位、再 XOR、再乘）重算一遍：与快路径形状不同，
 * 避免 ptxas 把两者合并。函数几乎从不执行，写长一点是纯赚。
 */
__device__ __noinline__ uint32_t slimeBallotSlow(int64_t baseX, int64_t baseZ) {
    constexpr uint64_t MULT = 0x5DEECE66DULL;
    constexpr uint64_t ADD = 0xBULL;
    constexpr uint64_t MASK48 = 0xFFFFFFFFFFFFULL;
    constexpr uint64_t XOR_ALL = 987234911ULL ^ 0x5DEECE66DULL;

    uint64_t rnd = ((uint64_t)baseX + (uint64_t)baseZ) & MASK48;
    rnd = ((rnd ^ XOR_ALL) * MULT + ADD) & MASK48;
    uint32_t bits = (uint32_t)(rnd >> 17);
    if (bits >= SLIME_REJECT) {  // Java 的拒绝路径：推进一次状态后重采（至多一次）
        rnd = (rnd * MULT + ADD) & MASK48;
        bits = (uint32_t)(rnd >> 17);
    }
    return __ballot_sync(0xFFFFFFFFu, slimeTestBits(bits));
}

__device__ __forceinline__ bool isSlimeChunk(int64_t baseX, int64_t baseZ) {
    constexpr uint32_t L2 = 0xE66D0000u;    // (MULT<<16) 的低 32 位；16 个尾零是「脏位免疫」的关键
    constexpr uint32_t H2 = 0x0005DEECu;    // (MULT<<16) >> 32
    constexpr uint32_t ADD2 = 0x000B0000u;  // 0xB << 16
    constexpr uint32_t C_LO = 0xE434E432u;  // C 的 bit0..31（C = 987234911 ^ 0x5DEECE66D）
    constexpr uint32_t C_HI = 0x00000005u;  // C 的 bit32..47（= C>>32；注意 C mod 2^16 = 0xE432，不是这个）
    constexpr uint32_t REJECT = 2147483640U;
    constexpr uint32_t MAGIC = 0x4CCCCCCDU;  // 10*MAGIC ≡ 2 (mod 2^32)；见 slimeTestBits 的穷举说明
    constexpr uint32_t DIV10_MAX = 0x19999999U;

    // 求和 + 异或；bit48..63 的脏数据由 L2 的 16 个尾零吸收，无需掩码
    uint64_t s = (uint64_t)baseX + (uint64_t)baseZ;
    uint32_t X_lo = (uint32_t)s ^ C_LO;
    uint32_t X_hi = (uint32_t)(s >> 32) ^ C_HI;
    // 48x35 乘法：低半给 64 位乘，高半 + 交叉项给高 32 位；0xB<<16 与进位由 64 位加数一次做完
    uint32_t z = X_hi * L2 + X_lo * H2;
    uint64_t Q = (uint64_t)X_lo * L2 + (((uint64_t)z << 32) | (uint64_t)ADD2);
    uint32_t V = (uint32_t)(Q >> 32);
    uint32_t bits = V >> 1;  // = (W >> 16) >> 1 = W >> 17

    if (__builtin_expect(bits < REJECT, 1)) {
        return (bits * MAGIC) <= DIV10_MAX;  // 单条 IMAD + 单条 ISETP（无需循环右移）
    }
    // Java 的拒绝路径：推进一次状态后重采（至多一次）。旧状态 = Q>>16 (mod 2^48)
    uint32_t s1_lo = __funnelshift_r((uint32_t)Q, V, 16);  // = (Q_lo>>16)|(V<<16)
    uint32_t s1_hi = (V >> 16) & 0xFFFFu;                  // 掩码冗余，但保留可让 ptxas 出真分支
    uint32_t lo = s1_lo * L2;                              // 低 32 位
    uint32_t hi = __umulhi(s1_lo, L2);                     // 高 32 位
    uint32_t z2 = s1_hi * L2 + s1_lo * H2;
    uint32_t lo2 = lo + ADD2;
    uint32_t V2 = hi + z2 + (uint32_t)(lo2 < ADD2);        // 含低位到高位的进位
    uint32_t bits2 = V2 >> 1;
    return (bits2 * MAGIC) <= DIV10_MAX;
}

#endif  // SLIME_HASH_CUH
