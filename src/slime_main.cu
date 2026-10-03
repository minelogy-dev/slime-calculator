/*
 * Copyright 2026 minelogy
 * Licensed under the Apache License, Version 2.0
 */

/*
 * slime_main —— 枚举「sizeX×sizeZ 区块窗口内史莱姆区块数 >= threshold」的全部位置，
 * 输出 CSV：x,z,slime_count（x/z 是窗口起始区块的方块坐标，恒为 16 的倍数）。
 *
 * 四条扫描路径（GPUWorker::run 里按 sizeX / sizeZ / K 选路，四者逐位等价）：
 *
 *   前提：K = FUSED_K 当「X 方向 tile 够多」或「tile 数 × Z 子块数够填满 GPU」，否则 K = 1。
 *
 *   1) scanRectFusedMerge4Kernel<K>   默认路径：K != 1 且 sizeX <= 29
 *        每 lane 负责 4 个**相邻**候选列，热路径只维护一个 bundle 级松上界
 *        U = sizeZ 行 × (sizeX+3) 列并集内的史莱姆数；U < threshold 即整组不可能命中。
 *        于是 X 方向的逐列滚动记账（SHF+LOP3+POPC+LDS.U8+STS.U8）被摊到 4 个候选上，
 *        过闸的 bundle 再从 shared 行位图环按需重算精确值。并集位段宽 sizeX+3 必须落在
 *        「相邻两字拼成的 64 位」内，这是 sizeX <= 29 的真正来源（详见该核上方注释）。
 *   2) scanRectFusedKernelCount<K>    sizeX ∈ [30,32] 且 sizeZ <= 运行期派生的上限
 *        shared 环存「每个候选列的 sizeX 窗口计数」（uint8）。Z 滚动 = 加新行、减旧行，
 *        取旧行只要 1 条 LDS —— 代价是动态 shared 随 sizeZ 增长，占用率会塌陷。
 *   3) scanRectFusedKernelBallot<K>   同族回退：K == 1，或 sizeZ 超过上面的上限
 *        shared 环存「每行的 32 位史莱姆位图」。取旧行要现算 SHF+LOP3+POPC，
 *        但 shared 占用与 sizeZ 无关。
 *   4) wideRowPrefixSumKernel + wideWindowScoreKernel    宽窗回退：sizeX > 32
 *        X 窗口宽到放不进「相邻两字拼成的 64 位位段」，于是退回两段式：先每行前缀和，
 *        再对每个候选用两次查表求区间和（上限 sizeX <= 255）。
 *
 * 四条路径共用同一套「不重不漏 + 不越界」协议：
 *   * 候选位置被 (block, warp, k, lane, row) 唯一划分，X 窗口永不跨 block（halo 走环/寄存器）；
 *   * 结果登记先 atomicAdd 领槽再判界，pos 单调递增 ⇒ [0,cap) 每槽恰好写一次，越界写结构性不可能；
 *   * 缓冲区满则本 block 立即停手、主机丢弃本轮并缩小 chunk 重试 ⇒ 不会丢命中（见 ResultPool / run()）。
 *
 * 命名约定：融合核的名字 = 它往 shared 环里放什么（行位图 / 逐列窗口计数）+ 记账方式；
 * wide* = sizeX > 32 的回退路径。
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <atomic>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>  // GlobalMemoryStatusEx：Windows 侧没有 sysconf
#include <io.h>       // _isatty / _fileno（进度条判断 stderr 是否为终端）
#else
#include <unistd.h>
#endif
#include <vector>

struct HitResult {
    int x, z, count;
};
static_assert(__is_pod(HitResult), "HitResult must be POD for zero-overhead");

// 进度条：默认开启（stderr 是终端时自动显示）。定义 SLIME_NO_PROGRESS_CODE 可把整段
// 进度代码编译掉 —— 用于证明它对设备侧零影响：两个二进制在同一条件下比较
// smsp__inst_executed / sm__cycles_elapsed 应当逐位相同。
#ifndef SLIME_NO_PROGRESS_CODE
#define SLIME_PROGRESS_CODE 1
#endif
int64_t seed;
int32_t startX, startZ;  // 这四个变量是指参与isSlimeChunk的区块的范围而不是Valid的值的范围
int32_t endX, endZ;
int32_t sizeX, sizeZ;   // 这个是检擦的每个矩形的大小
int32_t width, height;  // 这个是用startXZ与endXZ算出的宽和高
int32_t threshold;
int64_t step;
int64_t valid_endZ;

int64_t* h_baseX;
int64_t* h_baseZ;

#include "slime_hash.cuh"

#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

// 行前缀和：一行一个 warp 协作（Warp Shuffle），整行连续累加
#define WIDTH_ALIGN 256  // X 方向对齐粒度，必须是 WARP_SIZE 的整数倍
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

/*
 * 参数硬界：由下面的实现细节推导，main 中强制校验，不是经验约定。
 *
 *   sizeZ <= MAX_SIZE_Z
 *       wideWindowScoreKernel 用定长数组 winVals[MAX_SIZE_Z] 做滑动窗口的环形缓冲。
 *
 *   sizeX <= MAX_SIZE_X = 255
 *       前缀和整行连续累加并以 uint8_t 存储、允许自然回绕，消费者只做
 *       (uint8_t)(a - b) 的模 256 减法来求一行上的窗口计数。窗口在单行上的跨度恰为
 *       sizeX，所以只要 sizeX < 256 这个差值就一定等于真值：回绕发生在窗口左端点
 *       之前也好、之后也好，模 256 的差值都对（例如 (uint8_t)(1 - 255) == 2），
 *       依赖的是 C 对无符号整数「结果按 2^N 取模」的保证（uint8_t 即 unsigned char，
 *       无填充位、纯二进制），由硬件的 8bit 截断完成，无需任何显式取模或进位补偿。
 */
static constexpr int MAX_SIZE_Z = 32;
static constexpr int MAX_SIZE_X = 255;
// 融合路径的 X 窗口上限：窗口位段必须落在「相邻两个字拼成的一个 64 位」里
static constexpr int MAX_FUSED_SIZE_X = 32;
// T1 v2（merge4 核）的 X 窗口上限：bundle 并集位段宽 sizeX+3，要一次 32 位
// __funnelshift_r 取出 ⇒ sizeX+3 <= 32 ⇒ sizeX <= 29。sizeX ∈ [30,32] 仍走旧核。
static constexpr int MAX_MERGE4_SIZE_X = 29;
// 计数环变体的 sizeZ 上限：动态 shared = sizeZ × TILE_W 字节，占用率随
// sizeZ 塌陷。实测（ncu 交错，2340 MHz，K=4）相对ballot 环核：sizeZ=8 −7.0%、17 −4.9%、
// 24 +3.9%、32 +4.3% ⇒ 只有 sizeZ <= 20 才走计数环路径，超过回退 scanRectFusedKernelBallot。
static constexpr int COUNT_RING_MAX_SIZE_Z = 20;
// 融合核每线程负责的跨步列数 K。K=4 时每候选约 34.5 条线程指令（K=1 是 57.5），
// 寄存器约 37、shared 8.4KB，占用率不掉；K=8 静态收益趋平但寄存器压力上升。
static constexpr int FUSED_K = 4;
// K=4 时每个 warp 覆盖 128 个输出列 => warp 数只有 K=1 的 1/4；
// 批次列数低于该阈值时改回 K=1，避免 X 方向窄的批次把 GPU 填不满。
static constexpr int FUSED_K_MIN_COLS = 200000;
// K=4 与 K=1 的切换下界：K=4 的 block 数（tiles x zSub）低于这个数就回退 K=1。
// 34 SM x 6 blocks = 204 个常驻槽，取 2 个波形 408，避免只发几十个 block 时
// 大量发射槽空转（那正是「K=4 每候选省指令」吃不回来的地方）。
// **显式 -D 覆盖优先**：默认值取 -1 表示「按真实常驻槽数自动派生」（见
// tuneParallelismThresholds）。这样既有的测试旋钮语义完全不变：
//   -DSLIME_K4_MIN_BLOCKS=0          -> 强制 K=4（verify.sh 的 k4 目标、fullcheck.sh 依赖它）
//   -DSLIME_K4_MIN_BLOCKS=2147483647 -> 强制 K=1
// 若默认写成 408，则 `-D...=0` 会被派生逻辑用 max() 抬回 408，**静默废掉 K=4 覆盖**。
#define SLIME_K4_MIN_BLOCKS_AUTO (-1)
#ifndef SLIME_K4_MIN_BLOCKS
#define SLIME_K4_MIN_BLOCKS SLIME_K4_MIN_BLOCKS_AUTO
#endif
static constexpr int FUSED_K4_MIN_BLOCKS = SLIME_K4_MIN_BLOCKS;
// 标定基准：本机（4060 Ti, 34 SM）实测得到 6 blocks/SM ⇒ 204 个常驻槽。
// 下面三个阈值都是按这个基准**等比例缩放**的，slots == 204 时结果与改动前逐位相同。
static constexpr int CALIB_SLOTS = 204;

// 越界候选列的毒值：越界列的窗口掩码为 0，cnt[k] 于是恒等于这个值，
// 门的 `mx >= threshold` 永远不会被它触发（threshold 已在 main 里归一化到 >= 0）。
static constexpr int POISON_CNT = -(1 << 30);
// d_baseX 设备缓冲尾部填充条目数：>= 一个 block 覆盖的最大列数 TILE_W（K=4 时 1024），
// 这样融合核可以无条件 __ldg（越界列的值必被掩码丢掉）。
// 原来写死 64*32 = 2048（是 TILE_W 的 2 倍、无出处）；改成与 TILE_W 绑定并加 static_assert，
// 消除「将来把 K 或 WARPS_PER_BLOCK 调大 ⇒ 无条件 __ldg 静默越界读」的地雷。
static constexpr int BASE_X_PAD = WARPS_PER_BLOCK * WARP_SIZE * FUSED_K;
static_assert(BASE_X_PAD >= WARPS_PER_BLOCK * WARP_SIZE * FUSED_K,
              "BASE_X_PAD 必须 >= 一个 block 覆盖的最大列数，否则越界列的 __ldg 会越过缓冲尾部");
// 坐标硬界：原版世界边界是 ±1,874,999 区块，放宽到 ±2,000,000 仍然远小于
// INT_MAX（2,147,483,647）。这个界现在是「内存安全」论证的支点：
//   width <= 2*MAX_COORD + 2  =>  rangeX = width - sizeX + 1 <= 2*MAX_COORD + 2，
// 只要结果缓冲区 cap >= rangeX（见 run() 里的下限），"每批候选列数 <= cap" 的
// 几何就一定存在（safeRows = cap/rangeX >= 1），而命中数恒 <= 候选数，于是
// 按这个下界切块永不溢出；更宽的块靠实测密度放大，溢出则由内核 clamp 兜底。
static constexpr int MAX_COORD = 2000000;
// 结果缓冲区上限（768MB）与 pinned 中转缓冲（48MB）。cap 按空闲显存取 25%，
// 但下限必须是 rangeX（最多 48MB），上限防止把显存吃光。
static constexpr int64_t MAX_BATCH_OUTPUT_CAP = 64LL * 1024 * 1024;
// 结果池块大小：一次 drain 的量级即可（drain 太大会让写盘线程等更久才动起来）
// 结果池的"块"= 排序阶段的一个**有序段**。它同时决定两件事：
//   ① 排序线程的分工粒度（段数少了并行度就低，但排序本来就藏在 GPU 时间里，无所谓）；
//   ② 收尾 k 路归并的每记录比较次数（现在是线性选最小，O(k)）—— 这才是真正的成本：
//      1M/块时全图 40.7M 命中 = 41 个段，每记录平均比 ~20 次 ≈ 3~4 s；8M/块 ⇒ 6 个段、平均
//      比 ~3 次。块大了以后 spill（落盘）也更难触发，整份结果都能留在一次 malloc 的 arena 里。
static constexpr size_t POOL_BLOCK_SLOTS = 1 << 23;  // 8M 条 = 96MB/段
// 融合核把 chunk 的候选行再沿 Z 切给 gridDim.y 个 block，每块只算这么多行。
// 存在的理由是「波形量化」：block 数由 X 几何（rangeX/outW）决定，X 窄的负载只能发出
// 几十~几百个 block（ztall 只有 135 个，而 34 SM × 6 block = 204 个常驻槽），于是
// 大量发射槽空转（ncu: Active Warps Per Scheduler 只有 9.31/12）。沿 Z 切分把 block
// 数乘上 ceil(chunkRows/该值)，代价只是每块多读 sizeZ-1 行 halo（4096 行时 ≤0.76%）。
// 两个值都可以用 -DSLIME_Z_SUB_ROWS / -DSLIME_Z_SPLIT_MIN_TILES 覆盖，用来在测试里
// 强制走「不切 Z」（zSub=1）那条路径 —— 那正是本步第一版隐藏了 bug 的地方
// （zSub=1 时每个 block 仍被 Z_SUB_ROWS 截断，等于只扫了每个 chunk 的前 2048 行）。
#ifndef SLIME_Z_SUB_ROWS
#define SLIME_Z_SUB_ROWS 2048
#endif
#ifndef SLIME_Z_SPLIT_MIN_TILES
#define SLIME_Z_SPLIT_MIN_TILES 2048
#endif
static constexpr int Z_SUB_ROWS = SLIME_Z_SUB_ROWS;
// 但只在「X 几何发出的 block 数明显填不满 GPU」时才切：切分要重读 sizeZ-1 行 halo
// （≤1.5%），而 block 数已经够多（≥ 约 10 个波形）时波形量化本来就只剩 ~0.5%，
// 再切反而略亏（缩放点实测 7 组 chunk 下 7547 -> 7588 ms）。阈值取 2048 个 tile。
static constexpr int Z_SPLIT_MIN_TILES = SLIME_Z_SPLIT_MIN_TILES;


// 自适应切块：单次 launch 在被自适应的那条轴（融合路径=Z 行数，回退路径=X 列数）
// 上的跨度上限。上限只管「万一溢出时重做的量级」：内核一旦发现缓冲区满就停手，
// 所以失败的 launch 只花「填满缓冲区」的时间，重试的代价很小 —— 因此一开始就可以
// 直接用满这个跨度，不必从很小往上试探。
static constexpr int64_t MAX_CHUNK_ROWS = 65536;
// 下一块按「实测密度 × 50% 填充」放大，且最多翻倍，避免来回震荡。
static constexpr double DENSITY_FILL = 0.5;
// 单点重试上限：safeRows 的几何恒不溢出，这个计数只用于把内部 bug 变成明确报错而非死循环。
static constexpr int MAX_RETRIES = 64;

// 一行一个 warp，整行连续累加；不再按 X 分段，所以没有段内/跨段之分
__global__ void wideRowPrefixSumKernel(
    const int64_t* __restrict__ d_baseX,
    const int64_t* __restrict__ d_baseZ,
    int width,
    int blockHeight,
    int baseZ_offset,
    uint8_t* __restrict__ d_row_ps,
    int rowPsWidth) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (row >= blockHeight) return;

    int64_t bz = __ldg(d_baseZ + baseZ_offset + row);
    uint8_t* __restrict__ psRow = d_row_ps + (int64_t)row * rowPsWidth;
    int prefix = 0;  // 整行累加，允许 uint8_t 回绕，见 MAX_SIZE_X 处的说明
    for (int col = lane; col < width; col += WARP_SIZE) {
        // 同一个 block 的 8 个 warp 按相同顺序读 d_baseX，命中 L1/L2
        int val = isSlimeChunk(__ldg(d_baseX + col), bz);
#pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            int n = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val += n;
        }
        val += prefix;
        psRow[col] = (uint8_t)val;
        prefix = __shfl_sync(0xffffffff, val, WARP_SIZE - 1);
    }
}

/*
 * 位图融合扫描（sizeX <= MAX_FUSED_SIZE_X 时启用，取代「前缀核 + 滑窗核」两段式）。
 *
 * 线程 <-> 列、沿 Z 串行推进：
 *   * 每行每个 warp 用 __ballot_sync 把 32 个判定压成一个 32 位位图，lane0 写入
 *     shared 的 64 槽环形。RING=64 > sizeZ 上限 32，保证本行写入的槽不会覆盖
 *     掉「sizeZ 行前」那个还需要被减去的槽。
 *   * X 方向窗口完全不落显存：__funnelshift_r(本行本 warp 位图, 同行的下一个字, lane)
 *     一次取出「以本列为起点、宽 sizeX」的位段，__popc 计数。那个「下一个字」就是
 *     跨 warp 的 halo，所以每个 block 覆盖 TILE_W 个输入列、只输出
 *     OUT_W = TILE_W - sizeX + 1 个候选，窗口永不跨 block；最后一个 warp 的 halo 读到
 *     的是哨兵字 0，而会读到它的那些 lane 恰好被 outputActive 关掉。
 *   * Z 方向滚动 = 加本行的 popcount、减 sizeZ 行前那一行的同一个 popcount。
 *
 * 于是每个候选中心的 DRAM 流量为 0（旧实现是写 1 B 前缀 + 读 1 B），状态只有
 * RING*(WARPS_PER_BLOCK*K+1) 个 uint32 的 shared memory（K=4 时 8.4 KB）。
 *
 * 越界写的结构性防护（cap）：
 *   登记前先 atomicAdd 领槽再判界，pos 单调递增，所以 [0, cap) 里的每个槽都恰好被写
 *   一次（无空洞、无覆盖），pos >= cap 的命中直接丢弃并让本 block 立刻停手。于是
 *   「写 d_results 越界」在结构上不可能发生，跟主机端批次尺寸估计对不对无关。
 *   缓冲区满时主机丢弃本轮的全部结果、缩小 chunk 重试，所以丢弃的命中不会丢结果。
 */
// ballot 环核（每行位图存进 shared 环）：sizeZ > COUNT_RING_MAX_SIZE_Z 或 K == 1 时走这条路径。
template <int K>
__global__ void scanRectFusedKernelBallot(
    const int64_t* __restrict__ d_baseX,
    const int64_t* __restrict__ d_baseZ,
    int width, int blockHeight, int baseZ_offset,
    int sizeX, int sizeZ, int threshold,
    int startX, int baseZ,
    int colOffset, int validRangeX,
    HitResult* d_results, unsigned long long* d_pos, unsigned long long cap,
    int zSubRows) {  // 每个 Z 子块负责的候选行数（= ceil(chunk 候选行 / gridDim.y)）
    constexpr int TILE_W = WARPS_PER_BLOCK * WARP_SIZE * K;  // 一个 block 覆盖的输入列数
    constexpr int WORDS = WARPS_PER_BLOCK * K;
    // 行距取 36 而不是 33：36*4=144 字节是 16 的倍数，于是每行每 warp 的 4 个字
    // 落在 16B 边界上 —— lane0 可以用一条 STS.128 写完 4 个 ballot 字；
    // 行首也是 8B 对齐，旧行的 5 个字可以用 LDS.64 合并成 3 条。
    constexpr int STRIDE = ((WORDS + 4) / 4) * 4;  // 33->36 (K=4), 1->4 (K=1)
    constexpr int RING = 64;           // 必须 > 最大 sizeZ
    static_assert(RING > MAX_SIZE_Z, "ring must be larger than the largest sizeZ");
    static_assert(K >= 1, "K must be positive");
    __shared__ uint32_t s_rows[RING * STRIDE];

    const int tx = threadIdx.x;
    const int lane = tx & 31;
    const int warp = tx >> 5;
    const int wordBase = warp * K;

    // 本 block 负责的候选行区间 [subBaseZ, subBaseZ + subRows)（blockHeight 是 chunk 的
    // 输入行数，所以 chunk 的候选行数 = blockHeight - sizeZ + 1）。各 block 的输出行区间
    // 严格相邻且不重叠；输入多读 sizeZ-1 行 halo，只影响 ncu 里的指令数不影响结果。
    const int chunkCandRows = blockHeight - sizeZ + 1;
    const int subStart = blockIdx.y * zSubRows;
    const int subRows = min(zSubRows, chunkCandRows - subStart);
    const int subInputRows = subRows + sizeZ - 1;
    const int subBaseZ = baseZ + subStart;
    const int subZOffset = baseZ_offset + subStart;

    // 哨兵：索引 WORDS 恒为 0（只有最后一个 warp 的最后一个跨步列会读到它）
    for (int i = tx; i < RING; i += blockDim.x) s_rows[i * STRIDE + WORDS] = 0u;

    const int outW = TILE_W - sizeX + 1;
    const int colBase = blockIdx.x * outW;  // 本 block 的输入列起点（= 候选列起点）
    const uint32_t xmask = (sizeX >= 32) ? 0xFFFFFFFFu : ((1u << sizeX) - 1u);

    // 每线程 K 个跨步列：cl = (warp*K + k)*32 + lane  ->  warp 内 K 次 ballot 全在寄存器
    int64_t bx[K];
    bool outA[K];
    uint32_t xmaskLane[K];
    int cnt[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int cl = (wordBase + k) * WARP_SIZE + lane;  // block 内列
        const int clLocal = colBase + cl;                  // 批次内列
        const int gcol = colOffset + clLocal;              // 全局列
        outA[k] = (cl < outW) && (clLocal < validRangeX);
        // d_baseX 尾部有 BASE_X_PAD 个 0，越界列可以无条件读（值必被掩码丢掉）
        bx[k] = __ldg(d_baseX + gcol);
        // 越界列的窗口掩码取 0 => 该 lane 的窗口计数恒为 0 => cnt[k] 保持毒值
        xmaskLane[k] = outA[k] ? xmask : 0u;
        cnt[k] = outA[k] ? 0 : POISON_CNT;
    }
    unsigned stop = 0u;  // R5: 用 unsigned 而不是 bool，避免 ptxas 把它按字节打包/解包
    // 行循环拆两段（汇编审计第 2 条 + 建模 Agent 第 ② 条）：
    //   前置段 r ∈ [0, sizeZ-1)：窗口还没攒满 —— 只累加本行，不减旧行、不发射；
    //   主段   r ∈ [sizeZ-1, subInputRows)：稳态 —— 无条件减旧行 + 发射闸门，
    //          于是「r >= sizeZ」「r >= sizeZ-1」这两个每行都要判一次的单调条件整个消失。
    // 前置段之所以能"无条件减旧行"，是因为环形缓冲**整块预清零**：r < sizeZ 时
    // (r-sizeZ)&63 落在 32..63，而这些槽要到 r+64-sizeZ ≥ 32 > r 才会被写，此刻读到的就是 0，
    // popc(0&xmask)=0 —— 等于减了个零。RING=64 > sizeZ 保证旧行槽不会撞上本行。
    for (int i = tx; i < RING * STRIDE; i += blockDim.x) s_rows[i] = 0u;
    __syncthreads();
    const int peel = (sizeZ - 1 < subInputRows) ? sizeZ - 1 : subInputRows;
    // R1: bz 的行地址用指针归纳变量 —— d_baseZ + subZOffset 只在循环外算一次，
    //     之后每行只做一次 8 字节自增（uniform 通路），不再从 r 重新算一遍。
    const int64_t* __restrict__ bzPtr = d_baseZ + subZOffset;
    for (int r = 0; r < peel; ++r) {
        const int64_t bz = __ldg(bzPtr++);  // R1: 指针归纳变量，每行只自增 8 字节
        uint32_t ballots[K];
        bool rejAny = false;  // 本行 K 个格子里是否有 lane 落进 Java 的拒绝区间
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t bits = slimeBitsFast(bx[k], bz);
            rejAny = rejAny || (bits >= SLIME_REJECT);
            ballots[k] = __ballot_sync(0xFFFFFFFFu, slimeTestBits(bits));
        }
        // 概率 ~4e-9：整行只开一个 warp 均匀门控（VOTE.ANY -> 均匀谓词 -> @!P0 BRA）。
        // 均匀分支不划发散区，所以热路径里 4 对 BSSY/BSYNC + 4 条谓词 BRA 全部消失；
        // 慢路径是出线 CALL，只在该分支里出现。
        if (__builtin_expect(__any_sync(0xFFFFFFFFu, rejAny) != 0, 0)) {
#pragma unroll
            for (int k = 0; k < K; ++k) ballots[k] = slimeBallotSlow(bx[k], bz);
        }
        if (lane == 0) {
            if constexpr (K == 4) {  // 一条 STS.128 取代 4 条谓词化 STS.32
                *(uint4*)&s_rows[(r & (RING - 1)) * STRIDE + wordBase] =
                    make_uint4(ballots[0], ballots[1], ballots[2], ballots[3]);
            } else {
#pragma unroll
                for (int k = 0; k < K; ++k)
                    s_rows[(r & (RING - 1)) * STRIDE + wordBase + k] = ballots[k];
            }
        }
        __syncthreads();  // 前置段不可能置 stop，用普通 barrier 即可
        const uint32_t* row = s_rows + (r & (RING - 1)) * STRIDE;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t hi = (k + 1 < K) ? ballots[k + 1] : row[wordBase + K];
            cnt[k] += __popc(__funnelshift_r(ballots[k], hi, lane) & xmaskLane[k]);
        }
    }
    for (int r = peel; r < subInputRows; ++r) {
        const int64_t bz = __ldg(bzPtr++);  // R1: 指针归纳变量，每行只自增 8 字节
        uint32_t ballots[K];
        bool rejAny = false;  // 本行 K 个格子里是否有 lane 落进 Java 的拒绝区间
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t bits = slimeBitsFast(bx[k], bz);
            rejAny = rejAny || (bits >= SLIME_REJECT);
            ballots[k] = __ballot_sync(0xFFFFFFFFu, slimeTestBits(bits));
        }
        // 概率 ~4e-9：整行只开一个 warp 均匀门控（VOTE.ANY -> 均匀谓词 -> @!P0 BRA）。
        // 均匀分支不划发散区，所以热路径里 4 对 BSSY/BSYNC + 4 条谓词 BRA 全部消失；
        // 慢路径是出线 CALL，只在该分支里出现。
        if (__builtin_expect(__any_sync(0xFFFFFFFFu, rejAny) != 0, 0)) {
#pragma unroll
            for (int k = 0; k < K; ++k) ballots[k] = slimeBallotSlow(bx[k], bz);
        }
        if (lane == 0) {
            if constexpr (K == 4) {  // 一条 STS.128 取代 4 条谓词化 STS.32
                *(uint4*)&s_rows[(r & (RING - 1)) * STRIDE + wordBase] =
                    make_uint4(ballots[0], ballots[1], ballots[2], ballots[3]);
            } else {
#pragma unroll
                for (int k = 0; k < K; ++k)
                    s_rows[(r & (RING - 1)) * STRIDE + wordBase + k] = ballots[k];
            }
        }
        // ③ 这同一个 barrier 兼两用：给 halo 读建立可见性 + 把"缓冲区满"折成块均匀值。
        //    于是行循环的退出是均匀的（不再有按线程 break），编译器才有机会把行计数器、
        //    环槽地址、比较这些标量搬到 uniform 数据通路上。
        if (__syncthreads_or((int)stop)) break;
        const uint32_t* row = s_rows + (r & (RING - 1)) * STRIDE;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            // 本行的 halo 字：k < K-1 时它就是本 warp 自己的 ballots[k+1]（__ballot_sync 对
            // 全 lane 返回同一掩码，人人手上都有），只有 k == K-1 那个字属于下一个 warp、
            // 才必须读 shared。shared 环的作用本来是跨 warp 传 halo，warp 内部白绕一趟 LDS。
            const uint32_t hi = (k + 1 < K) ? ballots[k + 1] : row[wordBase + K];
            cnt[k] += __popc(__funnelshift_r(ballots[k], hi, lane) & xmaskLane[k]);
        }
        {  // 稳态：sizeZ 行前那一行必然已经写过（前置段已把环清零兜住 r = sizeZ-1）
            const uint32_t* old = s_rows + ((r - sizeZ) & (RING - 1)) * STRIDE;
#pragma unroll
            for (int k = 0; k < K; ++k)
                cnt[k] -= __popc(__funnelshift_r(old[wordBase + k], old[wordBase + k + 1], lane) & xmaskLane[k]);
        }
        {
            const int z = subBaseZ + r - (sizeZ - 1);
#ifdef SLIME_SCAN_ONLY
            // 只保留命中的数据依赖（防止整段被优化掉），不做任何登记：
            // 无 atomicAdd、无 d_results 的 12 字节写入、无 D2H 拷贝、无 CSV 行
            (void)z;
#pragma unroll
            for (int k = 0; k < K; ++k)
                if (outA[k] && cnt[k] >= threshold) ((int*)d_results)[tx] = cnt[k];
#else
            // ① 每行只开一个 warp 均匀的发射闸门：先把 K 个命中谓词归约成一个布尔，再用
            //    __any_sync 变成 warp 均匀值。均匀分支不需要 BSSY/BSYNC 划发散区，所以
            //    常见情形（本行无任何命中）只用一条 BRA 就跳过整段登记代码 —— 而原来的
            //    写法是每个 k 各一对 BSSY/BSYNC + 谓词 BRA（K=4 时每行 ~40 条簿记）。
            // 越界列被毒值保护，所以只需要「最大值是否够阈值」；K-1 条 IMNMX + 1 条 ISETP
            int mx = cnt[0];
#pragma unroll
            for (int k = 1; k < K; ++k) mx = (mx > cnt[k]) ? mx : cnt[k];
            bool anyHit = (mx >= threshold);
            if (__any_sync(0xFFFFFFFFu, anyHit)) {
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    if (outA[k] && cnt[k] >= threshold) {
                        const unsigned long long pos = atomicAdd(d_pos, 1ULL);
                        if (pos >= cap) {  // 先判界再写：越界写在结构上不可能发生
                            stop = 1u;     // 缓冲区满 -> 停手，主机丢弃本轮并缩小 chunk 重试
                            break;
                        }
                        d_results[pos].x = (startX + colBase + (wordBase + k) * WARP_SIZE + lane) * 16;
                        d_results[pos].z = z * 16;
                        d_results[pos].count = cnt[k];
                    }
                }
            }
#endif
        }
    }
}

/*
 * 位图融合扫描 · 计数环变体：与 scanRectFusedKernelBallot 同构，
 * 唯一的差别是「Z 方向滚动记账」的载体 —— 从「每行 ballot 位图环」换成「每行每列的
 * 窗口计数环」。启用条件（两个守卫，见 GPUWorker::run 的 launch 点）：
 *     sizeZ <= COUNT_RING_MAX_SIZE_Z(20)  且  K > 1；否则一律回退 scanRectFusedKernelBallot。
 *
 * 动机：同一行的「位图 -> 窗口计数」翻译原本被付了两遍 —— 一遍算本行的 ws 加进 cnt，
 * 一遍在 sizeZ 行后从位图里把同一个值重算出来减掉（SHF + LOP3 + POPC，其中 2 条动 ALU）。
 * 把计数本身物化进环（uint8，计数 ∈ [0, sizeX] ≤ 32，逐位无损）之后，旧行只剩
 * 1 条 LDS.U8 + 一条三操作数 IADD3；环深恰为 sizeZ，读旧写新落在同一个槽（**读在写前**）。
 *
 * 跨 warp halo 只剩「下一个 warp 的首字」，但**每行仍要一个独立槽位**（HALO_RING=4）：
 * 若退化成单槽 s_halo[warp]，跑在前面的 warp w+1 会在第 r+1 行把该槽改写成下一行的首字，
 * 而 warp w 可能还没读完第 r 行的值 —— 实测到 ±1 的窗口计数错误，且两次运行结果互不相同。
 * D>=3 的充分性：写 halo(r+2) 必须先穿过 barrier(r+1)，而任何线程到达 barrier(r+1)
 * 之前一定已经读完 halo(r)（程序序），故 D=4 充分；哨兵槽整个 kernel 生命周期恒为 0。
 *
 * 为什么需要两个守卫：
 *   * sizeZ：动态 shared = sizeZ × TILE_W 字节，占用率随 sizeZ 塌陷。ncu 交错实测
 *     （2340 MHz，K=4）相对ballot 环核：sizeZ=8 −7.0%、17 −4.9%、**24 +3.9%、32 +4.3%**
 *     ⇒ 只在小 sizeZ 启用。
 *   * K：K=1 时每候选摊到的记账太少，计数环反而更贵（ALU/候选 13.75 -> 18.00，
 *     ncu cycles +38%）⇒ K=1 一律走 ballot 环核。
 *   （K 扫描：K=4 −4.9%、K=5 −4.6%、K=3 +4.2%、K=2 +13.5%，故 FUSED_K 保持 4。）
 *
 * 越界写的结构性防护（cap）与 ballot 环核完全一致：
 *   登记前先 atomicAdd 领槽再判界，pos 单调递增，所以 [0, cap) 里的每个槽都恰好被写
 *   一次（无空洞、无覆盖），pos >= cap 的命中直接丢弃并让本 block 立刻停手。于是
 *   「写 d_results 越界」在结构上不可能发生，跟主机端批次尺寸估计对不对无关。
 *   缓冲区满时主机丢弃本轮的全部结果、缩小 chunk 重试，所以丢弃的命中不会丢结果。
 */
template <int K>
__global__ void scanRectFusedKernelCount(
    const int64_t* __restrict__ d_baseX,
    const int64_t* __restrict__ d_baseZ,
    int width, int blockHeight, int baseZ_offset,
    int sizeX, int sizeZ, int threshold,
    int startX, int baseZ,
    int colOffset, int validRangeX,
    HitResult* d_results, unsigned long long* d_pos, unsigned long long cap,
    int zSubRows) {  // 每个 Z 子块负责的候选行数（= ceil(chunk 候选行 / gridDim.y)）
    constexpr int TILE_W = WARPS_PER_BLOCK * WARP_SIZE * K;  // 一个 block 覆盖的输入列数
    static_assert(K >= 1, "K must be positive");
    // 计数环：跨 warp halo 只剩「下一个 warp 的首字」，但**必须按行入环**：
    // 若写成单槽 s_halo[warp]，跑在前面的 warp w+1 会在第 r+1 行把该槽改写成下一行的
    // 首字，而 warp w 可能还没读完第 r 行的值 —— 这是一个真竞争。实测证据：±200000 负载上
    // 单槽写法在 K=5 时出现 ±1 的窗口计数错误，且两次运行结果互不相同。
    // 环深取 4（2 的幂，1 条 LOP3 取模）——**这是一个可证明的充分深度，不是经验值**：
    //   行 r 的读发生在 barrier(r) 之后；写 halo(r+1) 的 warp 不需要穿过任何 barrier，
    //   所以 slot(r+1) != slot(r) 就是充分条件（D>=2）；
    //   要在 r+2 行写，必须先穿过 barrier(r+1)；而任何线程到达 barrier(r+1) 之前
    //   一定已经执行过第 r 行的 halo 读（程序序：读 halo(r) -> … -> 写 halo(r+1) -> barrier(r+1)）
    //   ⇒ 全 block 都已读完 halo(r) 之后才可能出现 halo(r+2) 的写，而此时 slot(r+2)
    //   与 slot(r) 的关系已无关紧要。slot(r+2)==slot(r) 只在 D|2 时成立，故 D>=3 即可；
    //   取 4（144 B）既满足充分性又能用 & 取模。
    // 行距 WARPS_PER_BLOCK+1，索引 WARPS_PER_BLOCK 是哨兵 0（只有最后一个 warp 的
    // k==K-1 会读到它，而会读到它的 lane 恰好被 xmaskLane=0 关掉）。
    constexpr int HALO_RING = 4;
    __shared__ uint32_t s_halo[HALO_RING * (WARPS_PER_BLOCK + 1)];
    // 计数环：每行每列的 sizeX 窗口计数环，s_ws[slot*TILE_W + col]，slot = r mod sizeZ。
    // 环深 = sizeZ；读旧写新同槽（读在写前）；环是 warp 私有的（每 lane 只碰自己的列）
    // ⇒ 无竞争、无需额外 barrier。计数 ∈ [0, sizeX] ≤ 32，uint8 逐位无损。
    extern __shared__ uint8_t s_ws[];

    const int tx = threadIdx.x;
    const int lane = tx & 31;
    const int warp = tx >> 5;
    const int wordBase = warp * K;
    const int ownOff = wordBase * WARP_SIZE + lane;  // 本 lane 的列在「行」内的字节偏移
    const int haloWarpBase = warp * 4;              // 本 warp 的 halo 槽在行内槽区的字节偏移

    // 本 block 负责的候选行区间 [subBaseZ, subBaseZ + subRows)（blockHeight 是 chunk 的
    // 输入行数，所以 chunk 的候选行数 = blockHeight - sizeZ + 1）。各 block 的输出行区间
    // 严格相邻且不重叠；输入多读 sizeZ-1 行 halo，只影响 ncu 里的指令数不影响结果。
    const int chunkCandRows = blockHeight - sizeZ + 1;
    const int subStart = blockIdx.y * zSubRows;
    const int subRows = min(zSubRows, chunkCandRows - subStart);
    const int subInputRows = subRows + sizeZ - 1;
    const int subBaseZ = baseZ + subStart;
    const int subZOffset = baseZ_offset + subStart;

    // 哨兵：每一行的 s_halo[(r&63)*(WARPS_PER_BLOCK+1) + WARPS_PER_BLOCK] 恒为 0。写只落在
    // 行内 [0, WARPS_PER_BLOCK)，整块清零覆盖全部 64*(WARPS_PER_BLOCK+1) 个字，所以哨兵
    // 整个 kernel 生命周期都是 0。
    for (int i = tx; i < HALO_RING * (WARPS_PER_BLOCK + 1); i += blockDim.x) s_halo[i] = 0u;

    const int outW = TILE_W - sizeX + 1;
    const int colBase = blockIdx.x * outW;  // 本 block 的输入列起点（= 候选列起点）
    const uint32_t xmask = (sizeX >= 32) ? 0xFFFFFFFFu : ((1u << sizeX) - 1u);

    // 每线程 K 个跨步列：cl = (warp*K + k)*32 + lane  ->  warp 内 K 次 ballot 全在寄存器
    int64_t bx[K];
    bool outA[K];
    uint32_t xmaskLane[K];
    int cnt[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int cl = (wordBase + k) * WARP_SIZE + lane;  // block 内列
        const int clLocal = colBase + cl;                  // 批次内列
        const int gcol = colOffset + clLocal;              // 全局列
        outA[k] = (cl < outW) && (clLocal < validRangeX);
        // d_baseX 尾部有 BASE_X_PAD 个 0，越界列可以无条件读（值必被掩码丢掉）
        bx[k] = __ldg(d_baseX + gcol);
        // 越界列的窗口掩码取 0 => 该 lane 的窗口计数恒为 0 => cnt[k] 保持毒值
        xmaskLane[k] = outA[k] ? xmask : 0u;
        cnt[k] = outA[k] ? 0 : POISON_CNT;
    }
    unsigned stop = 0u;  // R5: 用 unsigned 而不是 bool，避免 ptxas 把它按字节打包/解包
    // 行循环拆两段（汇编审计第 2 条 + 建模 Agent 第 ② 条）：
    //   前置段 r ∈ [0, sizeZ-1)：窗口还没攒满 —— 只累加本行，不减旧行、不发射；
    //   主段   r ∈ [sizeZ-1, subInputRows)：稳态 —— 无条件减旧行 + 发射闸门，
    //          于是「r >= sizeZ」「r >= sizeZ-1」这两个每行都要判一次的单调条件整个消失。
    // 前置段之所以能"无条件减旧行"，是因为计数环**整块预清零**：r < sizeZ 时读到的
    // 槽恰是尚未写过的预清零槽 ⇒ old = 0，正是「sizeZ 行前那一行越界、窗口计数恒 0」
    // 的语义。环深 = sizeZ 且读在写前，所以 r ≥ sizeZ 时读到的必然是第 r-sizeZ 行那一槽
    // （坏槽必然尚未写过 ⇒ 读到 0 ⇒ 等价于「窗口在 sizeZ 行前没攒满」，sizeZ=1/32 亦然）。
    {
        uint32_t* __restrict__ p = (uint32_t*)s_ws;   // TILE_W 是 32 的倍数 ⇒ 总字节数是 4 的倍数
        const int n = (sizeZ * TILE_W) >> 2;
        for (int i = tx; i < n; i += blockDim.x) p[i] = 0u;
    }
    int slot = 0;  // 行 r 的环槽 = r mod sizeZ。整数 slot 索引 + 循环内重算地址，
                   // ptxas 才会把它放进 uniform 通路（指针绕回写法实测多 5 条 ALU/行）。
    __syncthreads();
    const int peel = (sizeZ - 1 < subInputRows) ? sizeZ - 1 : subInputRows;
    // R1: bz 的行地址用指针归纳变量 —— d_baseZ + subZOffset 只在循环外算一次，
    //     之后每行只做一次 8 字节自增（uniform 通路），不再从 r 重新算一遍。
    const int64_t* __restrict__ bzPtr = d_baseZ + subZOffset;
    for (int r = 0; r < peel; ++r) {
        const int64_t bz = __ldg(bzPtr++);  // R1: 指针归纳变量，每行只自增 8 字节
        uint32_t ballots[K];
        bool rejAny = false;  // 本行 K 个格子里是否有 lane 落进 Java 的拒绝区间
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t bits = slimeBitsFast(bx[k], bz);
            rejAny = rejAny || (bits >= SLIME_REJECT);
            ballots[k] = __ballot_sync(0xFFFFFFFFu, slimeTestBits(bits));
        }
        // 概率 ~4e-9：整行只开一个 warp 均匀门控（VOTE.ANY -> 均匀谓词 -> @!P0 BRA）。
        // 均匀分支不划发散区，所以热路径里 4 对 BSSY/BSYNC + 4 条谓词 BRA 全部消失；
        // 慢路径是出线 CALL，只在该分支里出现。
        if (__builtin_expect(__any_sync(0xFFFFFFFFu, rejAny) != 0, 0)) {
#pragma unroll
            for (int k = 0; k < K; ++k) ballots[k] = slimeBallotSlow(bx[k], bz);
        }
        // halo：每 warp 一个 uint32 槽（ballot 对全 lane 同值 ⇒ 去掉 lane==0 谓词、
        // 无条件 STS 是确定性的：多 lane 写同一地址时 CUDA 保证其中一个胜出，而值相同）。
        // 行内槽距 (WARPS_PER_BLOCK+1)*4 字节，本 warp 的槽基址 = haloWarpBase。
        // 实测（ncu，同区域）：稳态循环体 135 -> 127 条/行，inst/候选 29.25 -> 28.73，
        // kernel 时长 56.77 -> 55.46 ms。
        const int hoff = (r & (HALO_RING - 1)) * ((WARPS_PER_BLOCK + 1) * 4) + haloWarpBase;
        *(uint32_t*)((char*)s_halo + hoff) = ballots[0];  // 本行本 warp 的首字入行槽
        __syncthreads();  // 前置段不可能置 stop，用普通 barrier 即可
        const uint32_t hw = *(const uint32_t*)((const char*)s_halo + hoff + 4);
        uint8_t* const wp = s_ws + slot * TILE_W + ownOff;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t hi = (k + 1 < K) ? ballots[k + 1] : hw;
            const int ws = __popc(__funnelshift_r(ballots[k], hi, lane) & xmaskLane[k]);
            wp[k * WARP_SIZE] = (uint8_t)ws;   // 环里存的是「本行本列的窗口计数」
            cnt[k] += ws;
        }
        if (++slot == sizeZ) slot = 0;
    }
    for (int r = peel; r < subInputRows; ++r) {
        const int64_t bz = __ldg(bzPtr++);  // R1: 指针归纳变量，每行只自增 8 字节
        uint32_t ballots[K];
        bool rejAny = false;  // 本行 K 个格子里是否有 lane 落进 Java 的拒绝区间
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t bits = slimeBitsFast(bx[k], bz);
            rejAny = rejAny || (bits >= SLIME_REJECT);
            ballots[k] = __ballot_sync(0xFFFFFFFFu, slimeTestBits(bits));
        }
        // 概率 ~4e-9：整行只开一个 warp 均匀门控（VOTE.ANY -> 均匀谓词 -> @!P0 BRA）。
        // 均匀分支不划发散区，所以热路径里 4 对 BSSY/BSYNC + 4 条谓词 BRA 全部消失；
        // 慢路径是出线 CALL，只在该分支里出现。
        if (__builtin_expect(__any_sync(0xFFFFFFFFu, rejAny) != 0, 0)) {
#pragma unroll
            for (int k = 0; k < K; ++k) ballots[k] = slimeBallotSlow(bx[k], bz);
        }
        const int hoff = (r & (HALO_RING - 1)) * ((WARPS_PER_BLOCK + 1) * 4) + haloWarpBase;
        *(uint32_t*)((char*)s_halo + hoff) = ballots[0];  // 本行本 warp 的首字入行槽
        // ③ 这同一个 barrier 兼两用：给 halo 读建立可见性 + 把"缓冲区满"折成块均匀值。
        //    于是行循环的退出是均匀的（不再有按线程 break），编译器才有机会把行计数器、
        //    环槽地址、比较这些标量搬到 uniform 数据通路上。
        if (__syncthreads_or((int)stop)) break;
        const uint32_t hw = *(const uint32_t*)((const char*)s_halo + hoff + 4);
        uint8_t* const wp = s_ws + slot * TILE_W + ownOff;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            // 本行的 halo 字：k < K-1 时它就是本 warp 自己的 ballots[k+1]（__ballot_sync 对
            // 全 lane 返回同一掩码，人人手上都有），只有 k == K-1 那个字属于下一个 warp、
            // 才必须读 shared。
            const uint32_t hi = (k + 1 < K) ? ballots[k + 1] : hw;
            const int ws = __popc(__funnelshift_r(ballots[k], hi, lane) & xmaskLane[k]);
            const int old = wp[k * WARP_SIZE];   // 先读：= sizeZ 行前那一行同列的窗口计数
            wp[k * WARP_SIZE] = (uint8_t)ws;     // 后写：同槽（环深 = sizeZ）
            cnt[k] += ws - old;                  // 折成一条三操作数 IADD3
        }
        if (++slot == sizeZ) slot = 0;
        {
            const int z = subBaseZ + r - (sizeZ - 1);
#ifdef SLIME_SCAN_ONLY
            // 只保留命中的数据依赖（防止整段被优化掉），不做任何登记：
            // 无 atomicAdd、无 d_results 的 12 字节写入、无 D2H 拷贝、无 CSV 行
            (void)z;
#pragma unroll
            for (int k = 0; k < K; ++k)
                if (outA[k] && cnt[k] >= threshold) ((int*)d_results)[tx] = cnt[k];
#else
            // ① 每行只开一个 warp 均匀的发射闸门：先把 K 个命中谓词归约成一个布尔，再用
            //    __any_sync 变成 warp 均匀值。均匀分支不需要 BSSY/BSYNC 划发散区，所以
            //    常见情形（本行无任何命中）只用一条 BRA 就跳过整段登记代码 —— 而原来的
            //    写法是每个 k 各一对 BSSY/BSYNC + 谓词 BRA（K=4 时每行 ~40 条簿记）。
            // 越界列被毒值保护，所以只需要「最大值是否够阈值」；K-1 条 IMNMX + 1 条 ISETP
            int mx = cnt[0];
#pragma unroll
            for (int k = 1; k < K; ++k) mx = (mx > cnt[k]) ? mx : cnt[k];
            bool anyHit = (mx >= threshold);
            if (__any_sync(0xFFFFFFFFu, anyHit)) {
#pragma unroll
                for (int k = 0; k < K; ++k) {
                    if (outA[k] && cnt[k] >= threshold) {
                        const unsigned long long pos = atomicAdd(d_pos, 1ULL);
                        if (pos >= cap) {  // 先判界再写：越界写在结构上不可能发生
                            stop = 1u;     // 缓冲区满 -> 停手，主机丢弃本轮并缩小 chunk 重试
                            break;
                        }
                        d_results[pos].x = (startX + colBase + (wordBase + k) * WARP_SIZE + lane) * 16;
                        d_results[pos].z = z * 16;
                        d_results[pos].count = cnt[k];
                    }
                }
            }
#endif
        }
    }
}

/*
 * merge4 融合扫描（T1 v2：4 邻列共享松上界 + 过门精确重算）—— 默认路径。
 * 启用条件（host 侧守卫）：sizeX <= MAX_MERGE4_SIZE_X(29) 且 K != 1；其余形态仍走
 * scanRectFusedKernelCount（sizeX ∈ [30,32]）或
 * scanRectFusedKernelBallot（sizeZ 超过运行期派生的上限 / K == 1）。
 * 它只用静态 shared（9216 B = RING*STRIDE*4，与 sizeZ 无关），所以不受计数环那个
 * sizeZ 上限的约束。
 *
 * 与 scanRectFusedKernelBallot 的唯一差别是「X 方向的滚动记账」：
 *   旧核每线程负责 K 个跨步列，逐列维护 sizeX 宽窗口的滚动和 cnt[k]，每候选每行要付
 *   SHF+LOP3+POPC（本行窗口）+ LDS.U8+STS.U8+IADD3（计数环）≈ 6 条；
 *   merge4 改成「每 lane 负责 4 个相邻候选列」（gx = 4*tx），热路径只维护**一个**
 *   bundle 级松上界 U = sizeZ 行 × (sizeX+3) 列并集内的史莱姆数：
 *       U += popc(本行并集位段);   U -= popc(sizeZ 行前那一行的并集位段);
 *   每 bundle 每行 2 次 (2×LDS + SHF + LOP3 + POPC + IADD3)，与候选数无关。
 *
 * 上界（P1）：候选 c 的窗口 [gx+c, gx+c+sizeX-1] ⊆ 并集 [gx, gx+sizeX+2]（c <= 3），
 *   两者行区间相同、计数非负 ⇒ 精确值 E(c) <= U。于是 U < threshold 的 bundle 里不可能
 *   有命中。**闸门必须写 U >= threshold**：写成 U > threshold 会漏掉 E == threshold 的命中。
 *
 * 过门重算（P3）：U >= threshold 时，对该 bundle 的 4 个候选从 **shared 行位图环**按需
 *   重算精确 sizeZ × sizeX 和（每候选每行 SHF+LOP3+POPC）。热路径不保留任何 per-候选
 *   计数环 —— 那正是本方案要删掉的那次 popc，保留它等于把收益买回去。
 *
 * 读侧映射（一个 lane 一个 bundle，与写侧 ballot 的跨步列映射无关）：
 *       gx = 4*tx;   bword = tx>>3;   bshift = (tx&7)*4
 *   即 8 个 lane 共用同一个行位图字、4 位对齐。行位图本身仍是标准布局：
 *   ballot k 的 bit L = 列 (wordBase+k)*32 + L ⇒ 行位图字 w 覆盖列 [32w, 32w+32)。
 *   位段起点 = gx mod 32 = 4*(tx&7) = bshift，字索引 = gx >> 5 = tx>>3 = bword。
 *
 * 几何：B = ceil(outW/4) 个 bundle 覆盖 [0, outW)，最后一个 bundle 允许越过 outW
 *   （越界候选被 cmask 挡在登记之外 ⇒ 不重不漏）。并集位段越过 tile 右端时读到哨兵字 0
 *   （s_rows[...][WORDS] 恒 0）⇒ 并集被 tile 边界截断。这不是错误：有效候选满足
 *   gx+c < outW ⇒ 窗口右端 gx+c+sizeX-1 <= TILE_W-1，被截掉的列不属于任何有效候选的
 *   窗口，只会让 U 更紧（也保证不会读到 tile 外的数据）。
 *
 * 抽取宽度：并集位段 [bshift, bshift+sizeX+3) ≤ [28, 28+32) = [28,60) ⊆ 相邻两字的 64 位
 *   ⇒ 永远不需要第三个字；单次 32 位抽取是 sizeX <= 29 的真正来源。
 *
 * 越界写的结构性防护（cap）与另外两个融合核完全一致：先 atomicAdd 领槽再判界，
 *   pos 单调递增 ⇒ [0,cap) 每槽恰好写一次；pos >= cap 立即停手，主机丢弃整轮并重试。
 */
template <int K>
__global__ void scanRectFusedMerge4Kernel(
    const int64_t* __restrict__ d_baseX,
    const int64_t* __restrict__ d_baseZ,
    int width, int blockHeight, int baseZ_offset,
    int sizeX, int sizeZ, int threshold,
    int startX, int baseZ,
    int colOffset, int validRangeX,
    HitResult* d_results, unsigned long long* d_pos, unsigned long long cap,
    int zSubRows) {  // 每个 Z 子块负责的候选行数（= ceil(chunk 候选行 / gridDim.y)）
    static_assert(K == 4, "merge4 的一个 lane 管一个 4 邻列 bundle，要求 K == 4");
    constexpr int TILE_W = WARPS_PER_BLOCK * WARP_SIZE * K;  // 1024：一个 block 覆盖的输入列数
    constexpr int WORDS = WARPS_PER_BLOCK * K;               // 32：每行位图的字数
    constexpr int STRIDE = ((WORDS + 4) / 4) * 4;            // 36：16B 对齐 ⇒ 一条 STS.128
    constexpr int RING = 64;                                 // 必须 > 最大 sizeZ
    static_assert(RING > MAX_SIZE_Z, "ring must be larger than the largest sizeZ");
    static_assert(WORDS % 4 == 0, "STS.128 要求每 warp 的字数是 4 的倍数");
    __shared__ uint32_t s_rows[RING * STRIDE];

    const int tx = threadIdx.x;
    const int lane = tx & 31;
    const int warp = tx >> 5;
    const int wordBase = warp * K;

    // 本 block 负责的候选行区间 [subBaseZ, subBaseZ + subRows)（与另外两个融合核同构）
    const int chunkCandRows = blockHeight - sizeZ + 1;
    const int subStart = blockIdx.y * zSubRows;
    const int subRows = min(zSubRows, chunkCandRows - subStart);
    const int subInputRows = subRows + sizeZ - 1;
    const int subBaseZ = baseZ + subStart;
    const int subZOffset = baseZ_offset + subStart;

    // 哨兵：索引 WORDS 恒为 0（warp 7 的 lane >= 24 会读到它；STRIDE=36 > WORDS 故在界内）
    for (int i = tx; i < RING; i += blockDim.x) s_rows[i * STRIDE + WORDS] = 0u;

    const int outW = TILE_W - sizeX + 1;
    const int colBase = blockIdx.x * outW;

    // ---- 读侧：每 lane 一个 4 邻列 bundle ----
    const int gx = tx * 4;            // bundle 基准列（block 内输入列 = 候选列）
    const int bword = tx >> 3;        // 行位图字索引（0..WORDS-1）
    const int bshift = (tx & 7) * 4;  // 位段起点（0,4,...,28）

    // 4 个候选的活性：循环不变量（outW / validRangeX 在行循环内不变）⇒ 循环外算一次。
    // 越界候选不登记：既保证不重（下一个 tile 会负责）也不漏（覆盖引理）。
    unsigned cmask = 0u;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int cl = gx + c;
        if (cl < outW && colBase + cl < validRangeX) cmask |= (1u << c);
    }

    // ---- 写侧：RNG + ballot（与旧核完全相同的「每线程 K 个跨步列」映射）----
    int64_t bx[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int cl = (wordBase + k) * WARP_SIZE + lane;
        // d_baseX 尾部有 BASE_X_PAD 个 0，越界列可以无条件读（其位不属于任何有效候选窗口）
        bx[k] = __ldg(d_baseX + colOffset + colBase + cl);
    }

    // 并集位段掩码（宽 sizeX+3 <= 32）与候选窗口掩码（宽 sizeX <= 29）
    const uint32_t umask = (sizeX + 3 >= 32) ? 0xFFFFFFFFu : ((1u << (sizeX + 3)) - 1u);
    const uint32_t xmask = (1u << sizeX) - 1u;

    unsigned stop = 0u;  // 与旧核一致：用 unsigned 而不是 bool，避免按字节打包/解包
    int U = 0;           // bundle 级松上界：sizeZ 行 × (sizeX+3) 列并集内的史莱姆数

    // 环整块预清零：peel 段「无条件减 sizeZ 行前那一行」靠它读到 0（该槽此刻必未写过），
    // 于是「窗口还没攒满」这段单调条件整个消失。负向对照：不预清零会大面积漏报/误报。
    for (int i = tx; i < RING * STRIDE; i += blockDim.x) s_rows[i] = 0u;
    __syncthreads();

    const int peel = (sizeZ - 1 < subInputRows) ? sizeZ - 1 : subInputRows;
    // 老行槽位的独立归纳变量：slotOld ≡ (r - sizeZ) (mod RING)。
    // 原写法每行要 ULDC 取 sizeZ、一次减法、再一次 AND；归纳变量每行只需一次条件自增
    //（RING 是 2 的幂，uint 回绕 + &(RING-1) 与 mod 等价）。
    int slotOld = (peel - sizeZ) & (RING - 1);
    // R1: bz 的行地址用指针归纳变量（只在循环外算一次）
    const int64_t* __restrict__ bzPtr = d_baseZ + subZOffset;
    for (int r = 0; r < peel; ++r) {
        const int64_t bz = __ldg(bzPtr++);
        uint32_t ballots[K];
        bool rejAny = false;  // 本行 K 个格子里是否有 lane 落进 Java 的拒绝区间
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t bits = slimeBitsFast(bx[k], bz);
            rejAny = rejAny || (bits >= SLIME_REJECT);
            ballots[k] = __ballot_sync(0xFFFFFFFFu, slimeTestBits(bits));
        }
        // 概率 ~4e-9：整行只开一个 warp 均匀门控；慢路径是出线 CALL
        if (__builtin_expect(__any_sync(0xFFFFFFFFu, rejAny) != 0, 0)) {
#pragma unroll
            for (int k = 0; k < K; ++k) ballots[k] = slimeBallotSlow(bx[k], bz);
        }
        if (lane == 0) {  // 一条 STS.128 取代 4 条谓词化 STS.32
            *(uint4*)&s_rows[(r & (RING - 1)) * STRIDE + wordBase] =
                make_uint4(ballots[0], ballots[1], ballots[2], ballots[3]);
        }
        __syncthreads();  // 前置段不可能置 stop，用普通 barrier 即可
        const uint32_t* row = s_rows + (r & (RING - 1)) * STRIDE;
        // 并集 = 本行 4 个候选窗口之并；位段越界部分被哨兵字 0 截断
        U += __popc(__funnelshift_r(row[bword], row[bword + 1], bshift) & umask);
    }
    for (int r = peel; r < subInputRows; ++r) {
        const int64_t bz = __ldg(bzPtr++);
        uint32_t ballots[K];
        bool rejAny = false;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const uint32_t bits = slimeBitsFast(bx[k], bz);
            rejAny = rejAny || (bits >= SLIME_REJECT);
            ballots[k] = __ballot_sync(0xFFFFFFFFu, slimeTestBits(bits));
        }
        if (__builtin_expect(__any_sync(0xFFFFFFFFu, rejAny) != 0, 0)) {
#pragma unroll
            for (int k = 0; k < K; ++k) ballots[k] = slimeBallotSlow(bx[k], bz);
        }
        if (lane == 0) {
            *(uint4*)&s_rows[(r & (RING - 1)) * STRIDE + wordBase] =
                make_uint4(ballots[0], ballots[1], ballots[2], ballots[3]);
        }
        // ③ 这同一个 barrier 兼两用：给并集/重算的 shared 读建立可见性 + 把"缓冲区满"
        //    折成块均匀值。于是行循环的退出是均匀的，编译器才有机会把行计数器、环槽地址、
        //    比较这些标量搬到 uniform 数据通路上。
        // 可见性只需普通 barrier（1 条 BAR.SYNC）。`stop` 不必每行检查：
        //   缓冲区满时 d_results 的写入有 `pos >= cap` 判界硬保护（结构上不可能越界写），
        //   晚几行退出只多做一点无用功、不影响正确性；而 `stop` 一旦置位不复位，
        //   循环迟早退出。⇒ 省掉 sm_89 上 `__syncthreads_or` 的 5 条指令序列，
        //   代价是每行 1 条 BAR.SYNC；`stop` 改为在被 `__any_sync` 门控的冷路径里检查。
        __syncthreads();
        const uint32_t* row = s_rows + (r & (RING - 1)) * STRIDE;
        U += __popc(__funnelshift_r(row[bword], row[bword + 1], bshift) & umask);
        // 减 sizeZ 行前那一行：环深 64 ∤ sizeZ ⇒ 该槽与刚写的本行槽不同（读在写后也不冲突）
        const uint32_t* old = s_rows + slotOld * STRIDE;
        if (++slotOld == RING) slotOld = 0;   // 与本行槽同步推进
        U -= __popc(__funnelshift_r(old[bword], old[bword + 1], bshift) & umask);
        // 闸门：U >= threshold（必须 >=，写 > 会漏 exact == threshold）。
        // 用 warp 均匀的 __any_sync 开闸：常见情形（本行没有任何 bundle 过门）一条 BRA 跳过。
        // 注意闸门里**不**再与 cmask 相与：cmask == 0 的越界 bundle 白重算一次也无妨
        // （cmask 在登记处仍然把关），但每行少一条谓词合成 —— 实测热路径 105 -> 104 条。
        const bool pass = (U >= threshold);
        if (__any_sync(0xFFFFFFFFu, pass)) {
            if (stop) break;   // 缓冲区满：在冷路径（过门率 ~1e-5）里退出
            if (pass) {
                // ---- 过门：从行位图环按需重算 4 个候选的精确 sizeZ × sizeX 和 ----
                int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                for (int j = 0; j < sizeZ; ++j) {
                    const uint32_t* rj = s_rows + ((r - j) & (RING - 1)) * STRIDE;
                    const uint32_t lo = rj[bword], hi = rj[bword + 1];
                    e0 += __popc(__funnelshift_r(lo, hi, bshift) & xmask);
                    e1 += __popc(__funnelshift_r(lo, hi, bshift + 1) & xmask);
                    e2 += __popc(__funnelshift_r(lo, hi, bshift + 2) & xmask);
                    e3 += __popc(__funnelshift_r(lo, hi, bshift + 3) & xmask);
                }
#ifdef SLIME_SCAN_ONLY
                // 只保留命中的数据依赖（防止整段被优化掉），不做任何登记
                (void)cmask;
                const int keep = e0 + e1 + e2 + e3;
                if (keep >= threshold) ((int*)d_results)[tx] = keep;
#else
                const int e[4] = {e0, e1, e2, e3};
                const int z = subBaseZ + r - (sizeZ - 1);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (((cmask >> c) & 1u) && e[c] >= threshold) {
                        const unsigned long long pos = atomicAdd(d_pos, 1ULL);
                        if (pos >= cap) {  // 先判界再写：越界写在结构上不可能发生
                            stop = 1u;     // 缓冲区满 -> 停手，主机丢弃本轮并缩小 chunk 重试
                            break;
                        }
                        d_results[pos].x = (startX + colBase + gx + c) * 16;
                        d_results[pos].z = z * 16;
                        d_results[pos].count = e[c];
                    }
                }
#endif
            }
        }
    }
}


/*
 * ---- 并行度阈值的普适化（2026-10-04）----
 *
 * 下面三个阈值原本都是**按本机这张卡（34 SM × 6 常驻 block = 204 常驻槽）标定**的：
 *   * `FUSED_K4_MIN_BLOCKS = 408 = 2 × 204`（两个波形）
 *   * `Z_SPLIT_MIN_TILES  = 2048 ≈ 10 × 204`（十来个波形，波形量化损失 <1%）
 *   * `FUSED_K_MIN_COLS   = 200000 ≈ 204 × outW4(1008) = 205632`（"光靠 X 就能填满一个波形"）
 *
 * 在 128 SM 的大卡上，这三个判据相当于被"放宽 3.8 倍"：宽 X + 短 Z 的负载本可以填满 GPU，
 * 却被判成"不值得用 K=4"而回退到 K=1（推算最坏欠填到 0.34 个波形）。
 *
 * 因此改成**从真实常驻槽数派生**（`slots = SM 数 × 每个 SM 的常驻 block 数`），
 * 编译期常量退化为**下界**：
 *   * 本机（204 槽）：`2×204=408`、`10×204=2040`（与原 2048 同波形档）、`200000×204/204=200000`
 *     ⇒ **与改动前逐位相同**（`-arch=sm_89` 编译比对：6 个内核的 SASS 逐函数 sha256 全同）；
 *   * 大卡：阈值随 SM 数放大，选路自动跟上。
 *
 * `blocksPerSm` 用 occupancy API 查**真实的 K=4 核**（256 线程、静态 shared 9216 B），
 * 而不是写死 6 —— 换编译器/架构后寄存器一变（本仓有过 40 → 48 的记录），常驻 block 数会跟着变。
 * 查询失败或 `SLIME_NO_AUTOTUNE=1` 时回退到编译期常量（逃生开关）。
 */
static int g_k4MinBlocks   = FUSED_K4_MIN_BLOCKS;   // 低于此 block 数就不用 K=4
static int g_zSplitMinTile = Z_SPLIT_MIN_TILES;     // 低于此 tile 数就不切 Z
static int g_k4MinTiles    = FUSED_K_MIN_COLS / (WARPS_PER_BLOCK * WARP_SIZE * FUSED_K);
static int g_smUsed = 0, g_bpsmUsed = 0;   // 仅供打印
static int g_cntRingMaxSizeZ = COUNT_RING_MAX_SIZE_Z;   // 运行期按占用率交叉点派生
// 仅供测试钩子：模拟别的卡的 shared/SM 与 per-block 上限（0 = 用真实值）
static int g_fakeSharedPerSm = 0, g_fakeSharedPerBlock = 0;

// 计数环核在 sizeZ 下的动态 shared（= sizeZ × TILE_W 字节），与 launch 时的字节数一致。
static size_t cntRingDynBytes(int sizeZ) {
    return (size_t)sizeZ * (WARPS_PER_BLOCK * WARP_SIZE * FUSED_K);
}

/*
 * ---- COUNT_RING_MAX_SIZE_Z 的运行期交叉点（2026-10-04）----
 *
 * 阈值本来写死 20，但那是「本机 34 SM × 6 blocks、shared/SM = 100 KB」下的标定值：
 * 计数环核的动态 shared 随 sizeZ 线性增长，某个 sizeZ 之后它会把 blocks/SM 压低，
 * 于是「计数环省指令」的收益被占用率塌陷吃掉。实测本机（ncu 占用率，纯结构属性）：
 *     sizeZ=17 → 5 blocks/SM (83.2%)   sizeZ=18 → 5 blocks/SM (83.2%)
 *     sizeZ=19 → 4 blocks/SM (66.6%)   sizeZ=20 → 4 blocks/SM (66.6%)
 * ⇒ 交叉点在 18/19 之间，写 20 会让 19/20 走进已知更慢的档（同档的 sizeZ=24 实测 +3.9%）。
 *
 * 这个交叉点**因卡而异**：H100（shared/SM 228 KB）即便 sizeZ=32 也不塌陷 ⇒ 交叉点 = 32；
 * sm_75（64 KB/SM）更早。所以按设备算，而不是写死。
 *
 * **实测提醒（2026-10-04，本机 ncu cycles 交错 A/B，sizeZ=16..20）**：在这 5 个 sizeZ 上
 * 两个核**基本打平**（计数环在 16/20 快 0.7%/3.5%，在 17/18/19 慢 1.2%/1.7%/1.5%，
 * 方向不一致且都在 ±2% 噪声内）。也就是说：**占用率下滑并没有转化成可测的变慢**，
 * 「阈值 20 落在塌陷档所以应该改小」这个推论在本机得不到实测支持。
 * 计数器口径倒是稳定的：计数环始终少 3~5% 指令（238.9M vs 246.1M @sizeZ=20），
 * 但与本仓反复测到的「occupancy 弹性 ≈0.19」一致 —— **省下的指令几乎换不到时间**。
 * 所以本函数的价值是**换卡正确**（H100 拿 32、sm_75 拿 9），而不是让本机更快；
 * 本机上它与原常量 20 等价（都落在打平区）。
 *
 * 判据：取「blocks/SM 仍不低于 sizeZ=1 时」的最大 sizeZ。
 * 保守性：CUDA 对动态 shared 的默认上限（48 KB）小于实际 launch 的字节数时，
 *   驱动会在真正 launch 时报错；这里把搜索限制在默认上限之内，保证派生的值一定可 launch。
 *   （实际 launch 前仍会按需 cudaFuncSetAttribute，见 GPUWorker::run。）
 */
static int computeCntRingCrossover(int fakeSharedPerSm, int fakeSharedPerBlock) {
    if (getenv("SLIME_NO_AUTOTUNE")) return COUNT_RING_MAX_SIZE_Z;
    // 动态 shared 的默认上限（超过它的 sizeZ 不作为候选，避免派生出一个 launch 不了的值）
    int maxDynShared = 48 * 1024;
    int curDev = 0;
    if (cudaGetDevice(&curDev) != cudaSuccess) { (void)cudaGetLastError(); curDev = 0; }
    int v = 0;
    if (fakeSharedPerBlock > 0) {
        // 测试钩子：模拟别的卡的 per-block 上限，并据此模拟 occupancy 曲线。
        // sm_89 实测：6 blocks/SM 需要 bytes <= 15360（每 block 约 1 KB 保留）；
        // 按比例换算到假卡的 per-SM shared 上限即可复现该曲线（本机验过与 API 逐点一致）。
        const int maxBlocks = 6;
        int best = 0;
        for (int sizeZ = 1; sizeZ <= MAX_SIZE_Z; ++sizeZ) {
            const size_t bytes = cntRingDynBytes(sizeZ);
            if (bytes > (size_t)fakeSharedPerBlock) break;
            const int b = fakeSharedPerSm / ((int)bytes + 1024);
            if (b >= maxBlocks) best = sizeZ; else break;
            (void)b;
        }
        return best > 0 ? best : 1;
    }
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlock, curDev) == cudaSuccess && v > 0)
        maxDynShared = v;
    else
        (void)cudaGetLastError();
    int base = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &base, scanRectFusedKernelCount<FUSED_K>, WARPS_PER_BLOCK * WARP_SIZE,
            cntRingDynBytes(1)) != cudaSuccess || base <= 0) {
        (void)cudaGetLastError();
        return COUNT_RING_MAX_SIZE_Z;      // 查询失败：退回编译期常量
    }
    int best = 1;
    for (int sizeZ = 1; sizeZ <= MAX_SIZE_Z; ++sizeZ) {
        const size_t bytes = cntRingDynBytes(sizeZ);
        if (bytes > (size_t)maxDynShared) break;   // 超出默认动态 shared 上限，不再作候选
        int b = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &b, scanRectFusedKernelCount<FUSED_K>, WARPS_PER_BLOCK * WARP_SIZE, bytes)
            != cudaSuccess || b <= 0) {
            (void)cudaGetLastError();
            break;
        }
        if (b >= base) best = sizeZ;       // 仍保持最大占用率 ⇒ 计数环核仍然划算
        else break;                        // 占用率开始下滑 ⇒ 交叉点到此为止
    }
    return best;
}

// 在**当前 CUDA 设备上下文**里查真实值并重设阈值。
// 必须在 `cudaSetDevice(device_id)` 之后、真正 launch 之前调用 —— 每个 worker 各自调一次，
// 这样多卡时每个 worker 用的是**它自己那张卡**的 SM 数与占用率（早期版本在 main 的枚举
// 循环里调用，globals 会被最后一个设备覆盖，多卡时会用错卡）。
static void tuneParallelismThresholds() {
    if (getenv("SLIME_NO_AUTOTUNE")) {         // 逃生开关：退回改动前的编译期常量
        g_k4MinBlocks   = (FUSED_K4_MIN_BLOCKS == SLIME_K4_MIN_BLOCKS_AUTO) ? 408 : FUSED_K4_MIN_BLOCKS;
        g_zSplitMinTile = Z_SPLIT_MIN_TILES;
        g_k4MinTiles    = FUSED_K_MIN_COLS / (WARPS_PER_BLOCK * WARP_SIZE * FUSED_K);
        g_cntRingMaxSizeZ = COUNT_RING_MAX_SIZE_Z;
        return;
    }
    // 取**当前设备**（不是 device 0、也不是调用者传进来的 —— 多卡时必须用当前上下文那张卡）
    int curDev = 0;
    if (cudaGetDevice(&curDev) != cudaSuccess) { (void)cudaGetLastError(); curDev = 0; }
    int multiProcessorCount = 0;
    if (cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, curDev)
        != cudaSuccess) {
        (void)cudaGetLastError();
        multiProcessorCount = 0;
    }
    if (multiProcessorCount <= 0) {            // 查询失败：保持编译期常量，不做派生
        g_k4MinBlocks   = (FUSED_K4_MIN_BLOCKS == SLIME_K4_MIN_BLOCKS_AUTO) ? 408 : FUSED_K4_MIN_BLOCKS;
        g_zSplitMinTile = Z_SPLIT_MIN_TILES;
        g_k4MinTiles    = FUSED_K_MIN_COLS / (WARPS_PER_BLOCK * WARP_SIZE * FUSED_K);
        return;
    }
    // 测试钩子：在没有目标卡的情况下验证"换卡后阈值会怎么变"（见 tools/portability_check.sh）。
    // 例：SLIME_FAKE_SM=128 SLIME_FAKE_BPSM=6 模拟 128 SM 的大卡 ⇒ 阈值变 1536 / 12800 / 128。
    int fakeBpsm = 0;
    if (const char* fs = getenv("SLIME_FAKE_SM")) {
        const long v = strtol(fs, nullptr, 10);
        if (v > 0 && v < 100000) multiProcessorCount = (int)v;
        if (const char* fb = getenv("SLIME_FAKE_BPSM")) {
            const long b = strtol(fb, nullptr, 10);
            if (b > 0 && b < 1000) fakeBpsm = (int)b;
        }
        // 假卡的 shared 画像：交叉点只由 shared 决定，不注入就验证不到跨卡行为。
        if (const char* fm = getenv("SLIME_FAKE_SHARED_PER_SM")) {
            const long m = strtol(fm, nullptr, 10);
            if (m > 0) g_fakeSharedPerSm = (int)m;
        }
        if (const char* fp = getenv("SLIME_FAKE_SHARED_PER_BLOCK")) {
            const long p = strtol(fp, nullptr, 10);
            if (p > 0) g_fakeSharedPerBlock = (int)p;
        }
    }
    int blocksPerSm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, scanRectFusedMerge4Kernel<FUSED_K>, WARPS_PER_BLOCK * WARP_SIZE, 0)
        != cudaSuccess) {
        (void)cudaGetLastError();              // 清掉可能的 sticky error
        blocksPerSm = 0;
    }
    if (fakeBpsm > 0) blocksPerSm = fakeBpsm;  // 测试钩子优先
    else if (blocksPerSm <= 0) blocksPerSm = 6; // 回退：256 线程 + 40 寄存器的保守值
    const int64_t slots = (int64_t)multiProcessorCount * blocksPerSm;
    if (slots <= 0) return;
    g_smUsed = multiProcessorCount;   // 仅供打印（可能来自测试钩子）
    g_bpsmUsed = blocksPerSm;
    // 三项都**按标定基准等比例缩放**（slots == CALIB_SLOTS 时逐位不变），
    // 且显式 -D 覆盖优先（覆盖值用于强制某条路径的测试，不能被派生逻辑改掉）。
    if (FUSED_K4_MIN_BLOCKS == SLIME_K4_MIN_BLOCKS_AUTO)
        g_k4MinBlocks = (int)(2 * slots);
    else
        g_k4MinBlocks = FUSED_K4_MIN_BLOCKS;          // 显式覆盖：原样使用
    if (Z_SPLIT_MIN_TILES == 2048)                    // 2048 = 编译期默认值（未被 -D 覆盖）
        g_zSplitMinTile = (int)(10 * slots);          // 本机 10*204 = 2040 -> 与 2048 同档
    else
        g_zSplitMinTile = Z_SPLIT_MIN_TILES;          // 显式覆盖（如 zsub1 的 1e8）
    g_k4MinTiles = (int)((int64_t)FUSED_K_MIN_COLS * slots / CALIB_SLOTS);
    // 注：10*204 = 2040 与原常量 2048 相差 0.4%，属同一波形档（实测 483~486 ms 无差异）；
    //     若要严格保持 2048，可把基准写成 2048/10 —— 但 204 是真实槽数，更诚实。
    if (g_k4MinBlocks < 1) g_k4MinBlocks = 1;
    if (g_zSplitMinTile < 1) g_zSplitMinTile = 1;
    if (g_k4MinTiles < 1) g_k4MinTiles = 1;
    g_cntRingMaxSizeZ = computeCntRingCrossover(g_fakeSharedPerSm, g_fakeSharedPerBlock);
}


/*
 * 合并滑动窗口输出。
 *
 * 前缀和是「整行连续」的 uint8_t 序列 ps[row][c] = (Σ_{c'<=c} v_c') mod 256，回绕
 * 多少次都无所谓：窗口计数 = (uint8_t)(ps[row][right] - ps[row][left])，模 256 的
 * 差值就是精确值（每行跨度 <= sizeX < 256）。所以这里既不需要判断窗口落在哪个段，
 * 也不需要拼接两半，就是两次字节读取 + 一次减法。
 *
 * 唯一的边界是 globalJ == 0 的那一列没有左端点（约定 ps[row][-1] = 0），用 lmask
 * 把左操作数按 0 处理，这个掩码在进入行循环前算好，循环内没有分支。
 */

__global__ void wideWindowScoreKernel(
    const uint8_t* __restrict__ d_row_ps,
    int rowPsWidth, int blockHeight, int sizeX, int sizeZ,
    int threshold, int startX, int baseZ,
    int batchValidStartZ, int batchValidEndZ,
    int validRangeX,
    int colOffset,  // 新增：本批次在全局X列中的起始列偏移
    HitResult* d_results, unsigned long long* d_pos, unsigned long long cap) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= validRangeX) return;

    // 整行连续前缀下，窗口计数就是两端之差；只有最左列没有左端点
    int globalJ = colOffset + j;
    int x = startX + j;  // startX 已是本批次起始区块坐标，加局部 j 即全局区块坐标

    const uint8_t* __restrict__ pR = d_row_ps + globalJ + sizeX - 1;  // ps[row][right]
    const uint8_t* __restrict__ pL = d_row_ps + (globalJ > 0 ? globalJ - 1 : 0);
    const int lmask = (globalJ > 0) ? 0xFF : 0x00;  // globalJ == 0 时约定 ps[row][-1] == 0

    int sum = 0, winVals[MAX_SIZE_Z], ringIdx = 0;
    bool stop = false;
    for (int row = 0; row < sizeZ; ++row) {
        int val = (uint8_t)(pR[0] - (pL[0] & lmask));
        sum += val;
        winVals[row] = val;
        pR += rowPsWidth;
        pL += rowPsWidth;
    }

    for (int i = 0; i <= blockHeight - sizeZ; ++i) {
        int z = baseZ + i;
        if (z >= batchValidStartZ && z <= batchValidEndZ && sum >= threshold) {
#ifdef SLIME_SCAN_ONLY
            // 只保留命中的数据依赖（防止整段被优化掉），不做任何登记：
            // 无 atomicAdd、无 d_results 的 12 字节写入、无 D2H 拷贝、无 CSV 行
            (void)x;
            ((int*)d_results)[threadIdx.x] = 1;
#else
            const unsigned long long pos = atomicAdd(d_pos, 1ULL);
            if (pos < cap) {  // 同一道 clamp：缓冲区满时丢弃而不是越界写
                d_results[pos].x = x * 16;
                d_results[pos].z = z * 16;
                d_results[pos].count = sum;
            } else {
                stop = true;  // 缓冲区满：本线程停手，主机丢弃本批并缩小重试
            }
#endif
        }
        if (stop || i == blockHeight - sizeZ) break;
        sum -= winVals[ringIdx];
        int val = (uint8_t)(pR[0] - (pL[0] & lmask));
        sum += val;
        winVals[ringIdx] = val;
        if (++ringIdx >= sizeZ) ringIdx = 0;  // 取代 % sizeZ
        pR += rowPsWidth;
        pL += rowPsWidth;
    }
}

/*
 * 主机端结果流水：生产者（GPU worker）→ 排序线程池 → k 路归并输出。
 *
 * 内存：启动时**一次 malloc** 申请尽可能大的一块（可用物理内存的 50%，上限 8GB，下限
 * 4 个块；拿不到就减半重试），之后所有槽块都从这块里切（bump），消费者还回来的块进空闲
 * 表复用；到全部结束前不 free、也不重新 malloc。
 *
 *   --sort=off：消费者是唯一的写盘线程，边取块边格式化写 CSV，块用完即还（内存里只流转
 *               几个块）。
 *   --sort=on ：消费者是 N 个排序线程。每个线程把一整块**就地** std::sort 成 (x,z) 有序段
 *               后保留下来 —— 所以排序完全发生在「GPU 还在算下一批」的时候，段与段彼此
 *               独立，多线程线性加速。内存不够时（大块切完且无空闲块）进入 spill 模式：
 *               排序线程把该段落盘成 <out>.runN 并还块，流水继续。收尾时对「内存段 +
 *               落盘段」做一遍 k 路归并（k 很小，线性选最小即可）直接写 CSV。
 *
 * 生产者只在「手上没有半满块」时才等空闲块，不会死锁：它一等的空档，排序线程必然还能继续
 * 消化 ready 队列并（spill 模式下）还块。
 */
static char* fmtUInt(char* p, unsigned v) {
    char t[12];
    int n = 0;
    do {
        t[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    while (n) *p++ = t[--n];
    return p;
}
static char* fmtInt(char* p, int v) {
    if (v < 0) {
        *p++ = '-';
        return fmtUInt(p, (unsigned)(-(long long)v));
    }
    return fmtUInt(p, (unsigned)v);
}

// 往 stderr 写一整行，并与进度条互斥（先把进度行抹掉再写）。定义在下面：
// 这里只声明，让 GPUWorker 不必看到 ProgressTicker 的完整定义。
static void progressEmitLine(const char* s);

class ResultPool {
   public:
    ResultPool(size_t blockSlots, int minBlocks, const char* outPath)
        : C(blockSlots), minBlocks(minBlocks), runPrefix(outPath) {
        const size_t blockBytes = C * sizeof(HitResult);
        const size_t floorBytes = (size_t)minBlocks * blockBytes;
        size_t want = budgetBytes();
        if (want < floorBytes) want = floorBytes;
        while (!base) {  // 只在大块申请失败时降级；成功的那次不会被 free
            base = (char*)malloc(want);
            if (base) {
                bytes = want;
                break;
            }
            if (want <= floorBytes) break;
            want = std::max(floorBytes, want / 2);
        }
        if (!base) {
            fprintf(stderr, "OOM: cannot allocate result pool\n");
            exit(EXIT_FAILURE);
        }
        // SLIME_SORT_BLOCKS 是测试旋钮：限制可切出的块数，强制走 spill/落盘再归并那条路
        if (const char* e = getenv("SLIME_SORT_BLOCKS")) {
            const long long v = atoll(e);
            if (v > 0) maxBlocks = (size_t)v;
        }
    }
    ~ResultPool() { free(base); }
    void setProducers(int n) { producers = n; }
    void startSorters(int n) {
        for (int i = 0; i < n; ++i) sorters.emplace_back([this] { sorterLoop(); });
    }
    void joinSorters() {
        for (auto& t : sorters) t.join();
    }
    uint64_t totalRows() const { return pushed; }  // 生产者累计搬进池的行数
    bool mergeRunsToCsv(const char* outPath) {
        struct Src {
            const HitResult* p = nullptr;
            size_t n = 0, i = 0;
            FILE* f = nullptr;
            std::vector<HitResult> buf;
        };
        const size_t FILE_CHUNK = 1 << 16;
        std::vector<Src> s(runs.size());
        for (size_t i = 0; i < runs.size(); ++i) {
            if (runs[i].f) {
                s[i].f = runs[i].f;
                s[i].buf.resize(FILE_CHUNK);
                s[i].n = fread(s[i].buf.data(), sizeof(HitResult), FILE_CHUNK, s[i].f);
                s[i].p = s[i].buf.data();
            } else {
                s[i].p = runs[i].p;
                s[i].n = runs[i].n;
            }
        }
        FILE* fout = fopen(outPath, "wb");  // 二进制：Windows 上不做 \n -> \r\n 翻译，跨平台输出逐字节一致
        if (!fout) {
            fprintf(stderr, "Cannot open output file: %s\n", outPath);
            return false;
        }
        fprintf(fout, "x,z,slime_count\n");
        std::vector<char> ob(1 << 20);
        size_t obUsed = 0;
        auto flushOb = [&] {
            if (obUsed) {
                fwrite(ob.data(), 1, obUsed, fout);
                obUsed = 0;
            }
        };
        while (true) {
            size_t best = SIZE_MAX;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i].i >= s[i].n) {
                    if (s[i].f && s[i].n == FILE_CHUNK) {  // 上一次读满了，可能还有下一批
                        s[i].n = fread(s[i].buf.data(), sizeof(HitResult), FILE_CHUNK, s[i].f);
                        s[i].i = 0;
                    }
                    if (s[i].i >= s[i].n) continue;  // 该段读完
                }
                if (best == SIZE_MAX || less(s[i].p[s[i].i], s[best].p[s[best].i])) best = i;
            }
            if (best == SIZE_MAX) break;
            const HitResult& h = s[best].p[s[best].i];
            if (obUsed + 64 > ob.size()) flushOb();  // 一行最多 ~40 字节
            char* q = ob.data() + obUsed;
            q = fmtInt(q, h.x);
            *q++ = ',';
            q = fmtInt(q, h.z);
            *q++ = ',';
            q = fmtUInt(q, (unsigned)h.count);
            *q++ = '\n';
            obUsed = (size_t)(q - ob.data());
            ++s[best].i;
        }
        flushOb();
        fclose(fout);
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i].f) {
                fclose(s[i].f);
                // 按「文件名下标」删，不能用向量位置 i：前面可能夹着内存段，
                // 位置 != runPath 的后缀，会把不存在的 .runK 删掉而漏掉真正的落盘段。
                remove(runPath(runs[i].fileIdx).c_str());
            }
        }
        return true;
    }
    // 生产者：把设备上的 n 条结果搬进池（必要时跨块）
    void pushFromDevice(int& cur, const HitResult* d_src, int64_t n) {
        while (n > 0) {
            std::unique_lock<std::mutex> lk(mu);
            if (cur < 0) cur = carveOrWait(lk);  // 可能等空闲块
            const size_t take = std::min((size_t)n, C - blocks[cur].used);
            HitResult* dst = blocks[cur].p + blocks[cur].used;
            lk.unlock();  // 拷贝期间不占锁，消费者可以继续吐别的块
            CUDA_CHECK(cudaMemcpy(dst, d_src, take * sizeof(HitResult), cudaMemcpyDeviceToHost));
            lk.lock();
            blocks[cur].used += take;
            pushed += take;
            d_src += take;
            n -= (int64_t)take;
            if (blocks[cur].used == C) {
                ready.push_back(cur);
                cur = -1;
                cvReady.notify_one();
            }
        }
    }
    void flush(int& cur) {  // 生产者收尾：半满块也交给消费者
        std::lock_guard<std::mutex> lk(mu);
        if (cur >= 0) {
            ready.push_back(cur);
            cur = -1;
            cvReady.notify_one();
        }
    }
    void producerDone() {
        std::lock_guard<std::mutex> lk(mu);
        if (++doneProducers == producers) cvReady.notify_all();
    }
    // 消费者：取一个可消费的块；false = 生产者全部结束且已排空
    bool take(int& idx) {
        std::unique_lock<std::mutex> lk(mu);
        cvReady.wait(lk, [&] { return !ready.empty() || doneProducers == producers; });
        if (ready.empty()) return false;
        idx = ready.front();
        ready.pop_front();
        return true;
    }
    size_t rows(int idx) const { return blocks[idx].used; }
    const HitResult* data(int idx) const { return blocks[idx].p; }
    void release(int idx) {
        std::lock_guard<std::mutex> lk(mu);
        blocks[idx].used = 0;
        freeIdx.push_back(idx);
        cvFree.notify_one();
    }

   private:
    struct Block {
        HitResult* p = nullptr;
        size_t used = 0;
    };
    struct Run {
        const HitResult* p = nullptr;
        size_t n = 0;
        FILE* f = nullptr;
        size_t fileIdx = SIZE_MAX;  // 落盘段的文件名下标；内存段是 SIZE_MAX
    };
    static bool less(const HitResult& a, const HitResult& b) {
        if (a.x != b.x) return a.x < b.x;
        return a.z < b.z;
    }
    static size_t hostAvailBytes() {
#ifdef _WIN32
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        return GlobalMemoryStatusEx(&ms) ? (size_t)ms.ullAvailPhys : ((size_t)2 << 30);
#else
        const long pages = sysconf(_SC_AVPHYS_PAGES), psize = sysconf(_SC_PAGESIZE);
        return (pages > 0 && psize > 0) ? (size_t)pages * (size_t)psize : ((size_t)2 << 30);
#endif
    }
    static size_t budgetBytes() {
        const size_t avail = hostAvailBytes();
        size_t want = avail / 2;               // 可用物理内存的 50%
        if (want > ((size_t)8 << 30)) want = (size_t)8 << 30;
        return want;
    }
    std::string runPath(size_t i) const {
        return runPrefix + ".run" + std::to_string(i);
    }
    // 申请一个块（需要持有 mu）：优先空闲表，其次从大块 bump 切，都没有就等排序线程还块
    int carveOrWait(std::unique_lock<std::mutex>& lk) {
        for (;;) {
            if (!freeIdx.empty()) {
                const int i = freeIdx.back();
                freeIdx.pop_back();
                blocks[i].used = 0;
                return i;
            }
            if (blocks.size() < maxBlocks && bump + C * sizeof(HitResult) <= bytes) {
                blocks.push_back({(HitResult*)(base + bump), 0});
                bump += C * sizeof(HitResult);
                return (int)blocks.size() - 1;
            }
            spillMode = true;  // 大块切完了：让排序线程开始落盘还块
            cvFree.wait(lk);
        }
    }
    void sorterLoop() {
        int idx;
        while (take(idx)) {
            HitResult* p = blocks[idx].p;
            const size_t n = blocks[idx].used;
            std::sort(p, p + n, less);  // 独占该块，不需要持锁
            std::unique_lock<std::mutex> lk(mu);
            if (spillMode) {  // 内存吃紧：落盘成有序段，把块还给生产者
                const size_t runIdx = runCount++;
                lk.unlock();
                FILE* f = fopen(runPath(runIdx).c_str(), "wb+");
                const bool ok = f && fwrite(p, sizeof(HitResult), n, f) == n;
                if (ok) rewind(f);
                lk.lock();
                if (!ok) {
                    {
                        char line[512];
                        snprintf(line, sizeof(line), "Cannot write sort run %s (disk full?)\n",
                                 runPath(runIdx).c_str());
                        progressEmitLine(line);
                    }
                    exit(EXIT_FAILURE);
                }
                runs.push_back(Run{nullptr, n, f, runIdx});
                blocks[idx].used = 0;
                freeIdx.push_back(idx);
                cvFree.notify_one();
            } else {
                runs.push_back(Run{p, n, nullptr, SIZE_MAX});
            }
        }
    }

    size_t C, bytes = 0, bump = 0, maxBlocks = SIZE_MAX;
    uint64_t pushed = 0;
    int minBlocks;
    std::string runPrefix;
    size_t runCount = 0;
    bool spillMode = false;
    char* base = nullptr;
    std::vector<Block> blocks;
    std::vector<Run> runs;
    std::deque<int> ready;
    std::vector<int> freeIdx;
    std::vector<std::thread> sorters;
    std::mutex mu;
    std::condition_variable cvReady, cvFree;
    int producers = 1, doneProducers = 0;
};

void computeBases(int32_t xStart, int32_t width, int32_t zStart, int32_t height,
                  int64_t seed, int64_t* h_baseX, int64_t* h_baseZ) {
    for (int i = 0; i < width; ++i) h_baseX[i] = slimeQuadX(xStart + i) + seed;
    for (int i = 0; i < height; ++i) h_baseZ[i] = slimeQuadZ(zStart + i);
}

// 只在Z轴上切分，所以StartX, StartZ等均为共用的，使用全局变量管理
// 每个需要维护的只有自己的输出缓冲区，使用vector, 自己的H_max与偏移，把全图切成多块
class GPUWorker {
   public:
    GPUWorker() : device_id(0), pool(nullptr), H_max(0), offset(0) {}
    GPUWorker(int device_id, ResultPool* pool, int64_t H_max, int64_t offset) : device_id(device_id), pool(pool), H_max(H_max), offset(offset) {}
    void run() {
        cudaSetDevice(device_id);
        // 并行度阈值必须在这里定：此刻上下文是**本 worker 自己那张卡**。
        // （早期版本在 main 的设备枚举循环里调用，globals 会被最后一个设备覆盖。）
        tuneParallelismThresholds();
        if (getenv("SLIME_PRINT_TUNING")) {
            fprintf(stderr,
                    "[worker dev=%d] %d SM x %d blocks = %d slots -> K4>=%d blocks, Zsplit>=%d tiles, K4>=%d tiles, cntRing sizeZ<=%d\n",
                    device_id, g_smUsed, g_bpsmUsed, g_smUsed * g_bpsmUsed,
                    g_k4MinBlocks, g_zSplitMinTile, g_k4MinTiles, g_cntRingMaxSizeZ);
        }

        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));  // 用于计时的两个事件
        CUDA_CHECK(cudaEventCreate(&ev_stop));
#ifdef SLIME_PROFILE
        cudaEvent_t ev_p0, ev_p1;  // 仅用于把行前缀核与滑窗核的耗时分开
        CUDA_CHECK(cudaEventCreate(&ev_p0));
        CUDA_CHECK(cudaEventCreate(&ev_p1));
#endif

        const bool useFused = (sizeX <= MAX_FUSED_SIZE_X);
        const int rangeX = width - sizeX + 1;

        // ---- 结果缓冲区：按空闲显存放大（GPU 分配贵，四个缓冲仍只分配一次） ----
        // 不变式 cap >= rangeX：rangeX <= 2*MAX_COORD+2（硬界），所以最多 48MB；
        // 于是 safeRows = cap/rangeX >= 1 恒成立，而「命中数 <= 候选数」，按 safeRows
        // 切块永不溢出。更宽的块靠实测密度放大，真溢出了也有内核 clamp 兜底 + 本轮重试。
        size_t freeNow = 0, totalNow = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeNow, &totalNow));
        int64_t cap = (int64_t)((double)freeNow * 0.25 / (double)sizeof(HitResult));
        if (cap > MAX_BATCH_OUTPUT_CAP) cap = MAX_BATCH_OUTPUT_CAP;
        // SLIME_CAP_SLOTS 只是测试旋钮：用来把 cap 压小以强制走「缓冲区满 -> 重试/多轮」
        // 路径；它同样不能低于 rangeX，否则 safeRows 的论证就不成立了。
        if (const char* capEnv = getenv("SLIME_CAP_SLOTS")) {
            int64_t forced = atoll(capEnv);
            if (forced > 0) cap = forced;
        }
        if (cap < rangeX) cap = rangeX;

        // 一次 cudaMalloc 备齐 baseX/baseZ/results/pos 四个缓冲，
        // 每个子缓冲 256B 对齐（baseX 在偏移 0，天然对齐）。
        int64_t *d_baseX = nullptr, *d_baseZ = nullptr;
        HitResult* d_results = nullptr;
        const size_t offBaseZ = (((size_t)width + BASE_X_PAD) * sizeof(int64_t) + 255) & ~(size_t)255;
        const size_t offResults = (offBaseZ + (size_t)height * sizeof(int64_t) + 255) & ~(size_t)255;
        const size_t offPos = (offResults + (size_t)cap * sizeof(HitResult) + 255) & ~(size_t)255;
        void* d_pool = nullptr;
        CUDA_CHECK(cudaMalloc(&d_pool, offPos + 256));
        d_baseX = (int64_t*)((char*)d_pool);
        d_baseZ = (int64_t*)((char*)d_pool + offBaseZ);
        d_results = (HitResult*)((char*)d_pool + offResults);
        // 64 位计数器：threshold 0 这种「每个窗口都命中」的负载会超过 2^32，用 32 位会让
        // 主机把「回绕后的 0」误判成完成，从而静默丢结果。
        unsigned long long* d_pos = (unsigned long long*)((char*)d_pool + offPos);
        CUDA_CHECK(cudaMemcpy(d_baseX, h_baseX, width * sizeof(int64_t), cudaMemcpyHostToDevice));
        // 尾部填充清零：内核会无条件读这一小段（越界列的值必被窗口掩码丢掉）
        CUDA_CHECK(cudaMemset((char*)d_baseX + (size_t)width * sizeof(int64_t), 0,
                              (size_t)BASE_X_PAD * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(d_baseZ, h_baseZ, height * sizeof(int64_t), cudaMemcpyHostToDevice));

        // 只有回退路径（sizeX > MAX_FUSED_SIZE_X）需要行前缀数组
        uint8_t* d_row_ps = nullptr;
        if (!useFused) CUDA_CHECK(cudaMalloc(&d_row_ps, width * H_max * sizeof(uint8_t)));

        // 直接拷进结果池的当前块（FIFO 入队，写盘线程按同一顺序消费）
        auto drain = [&](int64_t n) {
            if (n > 0) pool->pushFromDevice(curBlock, d_results, n);
        };

        for (int64_t baseZ = startZ; baseZ <= valid_endZ; baseZ += step) {  // 这一段对于所有卡都是一样的，也就是说明这是对于全局而言的大块概念
            progRecord(0);  // 每个 Z 大块重新开始计数（否则第一块跑完就显示 100%）
            int64_t blockStartZ = baseZ + offset;                           // 定位到在这个大块中本卡的任务起始点
            int64_t blockEndZ = blockStartZ + H_max - 1;                    // 根据高度为H_max算出本次任务的终点
            int64_t blockHeight = H_max;                                    // 本次任务的高度
            // 有效输出窗口起始点范围 [blockStartZ, blockStartZ + H_max - sizeZ]（闭区间）
            int64_t validStartZ = blockStartZ;                // 起始与本次任务的起始保持一致
            int64_t validEndZ = validStartZ + H_max - sizeZ;  // 但是结尾要短一点，短sizeZ-1

            CUDA_CHECK(cudaEventRecord(ev_start));
#ifdef SLIME_PROFILE
            CUDA_CHECK(cudaEventRecord(ev_p0));
#endif

            // 回退路径才需要先算整行连续前缀；融合路径直接读 d_baseX
            if (!useFused) {
                dim3 grid((unsigned)((blockHeight + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
                dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
                wideRowPrefixSumKernel<<<grid, block>>>(
                    d_baseX, d_baseZ, width, (int)blockHeight,
                    (int)(blockStartZ - startZ), d_row_ps, width);
            }
#ifdef SLIME_PROFILE
            CUDA_CHECK(cudaEventRecord(ev_p1));
#endif

            int64_t totalHits = 0, totalChunks = 0, totalRedo = 0;

            // 计数环变体的选路守卫（两个条件都与 chunk 无关，故在行循环外算一次）：
            //   * K == 1：计数环在 K=1 上实测更慢（ALU/候选 13.75 -> 18.00，ncu cycles
            //     +38%）⇒ 编译期直接回退ballot 环核；
            //   * sizeZ > COUNT_RING_MAX_SIZE_Z(20)：动态 shared ∝ sizeZ，占用率塌陷
            //     （实测 sizeZ=24 +3.9%、32 +4.3%）⇒ 运行期回退ballot 环核。
            constexpr bool COUNT_RING_OK = (FUSED_K != 1);
            const bool cntRing = COUNT_RING_OK && (sizeZ <= g_cntRingMaxSizeZ);
            // T1 v2 的 merge4 核（4 邻列共享松上界 + 过门精确重算）：并集位段宽 sizeX+3
            // 要一次 32 位 funnel 抽出 ⇒ sizeX <= 29。它的 shared 全静态 9216 B 且与 sizeZ
            // 无关，所以适用域比计数环核宽（sizeZ 不受 20 限制）；sizeX ∈ [30,32] 仍走旧核。
            const bool merge4 = COUNT_RING_OK && (sizeX <= MAX_MERGE4_SIZE_X);
            // 计数环 路径的核用动态 shared（= sizeZ × TILE_W 字节，读旧写新同槽，环深 = sizeZ）。
            // 守卫已把 sizeZ 限在 <= 20，K=4 时最大 20 × 1024 = 20 KB < 48 KB 默认上限，
            // 所以下面这条按需 opt-in 现在不会触发；保留它是为了将来放宽守卫时不会因为
            // 超过默认上限而 launch 失败。ballot 环核用静态 shared、不传动态字节数，因此守卫
            // 分支（base）下不存在「用到未设置的属性」的情况。
            if constexpr (COUNT_RING_OK) {
                const size_t dynBytes = (size_t)sizeZ * (WARPS_PER_BLOCK * WARP_SIZE * FUSED_K);
                if (dynBytes > 48 * 1024)
                    CUDA_CHECK(cudaFuncSetAttribute(scanRectFusedKernelCount<FUSED_K>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dynBytes));
            }
            // 一次 launch 覆盖整个 X 范围，不再按 X 分批：K 只按 X 宽度选一次
            auto launchFusedChunk = [&](int64_t candZ, int64_t rows) {
                const int launchRows = (int)(rows + sizeZ - 1);  // 输入行数含 Z halo
                const int zSubMax = (int)((rows + Z_SUB_ROWS - 1) / Z_SUB_ROWS);  // 抗波形量化
                // K=4 每个 block 覆盖 4 倍的列，并行度只有 K=1 的 1/4；只有 block 数
                // 够填满 GPU 时才值得用它换指令数。
                // 判据用运行期阈值（见 tuneParallelismThresholds）：本机 408 / 204 与原值相同，
                // 大卡上随常驻槽数放大。第一支 `tiles4probe >= g_k4MinTiles` 等价于原来的
                // `rangeX >= FUSED_K_MIN_COLS`（200000 / 1008 ≈ 198 个 tile），但用真实槽数表达。
                const int outW4probe = WARPS_PER_BLOCK * WARP_SIZE * FUSED_K - sizeX + 1;
                const int tiles4probe = (rangeX + outW4probe - 1) / outW4probe;
                if (tiles4probe >= g_k4MinTiles ||
                    (int64_t)tiles4probe * zSubMax >= g_k4MinBlocks) {
                    const int outW = WARPS_PER_BLOCK * WARP_SIZE * FUSED_K - sizeX + 1;
                    const int numTiles = (rangeX + outW - 1) / outW;
                    const int zSub = (numTiles < g_zSplitMinTile) ? zSubMax : 1;
                    const int perSub = (int)((rows + zSub - 1) / zSub);  // 每个 Z 子块的行数
                    const dim3 grid(numTiles, zSub);
                    if (merge4) {  // T1 v2 默认路径：静态 shared 9216 B，无动态 shared
                        scanRectFusedMerge4Kernel<FUSED_K><<<grid, WARPS_PER_BLOCK * WARP_SIZE>>>(
                            d_baseX, d_baseZ, width, launchRows, (int)(candZ - startZ),
                            sizeX, sizeZ, threshold, startX, (int)candZ,
                            0, rangeX, d_results, d_pos, (unsigned long long)cap, perSub);
                    } else if (cntRing) {  // sizeX ∈ [30,32]：动态 shared = sizeZ × TILE_W
                        scanRectFusedKernelCount<FUSED_K><<<grid, WARPS_PER_BLOCK * WARP_SIZE,
                            (size_t)sizeZ * (WARPS_PER_BLOCK * WARP_SIZE * FUSED_K)>>>(
                            d_baseX, d_baseZ, width, launchRows, (int)(candZ - startZ),
                            sizeX, sizeZ, threshold, startX, (int)candZ,
                            0, rangeX, d_results, d_pos, (unsigned long long)cap, perSub);
                    } else {  // sizeZ > 20（占用率塌陷）或 FUSED_K == 1（实测更慢）：ballot 环核
                        scanRectFusedKernelBallot<FUSED_K><<<grid, WARPS_PER_BLOCK * WARP_SIZE>>>(
                            d_baseX, d_baseZ, width, launchRows, (int)(candZ - startZ),
                            sizeX, sizeZ, threshold, startX, (int)candZ,
                            0, rangeX, d_results, d_pos, (unsigned long long)cap, perSub);
                    }
                } else {
                    const int outW = WARPS_PER_BLOCK * WARP_SIZE - sizeX + 1;
                    const int numTiles = (rangeX + outW - 1) / outW;
                    const int zSub = (numTiles < g_zSplitMinTile) ? zSubMax : 1;
                    const int perSub = (int)((rows + zSub - 1) / zSub);  // 每个 Z 子块的行数
                    const dim3 grid(numTiles, zSub);
                    // K=1：计数环在 K=1 上实测更慢（ALU/候选 13.75 -> 18.00），恒走ballot 环核
                    scanRectFusedKernelBallot<1><<<grid, WARPS_PER_BLOCK * WARP_SIZE>>>(
                        d_baseX, d_baseZ, width, launchRows, (int)(candZ - startZ),
                        sizeX, sizeZ, threshold, startX, (int)candZ,
                        0, rangeX, d_results, d_pos, (unsigned long long)cap, perSub);
                }
            };
            auto launchFallbackBatch = [&](int colOffset, int batchRangeX) {
                const int gridSw = (batchRangeX + 255) / 256;
                wideWindowScoreKernel<<<gridSw, 256>>>(
                    d_row_ps, width, (int)blockHeight, sizeX, sizeZ,
                    threshold, startX + colOffset, (int)blockStartZ,
                    (int)validStartZ, (int)validEndZ, batchRangeX,
                    colOffset, d_results, d_pos, (unsigned long long)cap);
            };
            auto redoTooMany = [&](int64_t at) {
                {
                    char line[160];
                    snprintf(line, sizeof(line), "GPU:%d internal: chunk retry limit (%d) at %" PRId64 "\n",
                             device_id, MAX_RETRIES, at);
                    progressEmitLine(line);
                }
                exit(EXIT_FAILURE);
            };

            if (useFused) {
                // Z 方向自适应 chunk：先按跨度上限起步（失败也便宜，见 MAX_CHUNK_ROWS），
                // 之后按上一块的实测密度放大，把命中数维持在缓冲区的一半左右。
                const int64_t candZ0 = validStartZ, candZ1 = validEndZ;
                const int64_t safeRows = std::max<int64_t>(1, cap / rangeX);
                int64_t rows = std::min<int64_t>(candZ1 - candZ0 + 1, MAX_CHUNK_ROWS);
                int retries = 0;
                progInit(candZ1 - candZ0 + 1);
                for (int64_t z = candZ0; z <= candZ1;) {
                    const int64_t remain = candZ1 - z + 1;
                    if (rows > remain) rows = remain;
                    CUDA_CHECK(cudaMemset(d_pos, 0, sizeof(unsigned long long)));
                    launchFusedChunk(z, rows);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    unsigned long long n = 0;
                    CUDA_CHECK(cudaMemcpy(&n, d_pos, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                    if (n > (unsigned long long)cap) {
                        // 缓冲区满：内核已停手，本轮结果作废 -> 缩小 chunk 重试同一段 Z
                        if (++retries > MAX_RETRIES) redoTooMany(z);
                        ++totalRedo;
                        rows = std::max<int64_t>(safeRows, rows / 2);
                        continue;
                    }
                    retries = 0;
                    drain((int64_t)n);
                    totalHits += (int64_t)n;
                    ++totalChunks;
                    const int64_t scanned = rows;
                    const double density = (double)n / (double)((int64_t)rangeX * scanned);
                    int64_t grow = (density > 0.0)
                                       ? (int64_t)(DENSITY_FILL * (double)cap / (density * (double)rangeX))
                                       : MAX_CHUNK_ROWS;
                    if (grow < scanned) grow = scanned;
                    if (grow > scanned * 2) grow = scanned * 2;  // 最多翻倍，避免震荡
                    rows = std::min<int64_t>(MAX_CHUNK_ROWS, grow);
                    z += scanned;
                    progRecord(z - candZ0);  // 已覆盖的候选 Z 行数（重试不推进，符合直觉）
                }
            } else {
                // 回退路径：前缀已算好，按 X 自适应分批滑窗（同样是「先保证不溢出的列数」）
                const int64_t rowsPerBatch = std::max<int64_t>(1, blockHeight - sizeZ + 1);
                const int64_t safeCols = std::max<int64_t>(1, cap / rowsPerBatch);
                int64_t cols = std::min<int64_t>((int64_t)rangeX, MAX_CHUNK_ROWS);
                int retries = 0;
                progInit(rangeX);
                for (int64_t c = 0; c < rangeX;) {
                    const int64_t remain = rangeX - c;
                    if (cols > remain) cols = remain;
                    CUDA_CHECK(cudaMemset(d_pos, 0, sizeof(unsigned long long)));
                    launchFallbackBatch((int)c, (int)cols);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    unsigned long long n = 0;
                    CUDA_CHECK(cudaMemcpy(&n, d_pos, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                    if (n > (unsigned long long)cap) {
                        if (++retries > MAX_RETRIES) redoTooMany(c);
                        ++totalRedo;
                        cols = std::max<int64_t>(safeCols, cols / 2);
                        continue;
                    }
                    retries = 0;
                    drain((int64_t)n);
                    totalHits += (int64_t)n;
                    ++totalChunks;
                    const int64_t scanned = cols;
                    const double density = (double)n / (double)(scanned * rowsPerBatch);
                    int64_t grow = (density > 0.0)
                                       ? (int64_t)(DENSITY_FILL * (double)cap / (density * (double)rowsPerBatch))
                                       : (int64_t)rangeX;
                    if (grow < scanned) grow = scanned;
                    if (grow > scanned * 2) grow = scanned * 2;
                    cols = std::min<int64_t>((int64_t)rangeX, grow);
                    c += scanned;
                    progRecord(c);
                }
            }

            progFinish();
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
#ifdef SLIME_PROFILE
            float ms_prefix = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms_prefix, ev_p0, ev_p1));
            prof_prefix_ms += ms_prefix;
            prof_window_ms += (double)ms - ms_prefix;  // 含分批的 D2H 拷贝与同步间隙
#endif
            // 必须走 stderr 并与进度条互斥：stdout 是块缓冲的，printf 的内容要等程序
            // 结束时才 flush，落到终端时正好压在进度行中间，把行首吃掉。
            {
                char line[256];
                snprintf(line, sizeof(line),
                         "GPU:%-2d Block Z[%" PRId64 ",%" PRId64 "] height %" PRId64 ", valid Z[%" PRId64 ",%" PRId64
                         "] -> %" PRId64 " valid, %" PRId64 " chunk(s), %" PRId64 " redo, %.2f ms\n",
                         device_id, blockStartZ, blockEndZ, blockHeight, validStartZ, validEndZ,
                         totalHits, totalChunks, totalRedo, ms);
                progressEmitLine(line);
            }
            gpu_time_ms += ms;
        }
        if (d_row_ps) CUDA_CHECK(cudaFree(d_row_ps));
        CUDA_CHECK(cudaFree(d_pool));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
#ifdef SLIME_PROFILE
        CUDA_CHECK(cudaEventDestroy(ev_p0));
        CUDA_CHECK(cudaEventDestroy(ev_p1));
#endif
        pool->flush(curBlock);   // 半满块也要交给写盘线程
        pool->producerDone();
    }
    double getGpuTime() const { return gpu_time_ms; }
    // ---- 进度（供 ProgressTicker 读）----
    // 粒度是 Z 行覆盖：每个 chunk launch 前写一次原子（全图规模下每卡几十次），
    // 不在 kernel 热路径上、不影响 CSV 内容与计时。
#ifdef SLIME_PROGRESS_CODE
    void progInit(int64_t zLen) { progZLen_ = zLen > 0 ? zLen : 1; }
    void progRecord(int64_t zNow) { progZNow.store(zNow, std::memory_order_relaxed); }
    void progFinish() { progZNow.store(progZLen_, std::memory_order_relaxed); }
    int64_t progTotal() const { return progZLen_; }
    int64_t progDone() const { return progZNow.load(std::memory_order_relaxed); }
#else
    void progInit(int64_t) {}
    void progRecord(int64_t) {}
    void progFinish() {}
    int64_t progTotal() const { return 1; }
    int64_t progDone() const { return 1; }
#endif
#ifdef SLIME_PROFILE
    double getProfPrefixMs() const { return prof_prefix_ms; }
    double getProfWindowMs() const { return prof_window_ms; }
#endif

   private:
    int device_id;
    ResultPool* pool;
    int curBlock = -1;  // 本生产者正在填的池块（-1 = 手上没有半满块）
    int64_t H_max;
    int64_t offset;
    double gpu_time_ms = 0.0;
#ifdef SLIME_PROGRESS_CODE
    int64_t progZLen_ = 1;                  // 本 worker 负责的有效候选 Z 行数（启动后只读）
    std::atomic<int64_t> progZNow{0};       // 已扫过的候选 Z 行数
#endif
#ifdef SLIME_PROFILE
    double prof_prefix_ms = 0.0;
    double prof_window_ms = 0.0;
#endif
};

/*
 * 进度条：一个独立的展示线程，按固定间隔读各 worker 的 Z 大块计数。
 *
 * 开销：被计数的对象是**每卡每 Z 大块一次**的原子加（全图规模下几十次），
 * 不在 kernel 热路径上、不影响 CSV 内容；展示侧每 RESCAN_MS 最多一次 write(2)。
 * 该量级低于计时噪声，所以对 `inst/候选` 与 B/A 比值都不构成影响。
 *
 * 输出走 stderr（stdout 留给 CSV 与汇总行），且只在 stderr 是终端时默认开启：
 * 重定向到文件或管道时自动关闭，日志与 benchmark 输出保持纯净。
 * SLIME_NO_PROGRESS=1 强制关闭，SLIME_PROGRESS=1 强制开启（也不要求是终端）。
 */
// stderr 是不是终端：POSIX 用 isatty(fileno(f))，MSVC 里这两个叫 _isatty/_fileno
// 且声明在 <io.h>（unistd.h 在 Windows 上不存在）。
static bool streamIsTty(FILE* f) {
#ifdef _WIN32
    return _isatty(_fileno(f)) != 0;
#else
    return isatty(fileno(f)) != 0;
#endif
}

class ProgressTicker {
   public:
    ProgressTicker(GPUWorker* w, int count, bool enabled) : workers_(w), count_(count), on_(enabled) {
        // 每个 worker 贡献 1：全局完成度 = Σ(zNow / zLen)。单卡时就是它自己的 Z 行占比。
        total_ = count_;
    }
    void start() {
        if (!on_ || total_ <= 0) return;
        thr_ = std::thread([this] { loop(); });
    }
    void stop() {
        if (thr_.joinable()) thr_.join();
        // 进度条用完就抹掉：终端上只留报告。它本来就是个瞬时显示，收尾时留在屏幕上
        // 只会和后面的汇总行抢位置。
        {
            std::lock_guard<std::mutex> lk(mu_);
            clearLocked();
        }
    }

    // 运行期还有别的线程往 stderr 写字（每个 Z 大块的 "GPU:n Block ..." 汇总、出错信息）。
    // 它们必须先把进度行从终端上抹掉：否则后一次进度刷新从行首覆盖，会把那行开头吃掉
    // （实测 "GPU:0  Block Z[...]" 被压成 "sGPU:0 ..."）。用空格覆盖而不是只回行首 ——
    // 只回行首的话，上一次较长的那行会留下尾巴。
    void pause() {
        std::lock_guard<std::mutex> lk(mu_);
        clearLocked();
    }
    void resume() { std::lock_guard<std::mutex> lk(mu_); }
    void emit(const char* s) {  // 让出进度行，写一整行，再交回
        pause();
        fputs(s, stderr);
        fflush(stderr);
        resume();
    }

   private:
    static constexpr int RESCAN_MS = 500;  // 首次即时显示，之后每 500 ms 刷新一次

    void loop() {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        auto next = t0;
        while (true) {
            const auto now = clock::now();
            const double el = std::chrono::duration<double>(now - t0).count();
            draw(el);
            // 全部 worker 都覆盖满自己的 Z 区间即结束
            bool all = true;
            for (int i = 0; i < count_; ++i) {
                if (workers_[i].progDone() < workers_[i].progTotal()) { all = false; break; }
            }
            if (all) break;
            next += std::chrono::milliseconds(RESCAN_MS);
            if (now < next) std::this_thread::sleep_until(next);
            else next = now;
        }
    }

    void draw(double el) const {
        constexpr int W = 24;
        // 完成度以 Z 行占比计（每个 worker 一段），进度条因此是细粒度的：
        // 全图只切成一个 Z 大块时，也能按 chunk 平滑推进。
        double done = 0.0;
        for (int i = 0; i < count_; ++i) {
            const int64_t len = workers_[i].progTotal();
            if (len > 0) done += (double)workers_[i].progDone() / (double)len;
        }
        if (done > (double)total_) done = (double)total_;
        const int pct = (int)(done * 100.0 / (double)total_ + 0.5);
        int fill = (int)(done * W / (double)total_);
        if (fill > W) fill = W;
        if (fill < 0) fill = 0;

        char bar[W + 1];
        for (int i = 0; i < W; ++i) bar[i] = (i < fill) ? '#' : '-';
        bar[W] = '\0';

        // 吞吐用已完成的大块数估计；ETA 只在有非零进度且尚未完成时给
        char eta[24] = "";
        if (done > 0.0 && done < (double)total_ && el > 0.0) {
            const double left = el * ((double)total_ - done) / done;
            snprintf(eta, sizeof(eta), "  eta %5.0fs", left);
        }

        char line[192];
        const int n = snprintf(line, sizeof(line),
                               "\r[%s] %3d%%  elapsed %6.1fs%s",
                               bar, pct, el, eta);
        if (n > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            fwrite(line, 1, (size_t)n, stderr);
            line_.assign(line, (size_t)n);
            fflush(stderr);
        }
    }

    void clearLocked() {  // 调用方已持有 mu_：把当前进度行从终端上抹掉
        if (line_.empty()) return;
        std::string blank(line_.size(), ' ');
        fputc('\r', stderr);
        fwrite(blank.data(), 1, blank.size(), stderr);
        fputc('\r', stderr);
        fflush(stderr);
        line_.clear();
    }

    GPUWorker* workers_;
    int count_;
    bool on_;
    mutable std::mutex mu_;  // 保护 stderr 上的进度行与 line_
    mutable std::string line_;  // 当前画在终端上的那一行（不含前导 '\r'）
    int64_t total_ = 0;
    std::thread thr_;
};

// 进度条实例的全局指针。worker 线程先于 ticker 构造启动，所以读取时可能仍是 nullptr
// （那时也还没画过任何东西，直接输出即可）。
static ProgressTicker* g_ticker = nullptr;

static void progressEmitLine(const char* s) {
    if (g_ticker) g_ticker->emit(s);
    else { fputs(s, stderr); fflush(stderr); }
}

int main(int argc, char* argv[]) {
    // --sort=on|off 可在任意位置出现；默认 on（扫描阶段落二进制 spill，结束后一次分配
    // 大内存做排序/多路归并，产出 (x,z) 字典序的 CSV）
    bool sortOutput = true;
    std::vector<char*> pos;
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--sort=", 7) == 0) {
            const char* v = argv[i] + 7;
            if (strcmp(v, "on") == 0) {
                sortOutput = true;
            } else if (strcmp(v, "off") == 0) {
                sortOutput = false;
            } else {
                fprintf(stderr, "Invalid --sort value: %s (expected on/off)\n", v);
                return 1;
            }
        } else {
            pos.push_back(argv[i]);
        }
    }
    if (pos.size() != 9) {
        fprintf(stderr, "Usage: %s <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <output.csv> [--sort=on|off]\n", argv[0]);
        fprintf(stderr, "  sizeX must be in [1,%d], sizeZ in [1,%d]; --sort defaults to on\n", MAX_SIZE_X, MAX_SIZE_Z);
        return 1;
    }
    auto wall_start = std::chrono::steady_clock::now();
    seed = atoll(pos[0]);
    startX = atoi(pos[1]), startZ = atoi(pos[2]);
    endX = atoi(pos[3]), endZ = atoi(pos[4]);
    sizeX = atoi(pos[5]), sizeZ = atoi(pos[6]);
    threshold = atoi(pos[7]);
    if (threshold < 0) threshold = 0;  // cnt >= 负数 与 cnt >= 0 等价（cnt 恒非负），
                                       // 而归一化后毒值 POISON_CNT 才绝对安全
    const char* outfile = pos[8];
    // 打印用的请求范围（下面的 256 对齐会改 startX/endX/startZ/endZ）
    const int reqStartX = startX, reqEndX = endX, reqStartZ = startZ, reqEndZ = endZ;

    // 硬界校验：这些是上面注释里推导出的实现约束，越界会静默算错或越界写
    if (sizeX < 1 || sizeZ < 1 || sizeX > MAX_SIZE_X || sizeZ > MAX_SIZE_Z) {
        fprintf(stderr, "Invalid rect: sizeX must be in [1,%d] and sizeZ in [1,%d] (got %d,%d)\n",
                MAX_SIZE_X, MAX_SIZE_Z, sizeX, sizeZ);
        return 1;
    }
    // 坐标硬界校验：哨兵坐标 2147483647 必须永远不等于真实坐标（见 MAX_COORD 说明）
    if (llabs((long long)startX) > MAX_COORD || llabs((long long)endX) > MAX_COORD ||
        llabs((long long)startZ) > MAX_COORD || llabs((long long)endZ) > MAX_COORD) {
        fprintf(stderr, "Invalid range: |chunk coord| must be <= %d (got X[%d,%d] Z[%d,%d])\n",
                MAX_COORD, startX, endX, startZ, endZ);
        return 1;
    }

    width = endX - startX + 1;
    height = endZ - startZ + 1;
    if (height < 256) {
        int64_t diff = 256 - height;
        int64_t left = diff / 2;
        int64_t right = diff - left;
        startZ -= (int32_t)left;
        endZ += (int32_t)right;
        height = endZ - startZ + 1;  // 此时 height == 256
    }
    // X 轴对齐到 WIDTH_ALIGN（不影响 Z 轴逻辑）。必须是 WARP_SIZE 的整数倍：前缀核
    // 一行一个 warp、按 `col += WARP_SIZE` 走整行，末尾若不是 32 的倍数，最后一批
    // lane 会不进入循环，全掩码 __shfl_* 就是未定义行为。
    if (width % WIDTH_ALIGN) {
        int w = (width / WIDTH_ALIGN + 1) * WIDTH_ALIGN;
        startX -= (w - width) / 2;
        endX += (w - width + 1) / 2;
        width = w;
    }

    int device_count;
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    const bool useFused = (sizeX <= MAX_FUSED_SIZE_X);  // 与 GPUWorker::run 里的判据一致
    // 输出设备数
    int64_t* H_maxes;
    H_maxes = new int64_t[device_count];
    for (int i = 0; i < device_count; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        if (useFused) {
            /*
             * 融合路径完全不落 d_row_ps，显存不再是 Z 分块的约束：
             *   单卡 => H_max = height，整块 Z 一次到底。这顺带取消了老代码的“Z 范围
             *           向上取整到 step 的倍数”行为（那是为了让多卡均分工作量），所以
             *           实际扫描的 Z 范围不再被放大 —— 请求多少行就扫多少行。
             *   多卡 => Z 方向均分，保持跨卡并行（还是 256 行对齐，≥256）。
             */
            if (device_count == 1) {
                H_maxes[i] = height;
            } else {
                int64_t h = (height + device_count - 1) / device_count;
                h &= ~255LL;
                if (h < 256) h = 256;
                if (h > height) h = height;
                H_maxes[i] = h;
            }
        } else {
            double budget = (double)free_mem * 0.9;
            double bytesPerRow = width * sizeof(int8_t);  // d_row_ps 是 width*H_max 字节
            H_maxes[i] = (int64_t)(budget / bytesPerRow);
            if (H_maxes[i] > height) H_maxes[i] = height;
            H_maxes[i] &= ~255LL;
            // height 已保证 >= 256，所以这里 H_max 至少是一个 256 行的单元，也就恒 > sizeZ
            // （sizeZ <= MAX_SIZE_Z = 32）。必须在取整之后兜底：否则显存极小时 H_max 会被
            // 抹成 0，step 变负，分块偏移与 d_row_ps 尺寸全部失效。
            if (H_maxes[i] < 256) H_maxes[i] = 256;
        }
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s\n", i, prop.name);
        printf("  Total memory: %.2f MB\n", total_mem / (1024.0 * 1024.0));
        printf("  Free memory : %.2f MB\n", free_mem / (1024.0 * 1024.0));
        printf("  Used memory : %.2f MB\n", (total_mem - free_mem) / (1024.0 * 1024.0));
        // 这里只打印设备画像；**并行度阈值不在这里定** —— 它由每个 GPUWorker 在自己的
        // cudaSetDevice 之后调用 tuneParallelismThresholds() 各取各卡的真实值（多卡正确）。
        printf("  SM count: %d\n", prop.multiProcessorCount);
    }

    int64_t* offset;
    offset = new int64_t[device_count];
    offset[0] = 0;
    int64_t total_width = 0;
    for (int i = 0; i < device_count; ++i) {
        if (i > 0) offset[i] = offset[i - 1] + (H_maxes[i - 1] - sizeZ + 1);
        total_width += (H_maxes[i] - sizeZ + 1);
    }
    step = total_width;  // 全局步长

    /*
     * 这里的分块逻辑大概是把所有卡整合成一个大的块，每一块里面才分配给每一张卡，导致height的对齐有点复杂
     * 随便打个简单的比方，比如卡1负责Valid范围0~9的，那么卡2就负责Valid范围10~19的，抽象成一个Valid范围0~19的块
     * 那么下一个大块就是20~39， 其中卡1是20~29，卡2是20~29
     * 假设就这两个大块，那么height并不是39，因为height不是Valid的范围，而是总的范围，之间还差了一个sizeZ
     */
    int64_t valid_height = (int64_t)height - (sizeZ - 1);
    if (valid_height % step != 0) {
        int64_t new_valid_len = ((valid_height + step - 1) / step) * step;
        int64_t diff = new_valid_len - valid_height;
        int64_t left = diff / 2;
        int64_t right = diff - left;
        startZ -= (int32_t)left;
        endZ += (int32_t)right;
        height = endZ - startZ + 1;
    }

    ResultPool pool(POOL_BLOCK_SLOTS, std::max(4, device_count + 2), outfile);
    pool.setProducers(device_count);
    GPUWorker* workers;
    workers = new GPUWorker[device_count];
    for (int i = 0; i < device_count; ++i) {
        // GPUWorker 含 std::atomic，拷贝赋值被删除，所以在原地构造
        new (&workers[i]) GPUWorker(i, &pool, H_maxes[i], offset[i]);
    }

    valid_endZ = (int64_t)endZ - sizeZ + 1;


    printf("Global range: X[%d,%d] Z[%d,%d] -> [%d,%d] Z[%d,%d], rect %dx%d, thr %d, sort %s\n",
           reqStartX, reqEndX, reqStartZ, reqEndZ,
           startX, endX, startZ, endZ, sizeX, sizeZ, threshold, sortOutput ? "on" : "off");

    CUDA_CHECK(cudaMallocHost(&h_baseX, width * sizeof(int64_t)));
    CUDA_CHECK(cudaMallocHost(&h_baseZ, height * sizeof(int64_t)));
    computeBases(startX, width, startZ, height, seed, h_baseX, h_baseZ);

    // 消费者两条路：
    //   --sort=on  -> N 个排序线程：GPU 还在跑就把块排成有序段（CPU 与 GPU 并行、段间并行）
    //   --sort=off -> 唯一写盘线程：边取块边格式化写 CSV（同样与 GPU 重叠）
    FILE* fp = nullptr;
    std::thread writer;
    if (sortOutput) {
        const unsigned hw = std::thread::hardware_concurrency();
        const int nSort = (int)std::min(10u, std::max(1u, hw > 2 ? hw - 2 : 1u));
        pool.startSorters(nSort);
    } else {
        fp = fopen(outfile, "wb");  // 同上：CSV 一律二进制写，跨平台一致
        if (!fp) {
            fprintf(stderr, "Cannot open output file: %s\n", outfile);
            return 1;
        }
        fprintf(fp, "x,z,slime_count\n");
        writer = std::thread([&] {
            std::vector<char> obuf(1 << 20);
            size_t used = 0;
            int idx = -1;
            while (pool.take(idx)) {
                const HitResult* r = pool.data(idx);
                const size_t n = pool.rows(idx);
                for (size_t i = 0; i < n; ++i) {
                    if (used > obuf.size() - 64) {  // 一行最多 ~40 字节，留足余量
                        fwrite(obuf.data(), 1, used, fp);
                        used = 0;
                    }
                    char* p = obuf.data() + used;
                    p = fmtInt(p, r[i].x);
                    *p++ = ',';
                    p = fmtInt(p, r[i].z);
                    *p++ = ',';
                    p = fmtUInt(p, (unsigned)r[i].count);
                    *p++ = '\n';
                    used = (size_t)(p - obuf.data());
                }
                pool.release(idx);
            }
            if (used) fwrite(obuf.data(), 1, used, fp);
        });
    }
    uint64_t totalValid = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < device_count; ++i) {
        threads.emplace_back(&GPUWorker::run, &workers[i]);
    }
#ifdef SLIME_PROGRESS_CODE
    // 进度条：stderr 是终端时默认开；重定向时自动关（见 ProgressTicker）。
    const char* progEnv = getenv("SLIME_PROGRESS");
    const char* noProgEnv = getenv("SLIME_NO_PROGRESS");
    bool showProgress = streamIsTty(stderr);
    if (noProgEnv && *noProgEnv && strcmp(noProgEnv, "0") != 0) showProgress = false;
    if (progEnv && *progEnv && strcmp(progEnv, "0") != 0) showProgress = true;
    ProgressTicker ticker(workers, device_count, showProgress);
    g_ticker = &ticker;
    ticker.start();
#endif
    for (auto& t : threads) t.join();
#ifdef SLIME_PROGRESS_CODE
    ticker.stop();
    g_ticker = nullptr;
#endif
    if (sortOutput) {
        pool.joinSorters();  // 排序线程退出时所有有序段都已就绪
        totalValid = pool.totalRows();
        if (!pool.mergeRunsToCsv(outfile)) return 1;
    } else {
        writer.join();
        totalValid = pool.totalRows();  // sort=off 时 release 已经清过 used 字段，这里用行数统计
        fclose(fp);
    }

#ifdef SLIME_PROFILE
    {
        double tp = 0.0, tw = 0.0;
        for (int i = 0; i < device_count; ++i) {
            tp += workers[i].getProfPrefixMs();
            tw += workers[i].getProfWindowMs();
        }
        printf("PROFILE: row-prefix kernel %.2f ms (%.1f%%), sliding-window + D2H/launch %.2f ms (%.1f%%), block total %.2f ms\n",
               tp, 100.0 * tp / (tp + tw), tw, 100.0 * tw / (tp + tw), tp + tw);
    }
#endif

    double total_gpu_ms = 0.0;
    for (int i = 0; i < device_count; ++i) total_gpu_ms += workers[i].getGpuTime();
    auto wall_end = std::chrono::steady_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    printf("Total valid: %" PRIu64 ", GPU compute time: %.2f ms, wall time: %.2f ms\n",
           totalValid, total_gpu_ms, total_wall_ms);
    cudaFreeHost(h_baseX);
    cudaFreeHost(h_baseZ);
    return 0;
}
