/*
 * Copyright 2026 minelogy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * slime_circle —— 给一批候选区域逐个打「圆盘分数」。
 *
 * 输入 CSV：第一行是表头（跳过），之后每行是一个任务 <x>,<z>,...，
 * 其中 x/z 是任务左上角的**方块坐标**（右移 4 位得到区块坐标）。
 * 对每个任务，在它自己的 sizeX × sizeZ 个候选位置上算：
 *
 *     score(i,j) = Σ_{dz=-r..r} Σ_{|dx|<=dx_max[dz]} slime(chunkX0+i+r+dx, chunkZ0+j+r+dz)
 *
 * dx_max[dz] 由主机按「圆盘 {dx²+dy² ≤ r²}」逐行反解（见 main 里的打印），
 * 即候选 (i,j) 的分数 = 以 (i+r, j+r) 为中心的离散圆盘内的史莱姆区块数。
 * 分数 >= threshold 的位置输出为 <x*16>,<z*16>,<score>，整体按 (x,z,score) 排序并去重。
 *
 * 两条 kernel（旧版是「逐行单线程前缀和 + 全量分数回传主机再扫描」，两处都推倒重写）：
 *
 *   1) rowPrefixSumKernel —— 一个 warp 负责一行的**协作前缀和**。
 *      每行每 lane 算一个格子的 0/1，`__ballot_sync` 一次拿到 32 个格子的位图，
 *      于是 lane 的前缀 = carry + popc(位图 & lanemask_le)，5 条指令一格。
 *      前缀以 **uint16** 存：消费者只做 (uint16)(a - b) 求一行上的区间和，
 *      而区间跨度 = 2*dx_max+1 <= 2r+1 < 65536，模 2^16 的差值恒等于真值
 *      （与 slime_main 宽窗回退用 uint8 前缀是同一个论证，这里换成 uint16
 *       是为了不对 radius 加额外限制）。
 *
 *   2) discScoreKernel —— 每个候选一个线程，2(2r+1) 次 shared/global 查表求和，
 *      分数达标就**在设备上**用 atomicAdd 领槽登记（结构性防越界，与 slime_main 同款）。
 *      主机因此只回传命中条目，而不是整个分数矩阵 —— 后者在 rect=1024 时是
 *      每批 3.7 GB 的 D2H，再加一遍 sizeX*sizeZ 的主机侧扫描，是旧版真正的瓶颈。
 *
 * 不丢结果的保证：每批的任务数按「最坏情况（每个候选都命中）也放得下」取
 * （BATCH <= cap / (sizeX*sizeZ)），所以命中缓冲区在结构上不可能溢出；
 * 内核里仍保留 pos < cap 的判界作为防御性兜底，并在主机侧断言 n <= cap。
 */

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "slime_hash.cuh"

// 候选窗口的合法上界（与 slime_main 保持一致）：sizeX 参与 DATA_X/gridDim 计算，
// sizeZ 参与 DATA_Z 计算，没有上界时会整型溢出。
static constexpr int MAX_SIZE_X = 255;
static constexpr int MAX_SIZE_Z = 32;

// 检查CUDA调用错误的宏
#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

struct Record {
    int64_t x;
    int64_t z;
    int64_t value;
};

// 命中条目：坐标直接写成最终输出的方块坐标（省掉主机侧再乘 16）
struct CircleHit {
    int32_t x;
    int32_t z;
    int32_t count;
};

// 每个 block 里负责几行（一行一个 warp）
static constexpr int WPB = 8;
// 每批最多多少个任务：受硬件 gridDim.y 上限与命中缓冲区几何共同约束
static constexpr int MAX_BATCH_TASKS = 65535;
// 命中缓冲区上限（条数）。条目 12 B，64M 条 = 768 MB —— 真的到这一步说明
// threshold 极低、命中比高，此时几何会把 BATCH 压到很小，不会吃满。
static constexpr int64_t MAX_HIT_SLOTS = 64LL << 20;
// 回传命中的 pinned 中转缓冲大小（条数）：4M 条 = 48 MB
static constexpr int64_t STAGE_SLOTS = 4LL << 20;

/*
 * 每行协作前缀和：一个 warp 一行，lane 依次处理 col = lane, lane+32, ...
 * 段内用 ballot 直接得到「含本 lane」的前缀（不用 5 步 shfl 扫描），段间用 carry 累加。
 */
__global__ void rowPrefixSumKernel(
    const int64_t* __restrict__ d_x,        // 每个任务的左上角区块 X
    const int64_t* __restrict__ d_z,        // 每个任务的左上角区块 Z
    uint16_t* __restrict__ d_prefix,        // [task][row][data_x+1]
    int data_x,
    int data_z,
    int radius,
    int64_t seed) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int task = blockIdx.y;
    const int row = blockIdx.x * WPB + warp;
    if (row >= data_z) return;  // 越界行整条 warp 退出，不参与后续 ballot

    const int32_t x0 = (int32_t)(d_x[task] - radius);
    const int32_t z = (int32_t)(d_z[task] - radius) + row;
    const int64_t bz = slimeQuadZ(z);  // 行内不变，一次算好给整条 warp 用

    uint16_t* row_prefix = d_prefix + ((size_t)task * data_z + row) * (size_t)(data_x + 1);
    if (lane == 0) row_prefix[0] = 0;

    uint32_t carry = 0;
    for (int base = 0; base < data_x; base += WARP_SIZE) {
        const int col = base + lane;
        const bool active = col < data_x;
        uint32_t bit = 0;
        if (active) {
            const int64_t bx = slimeQuadX(x0 + col) + seed;
            bit = isSlimeChunk(bx, bz) ? 1u : 0u;
        }
        const uint32_t bal = __ballot_sync(0xFFFFFFFFu, bit != 0u);
        // lane<=31 时 (2u<<lane)-1 给出「含本 lane 的低位掩码」；lane=31 时 2u<<31 回绕成 0，
        // 0-1 = 0xFFFFFFFF，恰好也是「含 lane 31」的全部 32 位。
        const uint32_t prefix_incl = carry + __popc(bal & ((2u << lane) - 1u));
        if (active) row_prefix[col + 1] = (uint16_t)prefix_incl;
        carry += __popc(bal);
    }
}

/*
 * 圆盘打分：一个线程一个候选 (i,j)。每行查两次前缀表得到该行在圆盘内的区间和，
 * 2r+1 行相加即圆盘分数；达标就在设备上登记，主机不再回传整个分数矩阵。
 */
__global__ void discScoreKernel(
    const uint16_t* __restrict__ d_prefix,
    const int32_t* __restrict__ d_dx_max,  // [2r+1]，每行的半宽
    const int64_t* __restrict__ d_x,
    const int64_t* __restrict__ d_z,
    int sizeX,
    int sizeZ,
    int data_x,
    int data_z,
    int radius,
    int threshold,
    CircleHit* __restrict__ d_hits,
    unsigned long long* __restrict__ d_pos,
    unsigned long long cap) {
    // dx_max 表进 shared：每个线程每行都要用，放 shared 只需一次 global 读
    extern __shared__ int32_t s_dx_max[];
    for (int i = threadIdx.x; i < 2 * radius + 1; i += blockDim.x) s_dx_max[i] = d_dx_max[i];
    __syncthreads();

    const int task = blockIdx.z;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // 候选列
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // 候选行
    if (i >= sizeX || j >= sizeZ) return;

    const uint16_t* task_prefix = d_prefix + (size_t)task * data_z * (size_t)(data_x + 1);

    int score = 0;
    for (int dz = -radius; dz <= radius; ++dz) {
        const int row = j + dz + radius;
        if (row < 0 || row >= data_z) continue;
        const int32_t dx_max = s_dx_max[dz + radius];
        int left = i - dx_max + radius;
        int right = i + dx_max + radius;
        if (left < 0) left = 0;
        if (right >= data_x) right = data_x - 1;
        if (left <= right) {
            const uint16_t* row_prefix = task_prefix + (size_t)row * (size_t)(data_x + 1);
            score += (int)(uint16_t)(row_prefix[right + 1] - row_prefix[left]);
        }
    }

    if (score >= threshold) {
        const unsigned long long pos = atomicAdd(d_pos, 1ULL);  // 先领槽再判界
        if (pos < cap) {
            d_hits[pos].x = (int32_t)((d_x[task] + i) * 16);
            d_hits[pos].z = (int32_t)((d_z[task] + j) * 16);
            d_hits[pos].count = score;
        }
    }
}

static bool cmpRecord(const Record& a, const Record& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.z != b.z) return a.z < b.z;
    return a.value < b.value;
}

static bool parse_i64_strict(const char* s, int64_t lo, int64_t hi, int64_t* out) {
    if (!s || !*s) return false;
    errno = 0;
    char* end = NULL;
    const long long v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return false;  // 非法字符 / 溢出
    if (v < lo || v > hi) return false;                        // 越界
    *out = (int64_t)v;
    return true;
}

int main(int argc, char* argv[]) {
    CUDA_CHECK(cudaSetDevice(0));
    if (argc != 8) {
        fprintf(stderr,
                "Usage: %s <input_csv> <radius> <sizeX> <sizeZ> <seed> <output_csv> <threshold>\n",
                argv[0]);
        return 1;
    }
    // 严格解析：atoll/atoi 对 "abc" 静默返回 0 —— 例如 radius=abc 会静默按 0 跑，
    // sizeX=abc 虽被 <=0 挡下，但报的是"Invalid arguments"而不是"参数本身非法"。
    // sizeX/sizeZ 加上界：它们参与 DATA_X/DATA_Z 与 gridDim 计算，无上界会整型溢出。
    int64_t radius64 = 0, sizeX64 = 0, sizeZ64 = 0, seed = 0, threshold64 = 0;
    if (!parse_i64_strict(argv[2], 0, 30000, &radius64) ||
        !parse_i64_strict(argv[3], 1, MAX_SIZE_X, &sizeX64) ||
        !parse_i64_strict(argv[4], 1, MAX_SIZE_Z, &sizeZ64) ||
        !parse_i64_strict(argv[5], INT64_MIN, INT64_MAX, &seed) ||
        !parse_i64_strict(argv[7], INT32_MIN, INT32_MAX, &threshold64)) {
        fprintf(stderr,
                "Invalid arguments: radius='%s' sizeX='%s' sizeZ='%s' seed='%s' threshold='%s'\n"
                "  expected: radius 0..30000, sizeX 1..%d, sizeZ 1..%d, integer seed, int32 threshold\n",
                argv[2], argv[3], argv[4], argv[5], argv[7], MAX_SIZE_X, MAX_SIZE_Z);
        return 1;
    }
    FILE* t = fopen(argv[1], "r");
    if (!t) {
        fprintf(stderr, "Cannot open input file: %s\n", argv[1]);
        return 1;
    }
    const int32_t radius = (int32_t)radius64;
    const int32_t sizeX = (int32_t)sizeX64;
    const int32_t sizeZ = (int32_t)sizeZ64;
    const int threshold = (int)threshold64;

    // dy_count 是圆盘的行数，h_dx_max 记录每行需要求和的半宽
    const int32_t dy_count = 2 * radius + 1;
    int32_t* h_dx_max = (int32_t*)malloc(dy_count * sizeof(int32_t));
    printf("Circle:");
    for (int32_t dy = -radius; dy <= radius; dy++) {
        // 像素行的有效距离：考虑像素的垂直半宽 0.5
        float y_eff = fabsf((float)dy) - 0.5f;
        if (y_eff < 0.0f) y_eff = 0.0f;
        float val = (float)(radius * radius) - y_eff * y_eff;
        if (val < 0.0f) val = 0.0f;
        int32_t dx_max = (int32_t)ceilf(sqrtf(val));
        h_dx_max[dy + radius] = dx_max;
        printf("%d ", dx_max);
    }
    printf("\n");

    int32_t* d_dx_max;
    CUDA_CHECK(cudaMalloc(&d_dx_max, dy_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_dx_max, h_dx_max, dy_count * sizeof(int32_t), cudaMemcpyHostToDevice));

    // 每个任务覆盖的输入网格（含圆盘 halo）；候选是左上角的 sizeX × sizeZ 个位置
    const int32_t DATA_X = sizeX + 2 * radius + 2;
    const int32_t DATA_Z = sizeZ + 2 * radius + 2;
    const size_t prefix_bytes_per_task =
        (size_t)DATA_Z * (size_t)(DATA_X + 1) * sizeof(uint16_t);
    const int64_t candidates_per_task = (int64_t)sizeX * sizeZ;

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("Total memory: %.2FMB\nAvailable free memory: %.2fMB\n",
           total_mem / 1024.0 / 1024.0, free_mem / 1024.0 / 1024.0);

    // 命中缓冲区 + 前缀缓冲都从「可用显存 - 1GB 保留」里出，命中缓冲占 1/4。
    const size_t reserved_mem = 1024 * 1024 * 1024;
    const size_t usable_mem = (free_mem > reserved_mem * 2) ? (free_mem - reserved_mem)
                                                            : (size_t)(free_mem * 0.9);
    int64_t hit_slots = (int64_t)(usable_mem / 4 / sizeof(CircleHit));
    if (hit_slots > MAX_HIT_SLOTS) hit_slots = MAX_HIT_SLOTS;
    // 下限：至少装得下一个任务的最坏情况（每个候选都命中）
    if (hit_slots < candidates_per_task) hit_slots = candidates_per_task;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("maxGridSize[2] = %d\n", prop.maxGridSize[2]);

    // 每批任务数 = min(gridDim.y 上限, 命中缓冲放得下最坏情况, 前缀缓冲放得下)
    int64_t batch = hit_slots / candidates_per_task;
    if (batch > MAX_BATCH_TASKS) batch = MAX_BATCH_TASKS;
    if (batch > prop.maxGridSize[2]) batch = prop.maxGridSize[2];
    const int64_t prefix_budget = (int64_t)(usable_mem / 2 / prefix_bytes_per_task);
    if (batch > prefix_budget) batch = prefix_budget;
    if (batch < 1) batch = 1;
    const int BATCH_SIZE = (int)batch;
    const unsigned long long cap = (unsigned long long)BATCH_SIZE * (unsigned long long)candidates_per_task;
    printf("Batch size set to %d (hit slots %" PRId64 ", cap %llu)\n", BATCH_SIZE, hit_slots, cap);

    uint16_t* d_prefix = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prefix, (size_t)BATCH_SIZE * prefix_bytes_per_task));
    int64_t *d_x = nullptr, *d_z = nullptr;
    CircleHit* d_hits = nullptr;
    unsigned long long* d_pos = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, BATCH_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_z, BATCH_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_hits, (size_t)cap * sizeof(CircleHit)));
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(unsigned long long)));

    int64_t *h_x = nullptr, *h_z = nullptr;
    CircleHit* h_hits = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&h_x, sizeof(int64_t) * BATCH_SIZE));
    CUDA_CHECK(cudaMallocHost((void**)&h_z, sizeof(int64_t) * BATCH_SIZE));
    // pinned 中转缓冲取固定大小、分块回传：按 cap 分配（小矩形时可达 200+ MB）会让
    // cudaMallocHost 变成启动期的主要开销，而它并不需要一次装下整批命中。
    const size_t stage_slots = (size_t)((cap < (STAGE_SLOTS)) ? cap : (STAGE_SLOTS));
    CUDA_CHECK(cudaMallocHost((void**)&h_hits, stage_slots * sizeof(CircleHit)));

    const size_t score_shmem = (size_t)dy_count * sizeof(int32_t);
    if (score_shmem > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(discScoreKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)score_shmem));

    std::vector<Record> ans;
    int task_counter = 0;
    int64_t processed_cnt = 0;
    char line[4096];

    // 与旧版一样的读法：先 fgets 吃掉上一行剩下的换行/首行表头，再用 fscanf 取一条任务
    while (true) {
        if (!fgets(line, sizeof(line), t)) break;
        int64_t x = 0, z = 0, count = 0;
        double dis = 0.0;
        const int ret = fscanf(t, "%" SCNd64 ",%" SCNd64 ",%" SCNd64 ",%lf", &x, &z, &count, &dis);
        if (ret >= 3) {
            h_x[task_counter] = x >> 4;  // x/z 是方块坐标，任务区域用区块坐标表示
            h_z[task_counter] = z >> 4;
            ++task_counter;
        }
        const bool flush_now = (ret < 3) || (task_counter == BATCH_SIZE);
        if (flush_now && task_counter > 0) {
            CUDA_CHECK(cudaMemcpy(d_x, h_x, (size_t)task_counter * sizeof(int64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_z, h_z, (size_t)task_counter * sizeof(int64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_pos, 0, sizeof(unsigned long long)));

            const dim3 block_prefix(WPB * WARP_SIZE);
            const dim3 grid_prefix((unsigned)((DATA_Z + WPB - 1) / WPB), (unsigned)task_counter);
            rowPrefixSumKernel<<<grid_prefix, block_prefix>>>(
                d_x, d_z, d_prefix, DATA_X, DATA_Z, radius, seed);

            const int threads_x = 16, threads_y = 16;
            const dim3 block_score(threads_x, threads_y, 1);
            const dim3 grid_score((unsigned)((sizeX + threads_x - 1) / threads_x),
                                  (unsigned)((sizeZ + threads_y - 1) / threads_y),
                                  (unsigned)task_counter);
            discScoreKernel<<<grid_score, block_score, score_shmem>>>(
                d_prefix, d_dx_max, d_x, d_z, sizeX, sizeZ, DATA_X, DATA_Z, radius, threshold,
                d_hits, d_pos, cap);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned long long n = 0;
            CUDA_CHECK(cudaMemcpy(&n, d_pos, sizeof(n), cudaMemcpyDeviceToHost));
            if (n > cap) {  // 几何上不可能；真出现说明主机端批次估计错了，明确报错而不是丢结果
                fprintf(stderr, "internal: hit buffer overflow (%llu > %llu)\n", n, cap);
                return 1;
            }
            for (unsigned long long off = 0; off < n; off += stage_slots) {
                const size_t chunk = (size_t)((n - off < stage_slots) ? (n - off) : stage_slots);
                CUDA_CHECK(cudaMemcpy(h_hits, d_hits + off, chunk * sizeof(CircleHit),
                                      cudaMemcpyDeviceToHost));
                for (size_t k = 0; k < chunk; ++k)
                    ans.push_back(Record{h_hits[k].x, h_hits[k].z, h_hits[k].count});
            }
            processed_cnt += task_counter;
            printf("Finished:%" PRId64 "\n", processed_cnt);
            task_counter = 0;
        }
        if (ret < 3) break;
    }
    fclose(t);

    std::sort(ans.begin(), ans.end(), cmpRecord);
    FILE* out = fopen(argv[6], "wb");
    if (!out) {
        fprintf(stderr, "Cannot open output file: %s\n", argv[6]);
        return 1;
    }
    fprintf(out, "x,z,slime_count\n");
    for (size_t i = 0; i < ans.size(); ++i) {
        if (i > 0 && ans[i].x == ans[i - 1].x && ans[i].z == ans[i - 1].z &&
            ans[i].value == ans[i - 1].value)
            continue;  // 任务重叠时同一位置可能被算多次，去重
        fprintf(out, "%" PRId64 ",%" PRId64 ",%" PRId64 "\n", ans[i].x, ans[i].z, ans[i].value);
    }
    fclose(out);

    CUDA_CHECK(cudaFree(d_dx_max));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_prefix));
    CUDA_CHECK(cudaFree(d_hits));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_z));
    CUDA_CHECK(cudaFreeHost(h_hits));
    free(h_dx_max);
    return 0;
}
