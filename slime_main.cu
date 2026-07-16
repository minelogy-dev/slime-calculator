/*
 * Copyright 2026 minelogy
 * Licensed under the Apache License, Version 2.0
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cinttypes>
#include <chrono>

struct HitResult {
    int x, z, count;
};
static_assert(__is_pod(HitResult), "HitResult must be POD for zero-overhead");
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
bool useWidePath;

/* 史莱姆区块判定（worldSeed 已预加至 baseX） */
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

#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

// 快速路径：单线程一行（用于中小宽度）
#define ROW_SCAN_BLOCK 256
__global__ void computeRowPrefixSumFast(
    const int64_t* __restrict__ d_baseX, const int64_t* __restrict__ d_baseZ,
    int width, int blockHeight, int baseZ_offset,
    int32_t* __restrict__ d_row_ps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    //if (row < blockHeight) {
        int64_t base = (int64_t)row * width;
        int32_t sum = 0;
        for (int col = 0; col < width; ++col) {
            sum += isSlimeChunk(__ldg(d_baseX + col), __ldg(d_baseZ + baseZ_offset + row));
            d_row_ps[base + col] = sum;
        }
    //}
}

// 超大宽度协作行前缀和（Warp Shuffle，作用于全局宽度）
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8  // 每块 256 线程
__global__ void computeRowPrefixSumWide(
    const int64_t* __restrict__ d_baseX, const int64_t* __restrict__ d_baseZ,
    int width, int blockHeight, int baseZ_offset,
    int32_t* __restrict__ d_row_ps) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warpId;
    //if (row >= blockHeight) return;

    int64_t bz = __ldg(d_baseZ + baseZ_offset + row);
    int64_t rowBase = (int64_t)row * width;
    int prefix = 0;

    for (int col = lane; col < width; col += WARP_SIZE) {
        int64_t bx = __ldg(d_baseX + col);
        int val = isSlimeChunk(bx, bz);

// Warp 内 inclusive scan
#pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            int n = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val += n;
        }
        val += prefix;
        d_row_ps[rowBase + col] = val;
        prefix = __shfl_sync(0xffffffff, val, WARP_SIZE - 1);
    }
}

/* 合并滑动窗口输出（与之前完全一致） */
__global__ void slidingWindowOutputKernel(
    const int32_t* __restrict__ d_row_ps,
    int rowPsWidth, int blockHeight, int sizeX, int sizeZ,
    int threshold, int startX, int baseZ,
    int batchValidStartZ, int batchValidEndZ,
    int validRangeX,
    HitResult* d_results, int* d_pos) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= validRangeX) return;
    int x = startX + j;
    int localRight = j + sizeX - 1;
    int localLeft = j - 1;
    int winVals[20], sum = 0;
    for (int row = 0; row < sizeZ; ++row) {
        int64_t rowBase = (int64_t)row * rowPsWidth;
        int val = __ldg(d_row_ps + rowBase + localRight) -
                  (localLeft >= 0 ? __ldg(d_row_ps + rowBase + localLeft) : 0);
        sum += val;
        winVals[row] = val;
    }
    int ringIdx = 0;
    for (int i = 0; i <= blockHeight - sizeZ; ++i) {
        int z = baseZ + i;
        if (z >= batchValidStartZ && z <= batchValidEndZ && sum >= threshold) {
            int pos = atomicAdd(d_pos, 1);
            d_results[pos].x = x * 16;
            d_results[pos].z = z * 16;
            d_results[pos].count = sum;
        }
        if (i == blockHeight - sizeZ) break;
        sum -= winVals[ringIdx];
        int nextRow = i + sizeZ;
        int64_t nextRowBase = (int64_t)nextRow * rowPsWidth;
        int newVal = __ldg(d_row_ps + nextRowBase + localRight) -
                     (localLeft >= 0 ? __ldg(d_row_ps + nextRowBase + localLeft) : 0);
        sum += newVal;
        winVals[ringIdx] = newVal;
        ringIdx = (ringIdx + 1) % sizeZ;
    }
}

void computeBases(int32_t xStart, int32_t width, int32_t zStart, int32_t height,
                  int64_t seed, int64_t* h_baseX, int64_t* h_baseZ) {
    for (int i = 0; i < width; ++i) {
        int32_t x = xStart + i;
        uint32_t ux = (uint32_t)x;
        h_baseX[i] = (int64_t)(int32_t)((ux * ux) * 4987142U) + (int64_t)(int32_t)(ux * 5947611U) + seed;
    }
    for (int i = 0; i < height; ++i) {
        int32_t z = zStart + i;
        uint32_t uz = (uint32_t)z;
        h_baseZ[i] = (int64_t)(int32_t)((uz * uz)) * 4392871LL + (int64_t)(int32_t)(uz * 389711U);
    }
}

// 只在Z轴上切分，所以StartX, StartZ等均为共用的，使用全局变量管理
// 每个需要维护的只有自己的输出缓冲区，使用vector, 自己的H_max与偏移，把全图切成多块
class GPUWorker {
   public:
    GPUWorker() : device_id(0), output(nullptr), H_max(0), offset(0) {}
    GPUWorker(int device_id, std::vector<HitResult>* output, int64_t H_max, int64_t offset) : device_id(device_id), output(output), H_max(H_max), offset(offset) {}
    void run() {
        cudaSetDevice(device_id);

        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));  // 用于计时的两个事件
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        HitResult *h_results = nullptr, *d_results = nullptr;
        int rangeX = width - sizeX + 1;
        int64_t maxOutputsPerBlock = rangeX * (H_max - sizeZ + 1);
        CUDA_CHECK(cudaMallocHost(&h_results, maxOutputsPerBlock * sizeof(HitResult)));
        CUDA_CHECK(cudaMalloc(&d_results, maxOutputsPerBlock * sizeof(HitResult)));

        int64_t *d_baseX, *d_baseZ;
        CUDA_CHECK(cudaMalloc(&d_baseX, width * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_baseZ, height * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(d_baseX, h_baseX, width * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_baseZ, h_baseZ, height * sizeof(int64_t), cudaMemcpyHostToDevice));

        int32_t* d_row_ps;
        CUDA_CHECK(cudaMalloc(&d_row_ps, width * H_max * sizeof(int32_t)));
        int32_t* d_pos;
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int32_t)));

        for (int64_t baseZ = startZ; baseZ <= valid_endZ; baseZ += step) {  // 这一段对于所有卡都是一样的，也就是说明这是对于全局而言的大块概念
            int64_t blockStartZ = baseZ + offset;                           // 定位到在这个大块中本卡的任务起始点
            int64_t blockEndZ = blockStartZ + H_max - 1;                    // 根据高度为H_max算出本次任务的终点
            int64_t blockHeight = H_max;                                    // 本次任务的高度
            // 有效输出窗口起始点范围 [blockStartZ, blockStartZ + H_max - sizeZ]（闭区间）
            int64_t validStartZ = blockStartZ;                // 起始与本次任务的起始保持一致
            int64_t validEndZ = validStartZ + H_max - sizeZ;  // 但是结尾要短一点，短sizeZ-1

            CUDA_CHECK(cudaEventRecord(ev_start));

            // 1. 行前缀和
            if (useWidePath) {
                int threads = WARPS_PER_BLOCK * WARP_SIZE;
                int gridRows = (int)((blockHeight + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
                computeRowPrefixSumWide<<<gridRows, threads>>>(
                    d_baseX, d_baseZ, width, (int)blockHeight, (int)(blockStartZ - startZ), d_row_ps);
            } else {
                int gridRows = (int)((blockHeight + ROW_SCAN_BLOCK - 1) / ROW_SCAN_BLOCK);
                computeRowPrefixSumFast<<<gridRows, ROW_SCAN_BLOCK>>>(
                    d_baseX, d_baseZ, width, (int)blockHeight, (int)(blockStartZ - startZ), d_row_ps);
            }

            // 2. 滑动窗口输出
            CUDA_CHECK(cudaMemset(d_pos, 0, sizeof(int)));
            int blockDim = (useWidePath ? WARPS_PER_BLOCK * WARP_SIZE : ROW_SCAN_BLOCK);
            int gridCols = (rangeX + blockDim - 1) / blockDim;
            slidingWindowOutputKernel<<<gridCols, blockDim>>>(
                d_row_ps, width, (int)blockHeight, sizeX, sizeZ,
                threshold, startX, (int)blockStartZ,
                (int)validStartZ, (int)validEndZ, rangeX,
                d_results, d_pos);

            CUDA_CHECK(cudaDeviceSynchronize());

            int numValid;
            CUDA_CHECK(cudaMemcpy(&numValid, d_pos, sizeof(int), cudaMemcpyDeviceToHost));
            if (numValid > 0) {
                CUDA_CHECK(cudaMemcpy(h_results, d_results, numValid * sizeof(HitResult), cudaMemcpyDeviceToHost));
                output->insert(output->end(), h_results, h_results + numValid);
            }

            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            printf("Block Z[%" PRId64 ",%" PRId64 "] height %" PRId64 ", valid Z[%" PRId64 ",%" PRId64 "] -> %d valid, %.2f ms\n",
                   blockStartZ, blockEndZ, blockHeight, validStartZ, validEndZ, numValid, ms);
            gpu_time_ms += ms;
        }
        CUDA_CHECK(cudaFree(d_baseX));
        CUDA_CHECK(cudaFree(d_baseZ));
        CUDA_CHECK(cudaFree(d_row_ps));
        CUDA_CHECK(cudaFree(d_pos));
        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFreeHost(h_results));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
    }
    const std::vector<HitResult>& getResults() const {
        return *output;
    }
    double getGpuTime() const { return gpu_time_ms; }

   private:
    int device_id;
    std::vector<HitResult>* output;
    int64_t H_max;
    int64_t offset;
    double gpu_time_ms = 0.0;
};

int main(int argc, char* argv[]) {
    if (argc != 10) {
        fprintf(stderr, "Usage: %s <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <output.csv>\n", argv[0]);
        return 1;
    }
    auto wall_start = std::chrono::steady_clock::now();
    seed = atoll(argv[1]);
    startX = atoi(argv[2]), startZ = atoi(argv[3]);
    endX = atoi(argv[4]), endZ = atoi(argv[5]);
    sizeX = atoi(argv[6]), sizeZ = atoi(argv[7]);
    threshold = atoi(argv[8]);
    const char* outfile = argv[9];

    width = endX - startX + 1;
    height = endZ - startZ + 1;
    if (height < 256) {
        int64_t diff = 256 - height;
        int64_t left = diff / 2;
        int64_t right = diff - left;
        startZ -= (int32_t)left;
        endZ   += (int32_t)right;
        height = endZ - startZ + 1;  // 此时 height == 256
    }
    // X 轴对齐到 256（不影响 Z 轴逻辑）
    if (width % 256) {
        int w = (width / 256 + 1) * 256;
        startX -= (w - width) / 2;
        endX += (w - width + 1) / 2;
        width = w;
    }
    useWidePath = (width > 262144);  // 大宽度使用 warp 协作

    int device_count;
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    // 输出设备数
    int64_t H_maxes[device_count];
    std::vector<HitResult> results[device_count];
    for (int i = 0; i < device_count; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        double budget = (double)free_mem * 0.9;
        double bytesPerRow = width * sizeof(int32_t);
        H_maxes[i] = (int64_t)(budget / bytesPerRow);
        if (H_maxes[i] < sizeZ) H_maxes[i] = sizeZ;
        if (H_maxes[i] > height) H_maxes[i] = height;
        H_maxes[i] &= ~255LL;
        // 输出设备基本信息，如名称，显存大小，已用大小
    }

    int64_t offset[device_count];
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

    GPUWorker workers[device_count];
    for (int i = 0; i < device_count; ++i) {
        workers[i] = GPUWorker(i, results + i, H_maxes[i], offset[i]);
    }

    valid_endZ = (int64_t)endZ - sizeZ + 1;

    printf("Global range: X[%d,%d] Z[%d,%d] -> [%d,%d] Z[%d,%d], rect %dx%d, thr %d\n",
           atoi(argv[2]), atoi(argv[4]), atoi(argv[3]), atoi(argv[5]),
           startX, endX, startZ, endZ, sizeX, sizeZ, threshold);

    CUDA_CHECK(cudaMallocHost(&h_baseX, width * sizeof(int64_t)));
    CUDA_CHECK(cudaMallocHost(&h_baseZ, height * sizeof(int64_t)));
    computeBases(startX, width, startZ, height, seed, h_baseX, h_baseZ);

    std::vector<std::thread> threads;
    for (int i = 0; i < device_count; ++i) {
        threads.emplace_back(&GPUWorker::run, &workers[i]);
    }
    for (auto& t : threads) t.join();

    // 合并所有 output vector 到文件
    uint64_t totalValid = 0;
    double total_gpu_ms = 0.0;
    FILE* fp = fopen(outfile, "w");
    fprintf(fp, "x,z,slime_count\n");
    for (int i = 0; i < device_count; ++i) {
        for (const auto& res : workers[i].getResults()) {
            fprintf(fp, "%d,%d,%d\n", res.x, res.z, res.count);
        }
        totalValid += workers[i].getResults().size();
        total_gpu_ms += workers[i].getGpuTime();
    }
    fclose(fp);
    auto wall_end = std::chrono::steady_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    printf("Total valid: %lld, GPU compute time: %.2f ms, wall time: %.2f ms\n",
       totalValid, total_gpu_ms, total_wall_ms);
    cudaFreeHost(h_baseX);
    cudaFreeHost(h_baseZ);
    return 0;
}
