/*
 * Copyright 2026 minelogy
 * Licensed under the Apache License, Version 2.0
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

__constant__ int64_t c_worldSeed;   // 种子已预加到 baseX，不再使用

/* 史莱姆区块判定（worldSeed 已预加至 baseX） */
__device__ __forceinline__ int isSlimeChunk(int64_t baseX, int64_t baseZ) {
    uint64_t acc = (uint64_t)baseX + (uint64_t)baseZ;
    acc ^= 987234911ULL;
    uint64_t seed48 = (acc ^ 0x5DEECE66DULL) & 0xFFFFFFFFFFFFULL;
    const uint64_t MULT = 0x5DEECE66DULL, ADD = 0xBULL;
    const uint32_t REJECT = 2147483640;
    seed48 = (seed48 * MULT + ADD) & 0xFFFFFFFFFFFFULL;
    uint32_t u = (uint32_t)(seed48 >> 17);
    uint32_t q = (uint32_t)(((uint64_t)u * 0xCCCCCCCDULL) >> 35);
    uint32_t r = u - q * 10;
    if (u < REJECT) return (r == 0) ? 1 : 0;
    const int m = 9;
    do {
        seed48 = (seed48 * MULT + ADD) & 0xFFFFFFFFFFFFULL;
        u = (uint32_t)(seed48 >> 17);
        q = (uint32_t)(((uint64_t)u * 0xCCCCCCCDULL) >> 35);
        r = u - q * 10;
    } while ((int32_t)(u - r + m) < 0);
    return (r == 0) ? 1 : 0;
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
    int32_t* __restrict__ d_row_ps)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < blockHeight) {
        int64_t base = (int64_t)row * width;
        int32_t sum = 0;
        for (int col = 0; col < width; ++col) {
            sum += isSlimeChunk(__ldg(d_baseX + col), __ldg(d_baseZ + baseZ_offset + row));
            d_row_ps[base + col] = sum;
        }
    }
}

// 超大宽度协作行前缀和（Warp Shuffle，作用于全局宽度）
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8   // 每块 256 线程
__global__ void computeRowPrefixSumWide(
    const int64_t* __restrict__ d_baseX, const int64_t* __restrict__ d_baseZ,
    int width, int blockHeight, int baseZ_offset,
    int32_t* __restrict__ d_row_ps)
{
    int warpId = threadIdx.x / WARP_SIZE;
    int lane   = threadIdx.x % WARP_SIZE;
    int row    = blockIdx.x * WARPS_PER_BLOCK + warpId;
    if (row >= blockHeight) return;

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
    int* d_out_x, int* d_out_z, int* d_out_count, int* d_pos)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= validRangeX) return;
    int x = startX + j;
    int localRight = j + sizeX - 1;
    int localLeft  = j - 1;
    int winVals[20], sum = 0;
    for (int row = 0; row < sizeZ; ++row) {
        int64_t rowBase = (int64_t)row * rowPsWidth;
        int val = __ldg(d_row_ps + rowBase + localRight) -
                  (localLeft >= 0 ? __ldg(d_row_ps + rowBase + localLeft) : 0);
        sum += val; winVals[row] = val;
    }
    int ringIdx = 0;
    for (int i = 0; i <= blockHeight - sizeZ; ++i) {
        int z = baseZ + i;
        if (z >= batchValidStartZ && z < batchValidEndZ && sum >= threshold) {
            int pos = atomicAdd(d_pos, 1);
            d_out_x[pos] = x; d_out_z[pos] = z; d_out_count[pos] = sum;
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
        h_baseX[i] = (int64_t)(int32_t)((ux * ux) * 4987142U)
                   + (int64_t)(int32_t)(ux * 5947611U) + seed;
    }
    for (int i = 0; i < height; ++i) {
        int32_t z = zStart + i;
        uint32_t uz = (uint32_t)z;
        h_baseZ[i] = (int64_t)(int32_t)((uz * uz)) * 4392871LL
                   + (int64_t)(int32_t)(uz * 389711U);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 10) {
        fprintf(stderr, "Usage: %s <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <output.csv>\n", argv[0]);
        return 1;
    }
    int64_t seed = atoll(argv[1]);
    int32_t startX = atoi(argv[2]), startZ = atoi(argv[3]);
    int32_t endX = atoi(argv[4]), endZ = atoi(argv[5]);
    int32_t sizeX = atoi(argv[6]), sizeZ = atoi(argv[7]);
    int threshold = atoi(argv[8]);
    const char* outfile = argv[9];

    int32_t width = endX - startX + 1, height = endZ - startZ + 1;
    // 对齐到 256
    if (width % 256) { int w = (width/256+1)*256; startX -= (w-width)/2; endX += (w-width+1)/2; width = w; }
    if (height % 256) { int h = (height/256+1)*256; startZ -= (h-height)/2; endZ += (h-height+1)/2; height = h; }

    printf("Global range: X[%d,%d] Z[%d,%d] -> [%d,%d] Z[%d,%d], rect %dx%d, thr %d\n",
           atoi(argv[2]), atoi(argv[4]), atoi(argv[3]), atoi(argv[5]),
           startX, endX, startZ, endZ, sizeX, sizeZ, threshold);

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpyToSymbol(c_worldSeed, &seed, sizeof(seed)));

    int64_t *d_baseX, *d_baseZ;
    CUDA_CHECK(cudaMalloc(&d_baseX, width * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_baseZ, height * sizeof(int64_t)));
    int64_t *h_baseX = (int64_t*)malloc(width * sizeof(int64_t));
    int64_t *h_baseZ = (int64_t*)malloc(height * sizeof(int64_t));
    computeBases(startX, width, startZ, height, seed, h_baseX, h_baseZ);
    CUDA_CHECK(cudaMemcpy(d_baseX, h_baseX, width*sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_baseZ, h_baseZ, height*sizeof(int64_t), cudaMemcpyHostToDevice));
    free(h_baseX); free(h_baseZ);

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    double budget = free_mem * 0.8;
    double bytesPerRow = width * sizeof(int32_t);
    int64_t H_max = (int64_t)(budget / bytesPerRow);
    if (H_max < sizeZ) H_max = sizeZ;
    if (H_max > height) H_max = height;
    H_max &= ~255LL;
    int64_t stepZ = H_max - (sizeZ - 1);
    if (stepZ <= 0) { fprintf(stderr, "Z step too small.\n"); return 1; }
    printf("Budget %.2f MB, Z block height %lld, stepZ %lld\n",
           budget/1048576.0, H_max, stepZ);

    int32_t* d_row_ps;
    CUDA_CHECK(cudaMalloc(&d_row_ps, width * H_max * sizeof(int32_t)));
    int rangeX = width - sizeX + 1;
    int *d_out_x, *d_out_z, *d_out_count, *d_pos;
    CUDA_CHECK(cudaMalloc(&d_out_x, rangeX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_z, rangeX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_count, rangeX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    double total_time = 0.0;
    int64_t totalValid = 0;
    FILE* fp = fopen(outfile, "w");
    fprintf(fp, "x,z,slime_count\n");

    int64_t globalValidEndZ = (int64_t)endZ - sizeZ + 1;
    bool useWidePath = (width > 262144);   // 大宽度使用 warp 协作

    for (int64_t baseZ = startZ; baseZ <= globalValidEndZ; baseZ += stepZ) {
        int64_t blockEndZ = baseZ + H_max - 1;
        if (blockEndZ > (int64_t)endZ + sizeZ - 1) blockEndZ = (int64_t)endZ + sizeZ - 1;
        int64_t blockHeight = blockEndZ - baseZ + 1;
        if (blockHeight < sizeZ) break;

        int64_t validStartZ = baseZ;
        int64_t validEndZ = baseZ + stepZ;
        if (validEndZ > globalValidEndZ + 1) validEndZ = globalValidEndZ + 1;

        printf("Block Z[%lld,%lld] height %lld, valid Z[%lld,%lld) ...\n",
               baseZ, blockEndZ, blockHeight, validStartZ, validEndZ-1);

        CUDA_CHECK(cudaEventRecord(ev_start));

        // 1. 行前缀和
        if (useWidePath) {
            int threads = WARPS_PER_BLOCK * WARP_SIZE;
            int gridRows = (int)((blockHeight + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
            computeRowPrefixSumWide<<<gridRows, threads>>>(
                d_baseX, d_baseZ, width, (int)blockHeight, (int)(baseZ - startZ), d_row_ps);
        } else {
            int gridRows = (int)((blockHeight + ROW_SCAN_BLOCK - 1) / ROW_SCAN_BLOCK);
            computeRowPrefixSumFast<<<gridRows, ROW_SCAN_BLOCK>>>(
                d_baseX, d_baseZ, width, (int)blockHeight, (int)(baseZ - startZ), d_row_ps);
        }

        // 2. 滑动窗口输出
        CUDA_CHECK(cudaMemset(d_pos, 0, sizeof(int)));
        int blockDim = (useWidePath ? WARPS_PER_BLOCK*WARP_SIZE : ROW_SCAN_BLOCK);
        int gridCols = (rangeX + blockDim - 1) / blockDim;
        slidingWindowOutputKernel<<<gridCols, blockDim>>>(
            d_row_ps, width, (int)blockHeight, sizeX, sizeZ,
            threshold, startX, (int)baseZ,
            (int)validStartZ, (int)validEndZ, rangeX,
            d_out_x, d_out_z, d_out_count, d_pos);

        CUDA_CHECK(cudaDeviceSynchronize());

        int numValid;
        CUDA_CHECK(cudaMemcpy(&numValid, d_pos, sizeof(int), cudaMemcpyDeviceToHost));
        if (numValid > 0) {
            int *h_x = (int*)malloc(numValid*sizeof(int));
            int *h_z = (int*)malloc(numValid*sizeof(int));
            int *h_cnt = (int*)malloc(numValid*sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_x, d_out_x, numValid*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_z, d_out_z, numValid*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_cnt, d_out_count, numValid*sizeof(int), cudaMemcpyDeviceToHost));
            for (int i=0; i<numValid; ++i)
                fprintf(fp, "%d,%d,%d\n", h_x[i]*16, h_z[i]*16, h_cnt[i]);
            free(h_x); free(h_z); free(h_cnt);
            totalValid += numValid;
        }

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_time += ms;
        printf("  -> %d valid, %.2f ms\n", numValid, ms);
    }

    printf("Total valid: %lld, total time: %.2f ms\n", totalValid, total_time);

    CUDA_CHECK(cudaFree(d_baseX)); CUDA_CHECK(cudaFree(d_baseZ));
    CUDA_CHECK(cudaFree(d_row_ps));
    CUDA_CHECK(cudaFree(d_out_x)); CUDA_CHECK(cudaFree(d_out_z));
    CUDA_CHECK(cudaFree(d_out_count)); CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaEventDestroy(ev_start)); CUDA_CHECK(cudaEventDestroy(ev_stop));
    fclose(fp);
    return 0;
}
