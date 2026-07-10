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

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>

// CUDA运行时头文件
#include <cuda_runtime.h>
#include <stdint.h>

/**
 * 判断给定的世界种子和区块坐标是否为史莱姆区块。
 * 返回值：1 表示是史莱姆区块，0 表示不是。
 */
int __device__ isSlimeChunk(int64_t worldSeed, int32_t chunkX, int32_t chunkZ) {
    /* ---- 第 1 步：按照 Java 表达式计算种子（保留所有溢出） ---- */
    uint32_t ux = (uint32_t)chunkX;
    uint32_t ux2 = ux * ux;
    uint32_t upart1 = ux2 * 4987142U;
    int64_t part1 = (int64_t)(int32_t)upart1;

    uint32_t upart2 = ux * 5947611U;
    int64_t part2 = (int64_t)(int32_t)upart2;

    uint32_t uz = (uint32_t)chunkZ;
    uint32_t uz2 = uz * uz;                     // 模拟int乘法溢出
    int64_t part3 = (int64_t)(int32_t)uz2 * 4392871LL; // 先符号扩展为long再乘

    uint32_t upart4 = uz * 389711U;
    int64_t part4 = (int64_t)(int32_t)upart4;

    uint64_t acc = (uint64_t)worldSeed;
    acc += (uint64_t)part1;
    acc += (uint64_t)part2;
    acc += (uint64_t)part3;
    acc += (uint64_t)part4;
    acc ^= 987234911ULL;

    int64_t seed64 = (int64_t)acc;

    /* ---- 第 2 步：Java Random 的种子初始化 ---- */
    uint64_t seed48 = (uint64_t)seed64 ^ 0x5DEECE66DULL;
    seed48 &= 0xFFFFFFFFFFFFULL;

    /* ---- 第 3 步：实现 nextInt(10) 的第一次调用 ---- */
    const uint64_t multiplier = 0x5DEECE66DULL;
    const uint64_t addend = 0xBULL;
    const uint64_t mask48 = 0xFFFFFFFFFFFFULL;
    const int bound = 10;
    const int m = bound - 1;

    uint32_t u, r;
    uint32_t temp;

    do {
        seed48 = (seed48 * multiplier + addend) & mask48;
        u = (uint32_t)(seed48 >> 17);
        r = u % bound;
        temp = u - r + m;
    } while (temp >= 0x80000000U);

    return (r == 0) ? 1 : 0;
}

// 检查CUDA调用错误的宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 内核：计算每个区块是否为史莱姆区块
__global__ void computeSlimeChunks(int64_t seed, int32_t startX, int32_t startZ,
                                   int32_t width, int32_t height, int* d_orig) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && z < height) {
        int64_t idx = (int64_t)z * width + x;
        int32_t worldX = startX + x;
        int32_t worldZ = startZ + z;
        d_orig[idx] = isSlimeChunk(seed, worldX, worldZ);
    }
}

// 内核：计算每行的前缀和
__global__ void rowPrefixSum(int* d_orig, int* d_row_ps, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height) {
        int64_t base = (int64_t)row * width;
        int sum = 0;
        for (int col = 0; col < width; ++col) {
            sum += d_orig[base + col];
            d_row_ps[base + col] = sum;
        }
    }
}

// 内核：根据行前缀和计算最终二维前缀和
__global__ void finalPrefixSum(int* d_row_ps, int* d_ps, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        int64_t stride = (int64_t)width + 1;
        int sum = 0;
        for (int row = 0; row < height; ++row) {
            sum += d_row_ps[(int64_t)row * width + col];
            int64_t idx = ((int64_t)row + 1) * stride + (col + 1);
            d_ps[idx] = sum;
        }
    }
}

// 内核（第一阶段）：统计满足条件的矩形个数（块内）
__global__ void countValidRects(int* d_ps, int startX, int startZ,
                                int width, int height,
                                int sizeX, int sizeZ,
                                int threshold,
                                int rangeX, int rangeZ,
                                int* d_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rangeX * rangeZ) return;

    int j = idx % rangeX;   // 列偏移 (x方向)
    int i = idx / rangeX;   // 行偏移 (z方向)

    int64_t stride = (int64_t)width + 1;
    int64_t tl = (int64_t)i * stride + j;
    int64_t tr = (int64_t)i * stride + (j + sizeX);
    int64_t bl = ((int64_t)i + sizeZ) * stride + j;
    int64_t br = ((int64_t)i + sizeZ) * stride + (j + sizeX);

    int count = d_ps[br] - d_ps[tr] - d_ps[bl] + d_ps[tl];
    if (count >= threshold) {
        atomicAdd(d_count, 1);
    }
}

// 内核（第二阶段）：存储满足条件的矩形坐标和史莱姆数量
__global__ void storeValidRects(int* d_ps, int startX, int startZ,
                                int width, int height,
                                int sizeX, int sizeZ,
                                int threshold,
                                int rangeX, int rangeZ,
                                int* d_pos, int* d_results, int* d_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rangeX * rangeZ) return;

    int j = idx % rangeX;
    int i = idx / rangeX;

    int64_t stride = (int64_t)width + 1;
    int64_t tl = (int64_t)i * stride + j;
    int64_t tr = (int64_t)i * stride + (j + sizeX);
    int64_t bl = ((int64_t)i + sizeZ) * stride + j;
    int64_t br = ((int64_t)i + sizeZ) * stride + (j + sizeX);

    int count = d_ps[br] - d_ps[tr] - d_ps[bl] + d_ps[tl];
    if (count >= threshold) {
        int pos = atomicAdd(d_pos, 1);
        int worldX = startX + j;
        int worldZ = startZ + i;
        d_results[2 * pos] = worldX;
        d_results[2 * pos + 1] = worldZ;
        d_counts[pos] = count;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 10) {
        fprintf(stderr, "Usage: %s <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <output.csv>\n", argv[0]);
        fprintf(stderr, "All coordinates are in chunk coordinates (integers).\n");
        return 1;
    }

    // 解析命令行参数
    int64_t seed = atoll(argv[1]);
    int32_t startX = atoi(argv[2]);
    int32_t startZ = atoi(argv[3]);
    int32_t endX   = atoi(argv[4]);
    int32_t endZ   = atoi(argv[5]);
    int32_t sizeX  = atoi(argv[6]);
    int32_t sizeZ  = atoi(argv[7]);
    int threshold  = atoi(argv[8]);
    const char* outfile = argv[9];

    // 计算搜索范围尺寸
    int32_t width  = endX - startX + 1;
    int32_t height = endZ - startZ + 1;
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Invalid search area: end must be >= start.\n");
        return 1;
    }
    if (sizeX <= 0 || sizeZ <= 0) {
        fprintf(stderr, "Rectangle size must be positive.\n");
        return 1;
    }

    // 整个范围允许的矩形左上角数量
    int32_t rangeX_global = width - sizeX + 1;
    int32_t rangeZ_global = height - sizeZ + 1;
    if (rangeX_global <= 0 || rangeZ_global <= 0) {
        fprintf(stderr, "No possible rectangle of given size within search area.\n");
        return 1;
    }

    printf("Global search range: X [%d, %d] Z [%d, %d], rectangle size %dx%d, threshold %d\n",
           startX, endX, startZ, endZ, sizeX, sizeZ, threshold);
    printf("Total possible rectangles: %lld\n", (int64_t)rangeX_global * rangeZ_global);

    // 选择GPU设备
    CUDA_CHECK(cudaSetDevice(0));

    // ----- 动态计算分块高度 -----
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // 使用 80% 的可用显存作为安全预算
    double budget = free_mem * 0.8;
    // 估算每行所需内存（字节）
    double bytes_per_row_orig = width * sizeof(int);               // d_orig
    double bytes_per_row_rowps = width * sizeof(int);              // d_row_ps
    double bytes_per_row_ps = (width + 1) * sizeof(int);           // d_ps 每行（ps 是 width+1 列）
    // 总内存 ≈ H * (orig + rowps) + (H+1) * ps ≈ H * (orig+rowps+ps) + ps
    // 因此 H_max = floor( (budget - ps) / (orig+rowps+ps) )
    double denominator = bytes_per_row_orig + bytes_per_row_rowps + bytes_per_row_ps;
    int64_t H_max = 0;
    if (denominator > 0) {
        double numerator = budget - bytes_per_row_ps;  // 减去第一行 ps 的固定开销
        if (numerator > 0) {
            H_max = (int64_t)(numerator / denominator);
        }
    }
    // 确保 H_max 至少为 sizeZ，且不超过整体高度
    if (H_max < sizeZ) H_max = sizeZ;
    if (H_max > height) H_max = height;
    // 同时考虑一个合理的安全上限（例如 20000），避免启动过多线程块
    const int64_t MAX_SAFE_HEIGHT = 20000;
    if (H_max > MAX_SAFE_HEIGHT) H_max = MAX_SAFE_HEIGHT;

    printf("Available free memory: %.2f MB, using budget %.2f MB, calculated block height: %lld\n",
           free_mem/1024.0/1024.0, budget/1024.0/1024.0, H_max);
    // ----- 动态分块高度计算结束 -----

    // 打开输出文件（先清空，写入标题行）
    FILE* fp_out = fopen(outfile, "w");
    if (!fp_out) {
        perror("Failed to open output file");
        return 1;
    }
    fprintf(fp_out, "x,z,slime_count\n");
    fclose(fp_out);

    // 分块处理 Z 方向
    int64_t totalValid = 0;
    int64_t step = H_max - (sizeZ - 1);  // 确保步长 > 0
    if (step <= 0) {
        fprintf(stderr, "Calculated block height %lld is not greater than sizeZ-1 (%d).\n",
                H_max, sizeZ - 1);
        return 1;
    }

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    double total_time = 0.0;

    for (int64_t baseZ = startZ; baseZ <= (int64_t)endZ - sizeZ + 1; baseZ += step) {
        CUDA_CHECK(cudaEventRecord(start, 0));  // 开始计时

        // 计算当前块的结束 Z（包含），需要覆盖到 endZ + sizeZ - 1
        int64_t block_endZ = baseZ + H_max - 1;
        if (block_endZ > (int64_t)endZ + sizeZ - 1)
            block_endZ = (int64_t)endZ + sizeZ - 1;

        int64_t block_height = block_endZ - baseZ + 1;
        if (block_height < sizeZ) break;  // 剩余高度不足以形成矩形

        // 块内矩形左上角的 Z 范围
        int64_t block_rangeZ = block_height - sizeZ + 1;
        int64_t block_rangeX = rangeX_global;  // X 方向不分块

        int64_t block_total_rects = block_rangeX * block_rangeZ;
        if (block_total_rects <= 0) continue;

        printf("Processing block: Z=[%lld, %lld] height=%lld, rangeZ=%lld, total rects in block=%lld\n",
               baseZ, block_endZ, block_height, block_rangeZ, block_total_rects);

        // 分配设备内存（基于块大小）
        int *d_orig, *d_row_ps, *d_ps;
        size_t orig_bytes = (size_t)width * block_height * sizeof(int);
        size_t ps_bytes = (size_t)(width + 1) * (block_height + 1) * sizeof(int);

        CUDA_CHECK(cudaMalloc(&d_orig, orig_bytes));
        CUDA_CHECK(cudaMalloc(&d_row_ps, orig_bytes));
        CUDA_CHECK(cudaMalloc(&d_ps, ps_bytes));
        CUDA_CHECK(cudaMemset(d_ps, 0, ps_bytes));

        // 内核1：计算块内每个区块的史莱姆标志
        dim3 block1(16, 16);
        dim3 grid1((width + block1.x - 1) / block1.x,
                   (block_height + block1.y - 1) / block1.y);
        computeSlimeChunks<<<grid1, block1>>>(seed, startX, (int32_t)baseZ,
                                               width, (int32_t)block_height, d_orig);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 内核2：行前缀和
        int block2 = 256;
        int grid2 = (block_height + block2 - 1) / block2;
        rowPrefixSum<<<grid2, block2>>>(d_orig, d_row_ps, width, (int32_t)block_height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 内核3：最终前缀和
        int block3 = 256;
        int grid3 = (width + block3 - 1) / block3;
        finalPrefixSum<<<grid3, block3>>>(d_row_ps, d_ps, width, (int32_t)block_height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 释放 d_orig 和 d_row_ps（不再需要）
        CUDA_CHECK(cudaFree(d_orig));
        CUDA_CHECK(cudaFree(d_row_ps));

        // 第一阶段计数（块内）
        int *d_count;
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

        int blockCount = 256;
        int gridCount = (block_total_rects + blockCount - 1) / blockCount;
        countValidRects<<<gridCount, blockCount>>>(d_ps, startX, (int32_t)baseZ,
                                                   width, (int32_t)block_height,
                                                   sizeX, sizeZ,
                                                   threshold,
                                                   (int32_t)block_rangeX, (int32_t)block_rangeZ,
                                                   d_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int numValidBlock;
        CUDA_CHECK(cudaMemcpy(&numValidBlock, d_count, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  Found %d valid rectangles in this block.\n", numValidBlock);
        totalValid += numValidBlock;

        // 第二阶段：存储结果（如果有）
        if (numValidBlock > 0) {
            int *d_results, *d_pos, *d_counts;
            CUDA_CHECK(cudaMalloc(&d_results, numValidBlock * 2 * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_counts, numValidBlock * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_pos, 0, sizeof(int)));

            storeValidRects<<<gridCount, blockCount>>>(d_ps, startX, (int32_t)baseZ,
                                                       width, (int32_t)block_height,
                                                       sizeX, sizeZ,
                                                       threshold,
                                                       (int32_t)block_rangeX, (int32_t)block_rangeZ,
                                                       d_pos, d_results, d_counts);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // 拷贝结果到主机
            int* h_results = (int*)malloc(numValidBlock * 2 * sizeof(int));
            int* h_counts = (int*)malloc(numValidBlock * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_results, d_results, numValidBlock * 2 * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_counts, d_counts, numValidBlock * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            // 追加到输出文件
            FILE* fp_append = fopen(outfile, "a");
            if (!fp_append) {
                perror("Failed to open output file for appending");
                free(h_results);
                free(h_counts);
                CUDA_CHECK(cudaFree(d_results));
                CUDA_CHECK(cudaFree(d_counts));
                CUDA_CHECK(cudaFree(d_pos));
                CUDA_CHECK(cudaFree(d_ps));
                CUDA_CHECK(cudaFree(d_count));
                return 1;
            }
            for (int i = 0; i < numValidBlock; ++i) {
                // 输出方块坐标（区块坐标乘以16）和史莱姆数量
                fprintf(fp_append, "%d,%d,%d\n", h_results[2*i] * 16, h_results[2*i+1] * 16, h_counts[i]);
            }
            fclose(fp_append);
            free(h_results);
            free(h_counts);

            CUDA_CHECK(cudaFree(d_results));
            CUDA_CHECK(cudaFree(d_counts));
            CUDA_CHECK(cudaFree(d_pos));
        }

        // 释放块内前缀和数组
        CUDA_CHECK(cudaFree(d_ps));
        CUDA_CHECK(cudaFree(d_count));

        // 记录并输出块处理时间
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("  Block processing time: %.2f ms\n", milliseconds);
        total_time += milliseconds;
    }

    printf("Total valid rectangles found: %lld\n", totalValid);
    printf("Total processing time: %.2f ms\n", total_time);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
