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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
// CUDA运行时头文件
#include <cuda_runtime.h>
#include <stdint.h>

// #define BATCH_SIZE 16384

int32_t radius;
int32_t sizeX;
int32_t sizeZ;
int64_t seed;
int threshold;

int64_t *h_x_old, *h_z_old, *h_x_new, *h_z_new, *h_output;

int task_counter = 0;
int64_t x, z, count;
int64_t* swap_ptr;
bool is_first_batch = true;
int last_batch_cnt = 0;
int processed_cnt = 0;
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
    uint32_t uz2 = uz * uz;                             // 模拟int乘法溢出
    int64_t part3 = (int64_t)(int32_t)uz2 * 4392871LL;  // 先符号扩展为long再乘

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
__global__ void prefix_kernel(
    const int64_t* d_x,
    const int64_t* d_z,
    int64_t* d_prefix,
    int32_t data_x,
    int32_t data_z,
    int32_t radius,
    int64_t seed,
    int32_t batch_size) {
    int task_id = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_id >= batch_size || row >= data_x) return;

    // 该任务对应的中心坐标
    int64_t start_x = d_x[task_id] - radius;
    int64_t start_z = d_z[task_id] - radius;

    // 当前行对应的真实 Z 坐标
    int64_t z = start_z + row;

    // 指向该任务该行的前缀和起始位置
    int64_t* row_prefix = &d_prefix[task_id * data_z * (data_x + 1) + row * (data_x + 1)];

    // 计算前缀和
    int64_t sum = 0;
    row_prefix[0] = 0;  // 边界
    for (int col = 0; col < data_x; col++) {
        int64_t x = start_x + col;
        int64_t val = isSlimeChunk(seed, x, z);
        sum += val;
        row_prefix[col + 1] = sum;
    }
}
__global__ void score_kernel(
    const int64_t* d_prefix,
    int64_t* d_output,
    const int32_t* d_dx_max,
    const int64_t* d_x,
    const int64_t* d_z,
    int32_t sizeX,
    int32_t sizeZ,
    int32_t data_x,
    int32_t data_z,
    int32_t radius,
    int32_t batch_size) {
    int task_id = blockIdx.z;
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cz = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_id >= batch_size || cx >= sizeX || cz >= sizeZ) return;

    const int64_t* task_prefix = &d_prefix[task_id * data_z * (data_x + 1)];

    int64_t score = 0;
    for (int32_t dz = -radius; dz <= radius; dz++) {
        int32_t row = cz + dz + radius;  // 相对于任务起始Z的偏移
        if (row < 0 || row >= data_z) continue;
        int32_t dx_max = d_dx_max[dz + radius];
        int32_t left = cx - dx_max + radius;
        int32_t right = cx + dx_max + radius;
        if (left < 0) left = 0;
        if (right >= data_x) right = data_x - 1;
        if (left <= right) {
            const int64_t* row_prefix = &task_prefix[row * (data_x + 1)];
            score += row_prefix[right + 1] - row_prefix[left];
        }
    }

    int out_idx = task_id * (sizeX * sizeZ) + cx * sizeZ + cz;
    d_output[out_idx] = score;
}
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
std::vector<Record> ans;

bool cmp(Record x, Record y) {
    if (x.x != y.x) return x.x < y.x;
    if (x.z != y.z) return x.z < y.z;
    return x.value < y.value;
}

void handle_output() {
    for (int n = 0; n < last_batch_cnt; n++) {
        for (int i = 0; i < sizeX; i++) {
            for (int j = 0; j < sizeZ; j++) {
                if (h_output[n * sizeX * sizeZ + i * sizeZ + j] >= threshold)
                    ans.push_back(Record{
                        .x = i + h_x_old[n],
                        .z = j + h_z_old[n],
                        .value = h_output[n * sizeX * sizeZ + i * sizeZ + j]});
            }
        }
    }
}

int main(int argc, char* argv[]) {
    CUDA_CHECK(cudaSetDevice(0));
    // 读入参数
    if (argc != 8) {
        printf("Usage: %s <input_csv> <radius> <sizeX> <sizeZ> <seed> <output_csv> <threshold>\n", argv[0]);
        return 0;
    }
    FILE* t = fopen(argv[1], "r");
    radius = atoll(argv[2]);
    sizeX = atoll(argv[3]);
    sizeZ = atoll(argv[4]);
    seed = atoll(argv[5]);
    threshold = atoi(argv[7]);
    // dy_count是圆的大小，随后计算h_dx_max来记录每行需要求和的宽度
    int32_t dy_count = 2 * radius + 1;
    int32_t* h_dx_max = (int32_t*)malloc(dy_count * sizeof(int32_t));
    printf("Circle:");
    for (int32_t dy = -radius; dy <= radius; dy++) {
        // 像素行的有效距离：考虑像素的垂直半宽 0.5
        float y_eff = fabsf((float)dy) - 0.5f;
        if (y_eff < 0.0f) y_eff = 0.0f;

        float val = (float)(radius * radius) - y_eff * y_eff;
        if (val < 0.0f) val = 0.0f;  // 防止负值

        int32_t dx_max = (int32_t)ceilf(sqrtf(val));
        h_dx_max[dy + radius] = dx_max;
        printf("%d ", dx_max);
    }
    printf("\n");
    // 分配设备内存并拷贝
    int32_t* d_dx_max;
    CUDA_CHECK(cudaMalloc(&d_dx_max, dy_count * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_dx_max, h_dx_max, dy_count * sizeof(int32_t), cudaMemcpyHostToDevice));

    int32_t DATA_X = sizeX + 2 * radius + 2;
    int32_t DATA_Z = sizeZ + 2 * radius + 2;
    size_t per_task_bytes = (DATA_Z * (DATA_X + 1) + sizeX * sizeZ) * sizeof(int64_t);
    // printf("%d %d\n", DATA_X, DATA_Z);
    //  双缓冲需要两份，外加一些余量
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("Total memory: %.2FMB\nAvailable free memory: %.2fMB\n",
           total_mem / 1024.0 / 1024.0, free_mem / 1024.0 / 1024.0);
    size_t reserved_mem = 1024 * 1024 * 1024;  // 1GB
    size_t usable_mem;
    if (free_mem > reserved_mem * 2) {
        usable_mem = free_mem - reserved_mem;
    } else {
        usable_mem = (size_t)(free_mem * 0.9);  // 小显存就留10%
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 0 表示设备 0
    printf("maxGridSize[2] = %d\n", prop.maxGridSize[2]);
    int BATCH_SIZE = (int)(usable_mem / (per_task_bytes * 2));  // 双缓冲
    if (BATCH_SIZE < 1) BATCH_SIZE = 1;
    if (BATCH_SIZE > prop.maxGridSize[2]) BATCH_SIZE = prop.maxGridSize[2];  // 防止硬件队列过深
    printf("Batch size set to %d\n", BATCH_SIZE);

    int64_t* d_prefix;
    CUDA_CHECK(cudaMalloc(&d_prefix, BATCH_SIZE * DATA_Z * (DATA_X + 1) * sizeof(int64_t)));

    int64_t *d_x, *d_z;
    int64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_x, BATCH_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_z, BATCH_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * sizeX * sizeZ * sizeof(int64_t)));
    CUDA_CHECK(cudaMallocHost((void**)&h_x_old, sizeof(int64_t) * BATCH_SIZE));
    CUDA_CHECK(cudaMallocHost((void**)&h_z_old, sizeof(int64_t) * BATCH_SIZE));
    CUDA_CHECK(cudaMallocHost((void**)&h_x_new, sizeof(int64_t) * BATCH_SIZE));
    CUDA_CHECK(cudaMallocHost((void**)&h_z_new, sizeof(int64_t) * BATCH_SIZE));
    CUDA_CHECK(cudaMallocHost((void**)&h_output, sizeof(int64_t) * BATCH_SIZE * sizeX * sizeZ));
    double dis;
    while (true) {
        int ret = fscanf(t, "%lld,%lld,%lld,%lf", &x, &z, &count, &dis);
        printf("%lld,%lld,%lld,%lf\n", x, z, count, dis);
        if (ret == 4) {  // x与z为区域左上角标识
            h_x_new[task_counter] = x >> 4;
            h_z_new[task_counter] = z >> 4;
            task_counter++;
        }
        bool should_process = (ret != 4) || (task_counter == BATCH_SIZE);
        if (should_process && task_counter > 0) {
            if (!is_first_batch) {
                cudaDeviceSynchronize();
                cudaMemcpy(h_output, d_output,
                           last_batch_cnt * sizeX * sizeZ * sizeof(int64_t),
                           cudaMemcpyDeviceToHost);
            }
            cudaMemcpy(d_x, h_x_new, task_counter * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_z, h_z_new, task_counter * sizeof(int64_t), cudaMemcpyHostToDevice);

            printf("%lld %lld %lld %lld %lld\n", DATA_X, DATA_Z, radius, seed, task_counter);

            int threads_y = 64;  // 可调，建议 32~128
            dim3 block_prefix(1, threads_y, 1);
            dim3 grid_prefix(1, (DATA_Z + threads_y - 1) / threads_y, task_counter);
            prefix_kernel<<<grid_prefix, block_prefix>>>(
                d_x, d_z, d_prefix, DATA_X, DATA_Z, radius, seed, task_counter);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            cudaDeviceSynchronize();
            /*
            puts("----");
            const int check_task = 0;
            size_t prefix_size = DATA_Z * (DATA_X + 1);  // 元素个数
            int64_t* h_prefix = nullptr;
            CUDA_CHECK(cudaMallocHost(&h_prefix, prefix_size * sizeof(int64_t)));

            CUDA_CHECK(cudaMemcpy(h_prefix,
                                  d_prefix + check_task * prefix_size,  // 偏移到第一个任务
                                  prefix_size * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost));

            // 按行打印，便于观察
            for (int row = 0; row < DATA_Z; row++) {
                for (int col = 0; col < DATA_X + 1; col++) {
                    printf("%lld ", h_prefix[row * (DATA_X + 1) + col]);
                }
                printf("\n");
            }
            puts("\n----");
            */
            int threads_x = 16;
            threads_y = 16;  // 或 8x8, 16x16, 32x4 等
            dim3 block_score(threads_x, threads_y, 1);
            dim3 grid_score((sizeX + threads_x - 1) / threads_x,
                            (sizeZ + threads_y - 1) / threads_y,
                            task_counter);
            score_kernel<<<grid_score, block_score>>>(
                d_prefix, d_output, d_dx_max, d_x, d_z, sizeX, sizeZ, DATA_X, DATA_Z, radius, task_counter);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            if (!is_first_batch) {
                handle_output();
            } else
                is_first_batch = false;
            last_batch_cnt = task_counter;  // 记录本次启动的批次大小（即最后一批）
            swap_ptr = h_x_old;
            h_x_old = h_x_new;
            h_x_new = swap_ptr;
            swap_ptr = h_z_old;
            h_z_old = h_z_new;
            h_z_new = swap_ptr;
            task_counter = 0;
            processed_cnt += last_batch_cnt;
            printf("Finished:%d\n", processed_cnt);
        }
        if (ret != 4) break;
    }
    if (last_batch_cnt > 0) {
        // 等待 GPU 完成最后一批
        cudaDeviceSynchronize();
        // 拷贝结果到 CPU
        cudaMemcpy(h_output, d_output,
                   last_batch_cnt * sizeX * sizeZ * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);

        // 处理后处理（注意：此时 h_x_old, h_z_old 指向最后一批的数据）
        handle_output();
    }

    std::sort(ans.begin(), ans.end(), cmp);

    fclose(t);
    t = fopen(argv[6], "w");
    if (ans.size() != 0) {
        Record r_l = ans[0];
        fprintf(t, "%lld,%lld,%lld\n", r_l.x, r_l.z, r_l.value);
        for (Record r : ans) {
            if (r_l.x == r.x && r_l.z == r.z && r_l.value == r.value) continue;
            r_l = r;
            fprintf(t, "%lld,%lld,%lld\n", r.x * 16, r.z * 16, r.value);
        }
    }
    cudaFree(d_dx_max);
    cudaFree(d_x);
    cudaFree(d_z);
    cudaFree(d_prefix);
    cudaFree(d_output);
    cudaFreeHost(h_x_old);
    cudaFreeHost(h_z_old);
    cudaFreeHost(h_x_new);
    cudaFreeHost(h_z_new);
    cudaFreeHost(h_output);
    free(h_dx_max);
    fclose(t);
}
