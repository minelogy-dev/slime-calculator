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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

/*
 * 严格整数解析：整串必须是一个十进制整数，且落在 [min,max] 内。
 * 原实现用 atoi/atoll：输入 "abc" 会被静默当成 0（例如 <distance>=abc 会变成
 * 「不限制距离」），"12x" 会被当成 12 —— 都属于静默改变语义。
 * 返回 1 = 成功（*out 已写），0 = 失败。
 */
static int parse_ll_strict(const char* s, long long lo, long long hi, long long* out) {
    if (!s || !*s) return 0;
    errno = 0;
    char* end = NULL;
    const long long v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return 0;  // 非法字符 / 溢出
    if (v < lo || v > hi) return 0;                        // 越界
    *out = v;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_csv> <distance> <threshold> <output_csv>\n", argv[0]);
        return 1;
    }
    long long max_dist = 0, min_count = 0;
    if (!parse_ll_strict(argv[2], 0, LLONG_MAX, &max_dist)) {
        fprintf(stderr, "Error: invalid <distance> '%s' (expected a non-negative integer)\n", argv[2]);
        return 1;
    }
    if (!parse_ll_strict(argv[3], LLONG_MIN, LLONG_MAX, &min_count)) {
        fprintf(stderr, "Error: invalid <threshold> '%s' (expected an integer)\n", argv[3]);
        return 1;
    }
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", argv[1]);
        return 1;
    }
    char line[4096];
    // 先开输出文件再决定是否提前返回：空输入也要产出「只有表头」的文件（与旧行为一致，
    // 但旧实现根本没写出这个文件），并且不检查 fopen 结果 —— 不可写时会段错误。
    FILE *out = fopen(argv[4], "w");
    if (!out) {
        fprintf(stderr, "Error: cannot open %s for writing\n", argv[4]);
        fclose(fp);
        return 1;
    }
    fprintf(out, "x,z,slime_count,distance\n");
    if (!fgets(line, sizeof(line), fp)) {   // 空输入 / 只有不可读的首行
        fclose(fp);
        if (fclose(out) != 0) {
            fprintf(stderr, "Error: failed to flush %s\n", argv[4]);
            return 1;
        }
        return 0;
    }
    long long x, z, count;
    long long n_parsed = 0, n_kept = 0, n_bad = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        if (sscanf(line, "%lld,%lld,%lld", &x, &z, &count) != 3) {
            ++n_bad;    // 旧实现静默跳过；这里只计数，结束时汇总到 stderr（不改输出内容）
            continue;
        }
        ++n_parsed;
        // 注：x、z 是 long long，x*x 本身就是 64 位乘法，不存在 int 截断问题
        double dist = sqrt((double)x * (double)x + (double)z * (double)z);
        if (dist > (double)max_dist || count < min_count) continue;
        fprintf(out, "%lld,%lld,%lld,%.6f\n", x, z, count, dist);
        ++n_kept;
    }
    fclose(fp);
    if (fclose(out) != 0) {
        fprintf(stderr, "Error: failed to flush %s\n", argv[4]);
        return 1;
    }
    if (n_bad > 0)
        fprintf(stderr, "Warning: %lld malformed line(s) skipped (%lld parsed, %lld kept)\n",
                n_bad, n_parsed, n_kept);
    return 0;
}
