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

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_csv> <distance> <threshold> <output_csv>\n", argv[0]);
        return 1;
    }
    int max_dist = atoi(argv[2]);
    int min_count = atoi(argv[3]);
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", argv[1]);
        return 1;
    }
    char line[4096];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 0;
    }
    FILE *out = fopen(argv[4], "w");
    fprintf(out, "x,z,slime_count,distance\n");
    long long x, z, count;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        if (sscanf(line, "%lld,%lld,%lld", &x, &z, &count) != 3) continue;
        double dist = sqrt(x * x + z * z);
        if (dist > max_dist || count < min_count) continue;
        fprintf(out, "%lld,%lld,%lld,%.6f\n", x, z, count, dist);
    }
    fclose(fp);
    return 0;
}
