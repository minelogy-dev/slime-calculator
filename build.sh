#!/bin/bash
#
# Copyright 2026 minelogy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
mkdir -p build

echo "=== Building slime_cmp (C) ==="
gcc -o build/slime_cmp slime_cmp.c -lm -O3 -march=native -mtune=native -flto -fwhole-program

echo "=== Building slime_main (CUDA) ==="
nvcc -o build/slime_main slime_main.cu -O3 -use_fast_math -arch=native -Xcompiler="-O3 -march=native"

echo "=== Building slime_circle (CUDA) ==="
nvcc -o build/slime_circle slime_circle.cu -O3 -use_fast_math -arch=native -Xcompiler="-O3 -march=native"

echo "Done. Binaries in build/"
