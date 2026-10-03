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
#
# Output is pure ASCII (it must survive a headless box with no UTF-8 locale).

set -euo pipefail
mkdir -p build

# ---- device code 目标（JIT 友好）------------------------------------------------
# 默认构建 = 本机架构的 SASS **加上**同一 compute capability 的 PTX。
#   * 驱动优先用精确匹配的 SASS（本机性能与只出 SASS 时完全相同，PTX 不参与）；
#   * 换到**更新**架构的卡时，驱动用内嵌 PTX 做 JIT ⇒ 不必重新编译就能跑
#     （PTX JIT 是前向兼容的：compute_89 的 PTX 可在 sm_90/sm_100/... 上 JIT）。
#   * 换到**更老**架构（如 sm_80/sm_86）时 PTX 也用不上，需要显式指定该架构，例如
#       CUDA_ARCH="-gencode arch=compute_80,code=sm_80" ./build.sh
# 说明：`-arch=native` 只产 SASS、无法附带 PTX，所以要显式探测 compute capability。
detect_ptx_arch() {
    local cc=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)
    fi
    if [[ "$cc" =~ ^[0-9]{2,3}$ ]]; then
        echo "compute_${cc}"
    else
        echo ""   # 探测不到（无 GPU 的构建机）就退回只出 SASS，行为与旧版一致
    fi
}

PTX_ARCH=$(detect_ptx_arch)
if [[ -n "${CUDA_ARCH:-}" ]]; then
    ARCH_FLAGS="$CUDA_ARCH"
    echo "=== device code: CUDA_ARCH override: $ARCH_FLAGS ==="
elif [[ -n "$PTX_ARCH" ]]; then
    ARCH_FLAGS="-arch=native -gencode arch=${PTX_ARCH},code=compute_${PTX_ARCH#compute_}"
    echo "=== device code: SASS(native) + PTX(${PTX_ARCH}) -- newer cards can JIT ==="
else
    ARCH_FLAGS="-arch=native"
    echo "=== device code: SASS(native) only (compute capability not detected) ==="
fi

echo "=== Building slime_cmp (C) ==="
gcc -o build/slime_cmp src/slime_cmp.c -lm -O3 -march=native -mtune=native -flto -fwhole-program

echo "=== Building slime_main (CUDA) ==="
nvcc -o build/slime_main src/slime_main.cu -O3 -use_fast_math ${ARCH_FLAGS} -Xcompiler="-O3 -march=native"

echo "=== Building slime_circle (CUDA) ==="
nvcc -o build/slime_circle src/slime_circle.cu -O3 -use_fast_math ${ARCH_FLAGS} -Xcompiler="-O3 -march=native"

echo "Done. Binaries in build/"
