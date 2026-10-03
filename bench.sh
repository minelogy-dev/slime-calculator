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
# slime_main benchmark: multi-workload speed regression.
#
# Output is pure ASCII (it must survive a headless box with no UTF-8 locale).
# Usage:
#   ./bench.sh [--quick|--default|--full] [--reps N] [--save FILE] [--baseline FILE]
#              [--bin PATH] [--seed N] [--timeout SEC] [--outdir DIR] [--dry-run]
#
# Notes:
#   * Every workload is warmed up once (not counted), then run --reps times; the table reports
#     min / median / mean.
#   * Speed is reported in B/s (billions of candidate centres per second). A candidate centre is
#     every window top-left position inside the requested region, i.e.
#     (endX-startX+1) * (endZ-startZ+1); the conversion is cells_M / ms == B/s.
#     B/s_min uses the fastest run, B/s_avg the mean over --reps. Two caveats:
#     (1) the internal scan range expands at both ends to align with step (see the padding
#     notes below), so this counts the REQUESTED region and is up to ~12% lower than the number
#     of centres actually checked; (2) B/s depends strongly on workload shape and H_max, so
#     compare it only within one workload, never across workloads.
#   * The standard test state of this repository is "GPU starting at 35 degC + SM 2340 MHz /
#     memory 9001 MHz locked"; unlocked, the same code runs ~19% faster, so absolute seconds
#     are only comparable within that state.
#   * wall_ms is the program's own end-to-end wall clock (includes CSV writing and VRAM
#     allocation); gpu_ms sums the per-device, per-block CUDA event times. On multi-GPU runs
#     gpu_ms can exceed wall_ms -- they mean different things, do not divide one by the other.
#   * Each repetition is compared by md5 of the output CSV sorted with `LC_ALL=C sort`, and the
#     det column marks whether the results are identical. The order is decided by atomicAdd and
#     is not stable, so sorting is mandatory; and the sort MUST use the C locale, because under
#     e.g. zh_CN.UTF-8 the `,`/`-` characters are ignored at the primary collation level, which
#     makes `sort` lose its total order and produced false DIFF reports.
#   * The Z-direction block length H_max comes from free VRAM at runtime, and the scan range
#     expands at both ends to align with step. So only compare versions on the same machine in a
#     similar memory state; for strict comparability pick workloads whose height is a multiple
#     of 256 and height <= 0.9*free/width (then H_max == height, the expansion is 0, and the
#     scan range does not depend on memory).
#   * Workload thresholds are deliberately high (sparse hits). The batch allocator prints
#     "Batch overflow" and exits(EXIT_FAILURE) when the hit density times windows-per-column
#     gets too large, so for low thresholds with a large Z range, shrink the Z range first.
#   * Large workloads occupy about 90% of free VRAM by design (d_row_ps = width * H_max); this
#     is expected.
#
# Environment variables: SLIME_BIN / SLIME_TIMEOUT replace --bin / --timeout.

set -euo pipefail

BIN="${SLIME_BIN:-build/slime_main}"
REPS=3
MODE=default
TIMEOUT="${SLIME_TIMEOUT:-3600}"
SEED=114514
SAVE=""
BASELINE=""
OUTDIR=""
DRY_RUN=0

usage() {
    cat <<'EOF'
slime_main benchmark: multi-workload speed regression.

Usage:
  ./bench.sh [options]

Options:
  --quick            only the two small workloads (a few seconds each)
  --default          small + X-dominant + Z-dominant (the default)
  --full             also the README 3.75M x 3.75M workload (~37 s per run on a 4060 Ti at 2340 MHz)
  --reps N           measurement repetitions per workload, default 3 (plus one warmup)
  --save FILE        write the per-run results to a CSV
  --baseline FILE    compare against an earlier --save file and print the delta
  --bin PATH         slime_main executable to use (default build/slime_main)
  --seed N           world seed, default 114514
  --timeout SEC      per-run timeout, default 3600
  --outdir DIR       keep each workload's CSV and log there (default: temp dir, removed on exit)
  --dry-run          print the commands that would run, execute nothing
  -h, --help         show this help

Workload format: <name> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold>
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)    MODE=quick;   shift ;;
        --default)  MODE=default; shift ;;
        --full)     MODE=full;    shift ;;
        --reps)     REPS="$2";    shift 2 ;;
        --save)     SAVE="$2";    shift 2 ;;
        --baseline) BASELINE="$2"; shift 2 ;;
        --bin)      BIN="$2";     shift 2 ;;
        --seed)     SEED="$2";    shift 2 ;;
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        --outdir)   OUTDIR="$2";  shift 2 ;;
        --dry-run)  DRY_RUN=1;    shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--reps must be a positive integer (got: $REPS)" >&2
    exit 2
fi
if ! [[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--timeout must be a positive integer (got: $TIMEOUT)" >&2
    exit 2
fi

# 负载表。coord 为区块坐标；阈值选得让命中稀疏，避免触发批次溢出。
WL_QUICK=(
    "rect17_300k   -150000 -150000  150000  150000  17 17  55"
    "rect32_100k    -50000  -50000   50000   50000  32 32 150"
)
WL_DEFAULT=(
    "${WL_QUICK[@]}"
    "xwide_2Mx32k -1000000  -16000 1000000   16000  17 17  55"
    "ztall_32kx256k   -16000 -256000   16000  256000  17 17  55"
)
WL_FULL=(
    "${WL_DEFAULT[@]}"
    "readme_3M75  -1875000 -1875000 1875000 1875000  17 17  55"
)

case "$MODE" in
    quick)   WORKLOADS=("${WL_QUICK[@]}") ;;
    default) WORKLOADS=("${WL_DEFAULT[@]}") ;;
    full)    WORKLOADS=("${WL_FULL[@]}") ;;
esac

if [[ -n "$OUTDIR" ]]; then
    mkdir -p "$OUTDIR"
    RAW="$OUTDIR"
    KEEP_RAW=1
else
    RAW="$(mktemp -d)"
    KEEP_RAW=0
fi
cleanup() { [[ "$KEEP_RAW" -eq 1 ]] || rm -rf "$RAW"; }
trap cleanup EXIT

if [[ "$DRY_RUN" -eq 0 && ! -x "$BIN" ]]; then
    echo "executable not found: $BIN (run ./build.sh first, or pass --bin/SLIME_BIN)" >&2
    exit 1
fi

RESULTS="$RAW/results.csv"
echo "workload,seed,x0,z0,x1,z1,sizeX,sizeZ,threshold,rep,wall_ms,gpu_ms,valid,csv_md5" > "$RESULTS"

# ---------- 运行一次 ----------
# 结果写入全局变量: RUN_WALL / RUN_GPU / RUN_VALID / RUN_MD5
run_once() {
    local name="$1" csv="$2"; shift 2
    local log="$RAW/${name}.log"
    local cmd=("$BIN" "$@" "$csv")

    if [[ "$DRY_RUN" -eq 1 ]]; then
        RUN_WALL=""; RUN_GPU=""; RUN_VALID=""; RUN_MD5=""
        printf 'DRY-RUN  %s\n' "${cmd[*]}"
        return 0
    fi

    local rc=0
    timeout "$TIMEOUT" "${cmd[@]}" > "$log" 2>&1 || rc=$?
    if [[ "$rc" -ne 0 ]]; then
        if [[ "$rc" -eq 124 ]]; then
            echo "timed out after ${TIMEOUT}s: ${cmd[*]}" >&2
        else
            echo "run failed (exit $rc): ${cmd[*]}" >&2
        fi
        tail -n 20 "$log" >&2 || true
        return 1
    fi

    local line
    line="$(grep -E '^Total valid: ' "$log" | tail -n 1 || true)"
    RUN_WALL="$(sed -n 's/.*wall time: \([0-9.]*\) ms.*/\1/p' <<<"$line")"
    RUN_GPU="$(sed -n 's/.*GPU compute time: \([0-9.]*\) ms.*/\1/p' <<<"$line")"
    RUN_VALID="$(sed -n 's/^Total valid: \([0-9]*\).*/\1/p' <<<"$line")"
    if [[ -z "$RUN_WALL" || -z "$RUN_GPU" || -z "$RUN_VALID" ]]; then
        echo "cannot parse the timing line from output: ${cmd[*]}" >&2
        tail -n 20 "$log" >&2 || true
        return 1
    fi
    RUN_MD5="$(LC_ALL=C sort "$csv" | md5sum | cut -d' ' -f1)"
    return 0
}

# ---------- 统计辅助 ----------
minof() { printf '%s\n' $1 | sort -n | awk 'NR == 1 { v = $1 } END { printf "%.2f", v }'; }
medof() { printf '%s\n' $1 | sort -n | awk '{a[NR]=$1} END { if (NR % 2) printf "%.2f", a[(NR+1)/2]; else printf "%.2f", (a[NR/2] + a[NR/2+1]) / 2 }'; }
meanof() { printf '%s\n' $1 | awk '{s += $1; n++} END { printf "%.2f", s / n }'; }
baseline_min() {
    [[ -n "$BASELINE" && -f "$BASELINE" ]] || return 0
    awk -F, -v k="$1" 'NR > 1 && $1 == k { v = $11 + 0; if (m == "" || v < m) m = v } END { if (m != "") printf "%.2f", m }' "$BASELINE"
}

declare -A WALLS GPUS VALIDS MD5SET MCELLS THRS RECTS

for spec in "${WORKLOADS[@]}"; do
    # shellcheck disable=SC2086
    set -- $spec
    name="$1"; x0="$2"; z0="$3"; x1="$4"; z1="$5"; sx="$6"; sz="$7"; thr="$8"
    csv="$RAW/${name}.csv"

    RECTS[$name]="${sx}x${sz}"
    THRS[$name]="$thr"
    MCELLS[$name]="$(awk -v a="$x0" -v b="$z0" -v c="$x1" -v d="$z1" \
        'BEGIN { printf "%.1f", ((c - a + 1) * (d - b + 1)) / 1e6 }')"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        run_once "$name" "$csv" "$SEED" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr"
        continue
    fi

    printf '>>> %-15s %-7s region %s Mcells thr %s warmup...\n' "$name" "${RECTS[$name]}" "${MCELLS[$name]}" "$thr"
    run_once "$name" "$csv" "$SEED" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr"

    for ((rep = 1; rep <= REPS; ++rep)); do
        run_once "$name" "$csv" "$SEED" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr"
        printf '    rep %d/%d  wall %8s ms  gpu %8s ms  valid %s\n' \
               "$rep" "$REPS" "$RUN_WALL" "$RUN_GPU" "$RUN_VALID"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s\n' \
               "$name" "$SEED" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr" \
               "$rep" "$RUN_WALL" "$RUN_GPU" "$RUN_VALID" "$RUN_MD5" >> "$RESULTS"
        WALLS[$name]="${WALLS[$name]:-} $RUN_WALL"
        GPUS[$name]="${GPUS[$name]:-} $RUN_GPU"
        VALIDS[$name]="$RUN_VALID"
        if [[ -z "${MD5SET[$name]:-}" ]]; then
            MD5SET[$name]="$RUN_MD5"
        elif [[ "${MD5SET[$name]}" != "$RUN_MD5" ]]; then
            MD5SET[$name]="DIFF"
        fi
    done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "dry-run done (no workload was executed)."
    exit 0
fi

# ---------- 汇总 ----------
echo
if [[ -n "$SAVE" ]]; then
    cp "$RESULTS" "$SAVE"
    echo "per-run results written to $SAVE"
fi
if [[ "$KEEP_RAW" -eq 1 ]]; then
    echo "raw CSVs and logs kept in $OUTDIR"
fi

if [[ -x "$BIN" ]]; then
    printf 'binary : %s (md5 %s)\n' "$BIN" "$(md5sum "$BIN" | cut -d' ' -f1)"
fi
if git -C "$(dirname "$0")" rev-parse --short HEAD >/dev/null 2>&1; then
    printf 'git    : %s%s\n' \
        "$(git -C "$(dirname "$0")" rev-parse --short HEAD)" \
        "$(git -C "$(dirname "$0")" diff --quiet 2>/dev/null || echo ' (dirty)')"
fi
printf 'model  : %s, reps=%d, seed=%d, timeout=%ds\n\n' "$MODE" "$REPS" "$SEED" "$TIMEOUT"

hdr="%-16s %-7s %11s %5s %10s %10s %10s %10s %9s %5s %8s %8s %10s"
row="%-16s %-7s %11s %5s %10s %10s %10s %10s %9s %5s %8s %8s %10s"
printf "$hdr\n" workload rect cells_M thr wall_min wall_med wall_avg gpu_min valid det B/s_min B/s_avg base_delta
printf "$hdr\n" ---------------- ------- ----------- ----- ---------- ---------- ---------- ---------- --------- ----- -------- -------- ----------

for spec in "${WORKLOADS[@]}"; do
    # shellcheck disable=SC2086
    set -- $spec
    name="$1"
    wmin="$(minof "${WALLS[$name]}")"
    wmed="$(medof "${WALLS[$name]}")"
    wavg="$(meanof "${WALLS[$name]}")"
    gmin="$(minof "${GPUS[$name]}")"
    det="ok"; [[ "${MD5SET[$name]}" == "DIFF" ]] && det="DIFF"
    # 候选中心数(cells_M, 百万) / 毫秒 = 十亿/秒(B/s)
    bmin="$(awk -v m="${MCELLS[$name]}" -v w="$wmin" 'BEGIN { if (w > 0) printf "%.2f", m / w; else print "-" }')"
    bavg="$(awk -v m="${MCELLS[$name]}" -v w="$wavg" 'BEGIN { if (w > 0) printf "%.2f", m / w; else print "-" }')"
    delta="-"
    if [[ -n "$BASELINE" && -f "$BASELINE" ]]; then
        b="$(baseline_min "$name")"
        if [[ -n "$b" ]]; then
            delta="$(awk -v c="$wmin" -v o="$b" 'BEGIN { if (o > 0) printf "%+.1f%%", (c - o) * 100 / o; else print "-" }')"
        fi
    fi
    printf "$row\n" "$name" "${RECTS[$name]}" "${MCELLS[$name]}" "${THRS[$name]}" \
           "$wmin" "$wmed" "$wavg" "$gmin" "${VALIDS[$name]}" "$det" "$bmin" "$bavg" "$delta"
done

echo
echo "note: wall_* is the program's own end-to-end wall clock (ms); gpu_min sums the per-block CUDA events;"
echo "      det=DIFF means the output CSV differed for identical input -- the timing is void."