#!/bin/bash
#
# quickbench.sh -- fast speed test for slime_main.
#
# ASCII output only.  Reports GPU time (sum of the per-chunk CUDA events) and
# wall time, plus throughput in B/s (billions of candidate centres per second,
# where cells_M / ms == B/s).
#
#   usage: quickbench.sh [options]
#     --bin PATH     binary under test          (default <root>/build/slime_main)
#     --ref PATH     optional reference binary; when given, runs are interleaved
#                    A/B/A/B and a ratio column is printed (>1 means BIN faster)
#     -n N           repetitions per workload   (default 2)
#     --scale        add the scaling workload   (~6 s of GPU per run)
#     --full         add the README full map    (~36 s of GPU per run)
#     --seed N       world seed                 (default 114514)
#     --sort on|off  output mode                (default on)
#     --root DIR     project root (auto-detected by default)
#     --no-wait      do not wait for the standard GPU start state
#     -h, --help
#
# Absolute times drift by 2-3% with machine state, so only ever compare numbers
# measured in the same session; that is what --ref does (interleaved A/B).
#
set -uo pipefail

BIN=""; REF=""; REPS=2; SEED=114514; SORTM=on; ROOT=""; DOWAIT=1
ADD_SCALE=0; ADD_FULL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)     BIN="$2"; shift 2 ;;
        --ref)     REF="$2"; shift 2 ;;
        -n)        REPS="$2"; shift 2 ;;
        --scale)   ADD_SCALE=1; shift ;;
        --full)    ADD_FULL=1; shift ;;
        --seed)    SEED="$2"; shift 2 ;;
        --sort)    SORTM="$2"; shift 2 ;;
        --root)    ROOT="$2"; shift 2 ;;
        --no-wait) DOWAIT=0; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---- locate the project root (a directory containing src/slime_main.cu) ----
find_root() {
    local d; d="$(cd "$(dirname "$0")" && pwd)"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/src/slime_main.cu" ]] && { echo "$d"; return 0; }
        d="$(dirname "$d")"
    done
    return 1
}
if [[ -z "$ROOT" ]]; then
    if [[ -n "${SLIME_ROOT:-}" ]]; then ROOT="$SLIME_ROOT"
    else ROOT="$(find_root)" || { echo "cannot find project root (no src/slime_main.cu above me); use --root" >&2; exit 2; }
    fi
fi
[[ -z "$BIN" ]] && BIN="$ROOT/build/slime_main"
[[ -x "$BIN" ]] || { echo "binary not found or not executable: $BIN (run ./build.sh first)" >&2; exit 2; }
if [[ -n "$REF" && ! -x "$REF" ]]; then echo "reference not executable: $REF" >&2; exit 2; fi

WORK="$ROOT/.work/quickbench"; mkdir -p "$WORK"
cleanup() { rm -f "$WORK"/*.csv "$WORK"/*.log; }
trap cleanup EXIT

# ---- workloads: name | x0 z0 x1 z1 | sizeX sizeZ | threshold ----
WL=(
  "rect17_300k|-150000 -150000 150000 150000|17 17|55"
  "xwide_2Mx32k|-1000000 -16000 1000000 16000|17 17|55"
  "ztall_32kx256k|-16000 -256000 16000 256000|17 17|55"
  "rect32_100k|-50000 -50000 50000 50000|32 32|150"
)
[[ $ADD_SCALE -eq 1 ]] && WL+=("scale_1.5G|-1875000 -200000 1875000 200000|17 17|55")
[[ $ADD_FULL  -eq 1 ]] && WL+=("fullmap_14G|-1875000 -1875000 1875000 1875000|17 17|55")

if [[ $DOWAIT -eq 1 ]] && [[ -x "$ROOT/tools/gpuinfo.sh" ]]; then
    "$ROOT/tools/gpuinfo.sh" -w 600 >/dev/null 2>&1 || true
fi

# ---- run one workload once; echoes "gpu_ms wall_ms cells_M valid" ----
run_once() {
    local bin="$1" x0="$2" z0="$3" x1="$4" z1="$5" sx="$6" sz="$7" thr="$8"
    local csv="$WORK/o.csv" log="$WORK/o.log"
    rm -f "$csv"
    timeout 1800 "$bin" "$SEED" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr" "$csv" --sort="$SORTM" \
        > "$log" 2>&1 || { echo "FAIL FAIL 0 0"; return 1; }
    local line gpu wall valid
    line="$(grep -E '^Total valid: ' "$log" | tail -1)"
    gpu="$(sed -n 's/.*GPU compute time: \([0-9.]*\) ms.*/\1/p' <<<"$line")"
    wall="$(sed -n 's/.*wall time: \([0-9.]*\) ms.*/\1/p' <<<"$line")"
    valid="$(sed -n 's/^Total valid: \([0-9]*\).*/\1/p' <<<"$line")"
    echo "$gpu $wall $valid"
}

cells_M() { awk -v a="$1" -v b="$2" -v c="$3" -v d="$4" 'BEGIN{printf "%.1f",((c-a+1)*(d-b+1))/1e6}'; }
bs()      { awk -v m="$1" -v t="$2" 'BEGIN{ if (t+0>0) printf "%.1f", m/t; else print "-" }'; }

HDR="%-16s %11s %5s %10s %10s %9s %9s %8s %5s"
ROW="%-16s %11s %5s %10s %10s %9s %9s %8s %5s"
printf 'binary : %s\n' "$BIN"
[[ -n "$REF" ]] && printf 'ref    : %s\n' "$REF"
printf 'reps   : %d   seed=%s   sort=%s\n\n' "$REPS" "$SEED" "$SORTM"
printf "$HDR\n" workload cells_M reps gpu_min wall_min B/s_gpu B/s_wall ratio det
printf "$HDR\n" ---------------- ----------- ----- ---------- ---------- --------- --------- -------- -----

for spec in "${WL[@]}"; do
    IFS='|' read -r name area rect thr <<<"$spec"
    set -- $area; x0=$1; z0=$2; x1=$3; z1=$4
    set -- $rect; sx=$1; sz=$2
    M="$(cells_M "$x0" "$z0" "$x1" "$z1")"

    gpua=(); walla=(); gpub=(); wallb=(); deta=""; detb=""
    for (( i=1; i<=REPS; i++ )); do
        read -r g w v <<<"$(run_once "$BIN" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr")"
        gpub+=("$g"); wallb+=("$w"); detb="$v"
        if [[ -n "$REF" ]]; then
            read -r g w v <<<"$(run_once "$REF" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr")"
            gpua+=("$g"); walla+=("$w"); deta="$v"
        fi
    done
    gmin="$(printf '%s\n' "${gpub[@]}"   | sort -n | head -1)"
    wmin="$(printf '%s\n' "${walla[@]:-0}" | sort -n | head -1)"
    wminb="$(printf '%s\n' "${wallb[@]}" | sort -n | head -1)"
    ratio="-"; det="ok"
    if [[ -n "$REF" ]]; then
        gmina="$(printf '%s\n' "${gpua[@]}" | sort -n | head -1)"
        ratio="$(awk -v a="$gmina" -v b="$gmin" 'BEGIN{ if(b+0>0) printf "%.3f", a/b; else print "-" }')"
    fi
    [[ "$deta" != "$detb" && -n "$REF" ]] && det="DIFF"
    printf "$ROW\n" "$name" "$M" "$REPS" "$gmin" "$wminb" "$(bs "$M" "$gmin")" "$(bs "$M" "$wminb")" "$ratio" "$det"
done

echo
echo "note: ratio = ref_gpu_min / bin_gpu_min, >1 means BIN is faster (interleaved A/B)."
echo "      B/s is the declared-cells throughput; compare it only within one workload."
echo "      det=DIFF means the two binaries produced different hit counts -- timing is void."
