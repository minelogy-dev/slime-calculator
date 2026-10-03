#!/bin/bash
#
# collect.sh -- collect the performance and correctness data of slime_main on a headless box.
#
#   usage: ./collect.sh [--ref OLD_BINARY] [--reps N] [--out DIR]
#
# Writes the results to <project root>/.work/collect/<timestamp>/, one file per data point, and
# then tars the whole directory into a sibling <timestamp>.tar.gz (the source directory is kept),
# so a single command yields one archive you can carry away. Inside the directory SUMMARY.md
# records what each file is, the key numbers, and the verdict of every stage.
# Without --ref the bit-exactness matrix is skipped (it needs an old executable to compare
# against). Output is pure ASCII.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

REF=""; REPS=1; OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)  REF="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --out)  OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

STAMP="$(date +%Y%m%d-%H%M%S)"
[[ -n "$OUT" ]] || OUT="$ROOT/.work/collect/$STAMP"
mkdir -p "$OUT"

say() { printf '\n=== %s ===\n' "$*"; }

# 小工具可能放在 ./slime-tools/（部署时的平铺布局）或 ./tools/（仓库原样）。
# 两处都找，找不到就让调用点自己决定是降级还是跳过。
find_tool() {
    local n="$1"
    for d in ./slime-tools ./tools; do
        [[ -x "$d/$n" ]] && { echo "$d/$n"; return 0; }
    done
    return 1
}

say "environment and identity"
{
    echo "host        $(hostname)"
    echo "date        $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "pwd         $ROOT"
    echo "nvcc        $(nvcc --version 2>/dev/null | tail -1 | sed 's/^ *//')"
    echo "gcc         $(gcc --version 2>/dev/null | head -1)"
    echo "git commit  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "src sha256  $(sha256sum src/slime_main.cu | cut -d' ' -f1)"
    echo "bin sha256  $(sha256sum build/slime_main 2>/dev/null | cut -d' ' -f1)"
    echo "bin sass    $(cuobjdump -sass build/slime_main 2>/dev/null | sha256sum | cut -d' ' -f1)"
    echo "git status  $(git status --porcelain 2>/dev/null | wc -l) modified path(s)"
} 2>&1 | tee "$OUT/00-identity.txt"

say "GPU profile"
if G="$(find_tool gpuinfo.sh)"; then
    "$G" 2>&1 | tee "$OUT/01-gpuinfo.txt"
    "$G" -1 2>&1 | tee "$OUT/01-gpuinfo-oneline.txt"
else
    echo "gpuinfo.sh not found, skipping (it may live in ./slime-tools/ or ./tools/)" | tee "$OUT/01-gpuinfo.txt"
    : > "$OUT/01-gpuinfo-oneline.txt"
fi

say "full-map timing and throughput"
# --ref 也透传给 quickbench：给了它才会交错 A/B 跑两个二进制，det 列才有意义
# （quickbench 在单二进制模式下 det 恒为 ok，不做任何一致性比对）。
if B="$(find_tool quickbench.sh)"; then
    if [[ -n "$REF" ]]; then
        "$B" --full -n "$REPS" --ref "$REF" 2>&1 | tee "$OUT/02-quickbench-full.txt"
    else
        echo "note: no --ref, so quickbench runs a single binary and its det column is not a check" \
            | tee "$OUT/02-quickbench-full.txt"
        "$B" --full -n "$REPS" 2>&1 | tee -a "$OUT/02-quickbench-full.txt"
    fi
else
    echo "quickbench.sh not found, falling back to bench.sh --full" >&2
    ./bench.sh --full 2>&1 | tee "$OUT/02-bench-full.txt"
fi

say "ncu: instruction counts and pipe mix (frequency-independent)"
if command -v ncu >/dev/null 2>&1; then
    M="smsp__inst_executed.sum,gpu__time_duration.sum,sm__cycles_elapsed.avg,"
    M+="launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,"
    M+="sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,"
    M+="sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_uniform.sum,"
    M+="sm__inst_executed_pipe_xu.sum,smsp__issue_active.avg.pct_of_peak_sustained_active"
    # 区域：X[-131200,131199] Z[-32776,32775] 17x17 thr=60 => 候选 17,195,597,824
    #
    # 跑两次：一次导出**完整原始报告**（.ncu-repz，可事后用 ncu --import 或 ncu-ui 查任意指标、
    # 看 stall 分解与内存明细），一次输出上面这些指标的 CSV 文本（便于直接 diff/汇总）。
    ncu --target-processes all --launch-skip 0 --launch-count 1 --force-overwrite \
        --export "$OUT/03-ncu-report" \
        ./build/slime_main 114514 -131200 -32776 131199 32775 17 17 60 /dev/null --sort=off \
        > "$OUT/03-ncu-report.log" 2>&1
    ncu --target-processes all --launch-skip 0 --launch-count 1 --csv --metrics "$M" \
        ./build/slime_main 114514 -131200 -32776 131199 32775 17 17 60 /dev/null --sort=off \
        > "$OUT/03-ncu.csv" 2>&1
    echo "raw report : 03-ncu-report.ncu-repz  (open with ncu --import or ncu-ui)"
    echo "metrics csv: 03-ncu.csv"
    # 换算 inst/候选（候选数固定为 17,195,597,824）
    inst=$(grep 'smsp__inst_executed.sum' "$OUT/03-ncu.csv" | sed 's/.*,"inst","//;s/"//' | tr -d ',')
    if [[ -n "$inst" ]]; then
        python3 -c "print('inst/candidate = %.3f' % ($inst*32/17195597824))" | tee "$OUT/03-inst-per-candidate.txt"
    fi
else
    echo "ncu not found, skipping" >&2
fi

say "bit-exactness matrix"
if [[ -n "$REF" ]]; then
    ./tools/verify.sh --ref "$REF" 2>&1 | tee "$OUT/04-verify-full.txt"
else
    echo "no --ref given, skipping (the matrix needs an old executable to compare against)" | tee "$OUT/04-verify-full.txt"
fi

say "cross-check against the original Minecraft Java (real JVM)"
if [[ -x ./tools/mccrosscheck.sh ]]; then
    ./tools/mccrosscheck.sh 2>&1 | tee "$OUT/05-mccrosscheck.txt"
fi

say "compare the complete output against the independent oracle"
if Q="$(find_tool quickcheck.sh)"; then
    "$Q" 2>&1 | tee "$OUT/06-quickcheck.txt"
else
    echo "quickcheck.sh not found, skipping" | tee "$OUT/06-quickcheck.txt"
fi

say "summary"
# 汇总分两段：一段把每个文件的结论抽出来，一段把可直接引用的关键数值算好 ——
# 后者省得再回头翻原始文件，也让归档自带一份"这份数据说了什么"。
# ncu 的 CSV 用 stdcsv 规则（字段可能带引号、数值带千分位逗号），交给 Python 解析。
# 先用 grep 按「指标名 + 内核名」筛出那一行再交给 Python —— 直接 grep 指标名会命中别的行。
NCU_MET() {  # $1 = 指标名，$2 = 内核名（可选）
    grep -a "\"$1\"," "$OUT/03-ncu.csv" 2>/dev/null \
        | { [[ -n "${2:-}" ]] && grep -a "$2" || cat; } \
        | head -1 \
        | python3 -c 'import csv,sys
for r in csv.reader(sys.stdin):
    if len(r) >= 4:
        print(r[-1].replace(",", "").strip(chr(34)))
        break' 2>/dev/null
}
BENCH_ROW() {  # $1 = 负载名 -> cells_M gpu_min wall_min B/s_wall
    # 表头是多空格对齐的，而且 "B/s_gpu"/"B/s_wall" 名字里带空格，按列位置或字段序号都会
    # 错位。数据行里**唯一含 "/" 的字段就是那两个分数字段**，所以先按空白切、跳过含 "/" 的，
    # 剩下的顺序固定：cells_M reps gpu_min wall_min B/s_gpu B/s_wall ratio det。
    awk -v w="$1" '
        $1 == w {
            n = 0; delete f
            for (i = 2; i <= NF; i++) if ($i !~ /\//) f[++n] = $i
            printf "%s %s %s %s\n", f[1], f[3], f[4], f[6]
            exit
        }
    ' "$OUT/02-quickbench-full.txt" 2>/dev/null
}
{
    echo "# collect.sh results"
    echo
    echo "timestamp   $STAMP"
    echo "host        $(hostname)"
    echo "src sha256  $(sha256sum src/slime_main.cu | cut -d' ' -f1)"
    echo "bin md5     $(md5sum build/slime_main 2>/dev/null | cut -d' ' -f1)"
    echo "reference   ${REF:-(none given; the bit-exactness matrix was skipped)}"
    echo "repetitions $REPS"

    echo
    echo "## key numbers"
    echo
    echo "| metric | value | source |"
    echo "|---|---|---|"
    ipc="$(cat "$OUT/03-inst-per-candidate.txt" 2>/dev/null | sed 's/.*= //')"
    [[ -n "$ipc" ]] && echo "| inst/candidate | $ipc | 03-inst-per-candidate.txt |"
    KERN=scanRectFusedMerge4Kernel
    fd="$(NCU_MET gpu__time_duration.sum "$KERN")"; [[ -n "$fd" ]] && \
        echo "| kernel duration | $(python3 -c "print('%.2f ms' % ($fd/1e6))" 2>/dev/null) | 03-ncu.csv |"
    reg="$(NCU_MET launch__registers_per_thread "$KERN")"; [[ -n "$reg" ]] && \
        echo "| registers/thread | $reg | 03-ncu.csv |"
    occ="$(NCU_MET sm__warps_active.avg.pct_of_peak_sustained_active "$KERN")"; [[ -n "$occ" ]] && \
        echo "| occupancy | $occ % | 03-ncu.csv |"
    cyc="$(NCU_MET sm__cycles_elapsed.avg "$KERN")"; [[ -n "$cyc" ]] && \
        echo "| SM cycles | $(python3 -c "print(f'{$cyc:,.0f}')" 2>/dev/null) | 03-ncu.csv |"
    fm="$(BENCH_ROW fullmap_14G)"
    if [[ -n "$fm" ]]; then
        set -- $fm   # cells_M gpu_min wall_min B/s_wall
        echo "| full map (14.06e12 candidates) | wall_min $3 ms, gpu_min $2 ms, B/s_wall $4 | 02-quickbench-full.txt |"
        echo "| full-map candidates | $1 M cells | 02-quickbench-full.txt |"
    fi
    tot=0; for m in alu fma lsu uniform xu; do
        v="$(NCU_MET sm__inst_executed_pipe_$m.sum "$KERN")"; [[ -n "$v" ]] && tot=$((tot + v))
    done
    if [[ $tot -gt 0 ]]; then
        printf "| pipe mix |"
        for m in alu fma lsu uniform xu; do
            v="$(NCU_MET sm__inst_executed_pipe_$m.sum "$KERN")"; [[ -z "$v" ]] && continue
            printf " %s %.1f%%;" "$m" "$(python3 -c "print(100*$v/$tot)" 2>/dev/null)"
        done
        echo " | 03-ncu.csv |"
    fi

    echo
    echo "## workloads"
    echo
    echo '```'
    sed -n '/^workload/,/^$/p' "$OUT/02-quickbench-full.txt" 2>/dev/null
    echo '```'

    echo
    echo "## files"
    echo
    printf '%-30s %s\n' "00-identity.txt" "environment and identity (toolchain versions, src/bin hashes, dirty-file count)"
    printf '%-30s %s\n' "01-gpuinfo.txt" "GPU profile (model, SM count, driver, clocks, temperature)"
    printf '%-30s %s\n' "01-gpuinfo-oneline.txt" "the same, single line"
    printf '%-30s %s\n' "02-quickbench-full.txt" "per-workload speed table (full-map wall clock and row count)"
    printf '%-30s %s\n' "03-ncu.csv" "ncu metrics (inst/candidate numerator, cycles, registers, occupancy, pipe mix)"
    printf '%-30s %s\n' "03-inst-per-candidate.txt" "inst/candidate (frequency-independent)"
    printf '%-30s %s\n' "03-ncu-report.ncu-repz" "raw ncu report; open with ncu --import or ncu-ui"
    printf '%-30s %s\n' "04-verify-full.txt" "bit-exactness matrix against --ref (byte-for-byte)"
    printf '%-30s %s\n' "05-mccrosscheck.txt" "cross-check vs the wiki Java on a real JVM"
    printf '%-30s %s\n' "06-quickcheck.txt" "complete-output comparison against the independent oracle"

    echo
    echo "## stage verdicts"
    echo
    for spec in "04-verify-full.txt:结论" "05-mccrosscheck.txt:RESULT" "06-quickcheck.txt:RESULT"; do
        f="${spec%%:*}"; key="${spec##*:}"
        line="$(grep -aE "$key" "$OUT/$f" 2>/dev/null | tail -1 | sed 's/^ *//;s/  */ /g')"
        printf '%-30s %s\n' "$f" "${line:-(no verdict line)}"
    done
} > "$OUT/SUMMARY.md"
cat "$OUT/SUMMARY.md"

say "packaging"
ARCHIVE="$OUT.tar.gz"
if tar -czf "$ARCHIVE" -C "$(dirname "$OUT")" "$(basename "$OUT")"; then
    echo "archive: $ARCHIVE"
    echo "size   : $(du -h "$ARCHIVE" | cut -f1)  (source directory kept)"
else
    echo "packaging failed; the data is still in $OUT" >&2
fi

say "done"
echo "results dir: $OUT"
echo "archive    : $ARCHIVE"
ls -la "$OUT"
