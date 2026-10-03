#!/bin/bash
#
# tools/verify.sh -- bit-exactness matrix for slime_main (A/B, byte for byte).
#
# Output is pure ASCII (it must survive a headless box with no UTF-8 locale).
# Goal
#   Compare the current build/slime_main against an older executable given explicitly via
#   --ref over the *same configuration matrix*, byte for byte. Any difference, crash or
#   counter mismatch makes this exit non-zero.
#
# Usage
#   tools/verify.sh [--ref BIN] [--new BIN] [--fast] [--no-extra] [--strict-oracle] [--plan-only]
#   --ref is REQUIRED: the frozen reference binary is not published, so pass its path yourself.
#   --new defaults to build/slime_main.
#     --fast            run a reduced subset (currently 27 entries, ~40 s total wall) and
#                       skip the compile-time variants
#     --no-extra        skip the zsub1 / k4 / scan_only compile-time variants (full mode
#                       compiles and includes them by default)
#     --strict-oracle   treat invariant-check warnings as FAIL (default: WARN only, does not
#                       affect the A/B verdict)
#     --plan-only       print the configuration matrix and exit (no GPU touched)
#   Every entry has a hard ${SLIME_VERIFY_TIMEOUT:-900}s timeout; a timeout counts as FAIL --
#   a deadlocked build must never keep holding the global GPU lock.
#
# Scale and cost (measured, RTX 4060 Ti)
#   Full = 42 configurations x {ref, new, zsub1} + 6 scan_only entries; ~46 s of program time,
#   ~65 s inside the lock (including sort + md5). Total wall clock depends on whether another
#   job holds the GPU lock (measured 88 s .. 185 s); this tool prints lock+cooldown and
#   in-lock execution separately.
#
# Verdict rules (every one must hold)
#   1) both ref and new exit with code 0 (except the reject_* entries, which require both to
#      reject with the SAME non-zero code);
#   2) both print a "Total valid:" line and the values are identical;
#   3) the CSVs are identical: LC_ALL=C sort <out.csv> | md5sum must match byte for byte.
#      Output order on the producer side is decided by atomicAdd, so the sort is mandatory;
#      another locale produces false DIFFs (known trap).
#   4) no CUDA error / "internal: chunk retry limit" markers in the logs.
#
# Extra checks (stronger than plain A/B -- they catch "both sides wrong the same way")
#   * invariant self-check (oracle; WARN by default, --strict-oracle promotes it to FAIL):
#     with threshold=0 every window must hit, so the hit set must be EXACTLY the whole window
#     grid inside the scanned range:
#         Total valid == rangeX*validZ == CSV data rows == distinct coordinates
#         distinct X count == rangeX, distinct Z count == validZ
#         X/Z extremes == the two ends of the scanned range
#         every coordinate must be a multiple of 16 (chunk coordinate x16)
#     The scanned range is taken from the program's own "Global range:" line (which already
#     includes the 256-column X alignment and the Z step alignment). The grid/extreme checks
#     catch bugs where the count is right but the coordinates are shifted overall -- something
#     plain A/B cannot see.
#     With a non-zero threshold this degrades to: coordinates 16-aligned and inside the range.
#     A threshold above the window maximum (sizeX*sizeZ) must yield header only, Total valid == 0.
#   * group self-check (group): the --sort=on and --sort=off twins must produce the same md5
#     and the same Total valid.
#   * reference integrity: if <ref>.md5 exists next to --ref, the reference binary is verified
#     against it; otherwise this is skipped with a note.
#
# Compile-time knobs (included by default in full mode; compiling is CPU-only and safe)
#   .work/verify/slime_zsub1     -DSLIME_Z_SUB_ROWS=100000000 -DSLIME_Z_SPLIT_MIN_TILES=100000000
#                                forces zSub==1 (the historical bug path); the result must be
#                                byte-identical to the reference
#   .work/verify/slime_scan_only -DSLIME_SCAN_ONLY: no result registration at all; only checks
#                                that the run completes without a CUDA error
#
# Disk convention
#   /tmp is a 7.6 GB tmpfs that this project once filled with CSVs, producing false ENOSPC
#   failures. All intermediate artefacts therefore live under .work/verify/ (real disk), the
#   sort temporary directory is pointed at .work/verify/tmp via TMPDIR, and every CSV is deleted
#   as soon as its md5 / distinct-coordinate counts are computed.
#
# Exit codes: 0 = all PASS; 1 = some FAIL; 2 = usage/environment error (missing binary, ...).

set -uo pipefail

# ---------------------------------------------------------------- 参数解析
REF=""
NEW="build/slime_main"
FAST=0
NOEXTRA=0
STRICT=0
PLAN_ONLY=0

usage() {   # 打印文件头部的注释块（到第一个非注释行为止）
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)  REF="$2"; shift 2 ;;
        --new)  NEW="$2"; shift 2 ;;
        --fast) FAST=1; shift ;;
        --no-extra) NOEXTRA=1; shift ;;
        --strict-oracle) STRICT=1; shift ;;
        --plan-only) PLAN_ONLY=1; shift ;;   # 只打印矩阵（不碰 GPU），用于检查配置
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

VDIR="$ROOT/.work/verify"
RUN="$VDIR/result"
LOGDIR="$RUN/logs"
CSVDIR="$RUN/csv"
TMPDIR_V="$VDIR/tmp"
PLAN="$VDIR/plan.tsv"
export TMPDIR="$TMPDIR_V"     # sort 的溢写临时文件绝不能落到 /tmp

mkdir -p "$VDIR" "$RUN" "$LOGDIR" "$CSVDIR" "$TMPDIR_V"
rm -f "$CSVDIR"/* "$LOGDIR"/* 2>/dev/null

# REF 无默认值：冻结参考二进制不随仓库发布（是某次构建的产物，且 CUDA 工具链在 ELF
# 层面不可复现，提交进 git 无意义）⇒ 必须由 --ref 显式给出一个旧可执行文件的路径。
# （这一行是占位，真正的空值检查在下面矩阵打印之后。）

# ---------------------------------------------------------------- 配置矩阵
# 每行：id|group|kind|env|args|flags（用 | 分隔：空字段在 read 下不会被折叠）
#   kind  = cmp（逐字节比对）| reject（必须被拒绝：非零退出码）| 其他一律按 cmp 处理
#   env   = 传给程序的环境变量（空格分隔，可为空）
#   args  = 程序参数（不含输出 CSV，worker 会追加）
#   flags = fast（精简子集）/ scan（scan_only 变体也跑）/ spill（采样 .runN 落盘文件）
plan_emit() {
    # ---- (a) 形状矩阵：X 0..255 / Z 0..255，thr=0（每个窗口都命中，可被 oracle 精确校验）----
    echo "shape_1x1||cmp||114514 0 0 255 255 1 1 0|fast"
    echo "shape_1x32||cmp||114514 0 0 255 255 1 32 0|"
    echo "shape_32x1||cmp||114514 0 0 255 255 32 1 0|"
    echo "shape_17x1||cmp||114514 0 0 255 255 17 1 0|"
    echo "shape_1x17||cmp||114514 0 0 255 255 1 17 0|"
    echo "shape_32x32||cmp||114514 0 0 255 255 32 32 0|fast,scan"
    # >32 -> 回退路径
    echo "shape_33x17||cmp||114514 0 0 255 255 33 17 0|fast"
    # 回退路径
    echo "shape_40x17||cmp||114514 0 0 255 255 40 17 0|scan"
    echo "shape_17x32||cmp||114514 0 0 255 255 17 32 0|"
    # 计数环选路的**边界**定向形状（T1「4 邻列共享 20 列松上界」就动这条路径）：
    # kernel 选路是 `sizeZ <= COUNT_RING_MAX_SIZE_Z(20)` 走 scanRectFusedKernelCount，
    # 否则回退 scanRectFusedKernelBallot。修改前矩阵里没有任何一项落在 sizeZ=20 上，
    # sizeZ=21 这个「刚好掉出计数环」的点也没有被区分执行过。
    # ★ X 跨度取 8192（rangeX=512 列）即可触发计数环选路（判据只看 sizeZ），
    #   而**不要**取 65536：验证协议的排序+哈希是锁内开销，巨形状会把 --fast 从 ~40 s
    #   拖到 ~500 s（实测），白白占着全局 GPU 锁。
    # 计数环路径（sizeZ<20）
    echo "shape_17x19_ring||cmp||114514 0 0 8191 255 17 19 0|fast"
    # 计数环路径（sizeZ==20，边界）
    echo "shape_17x20_ring||cmp||114514 0 0 8191 255 17 20 0|fast"
    # sizeZ==21，刚好掉出计数环 -> ballot 核
    echo "shape_17x21_ballot||cmp||114514 0 0 8191 255 17 21 0|fast"
    # **K=4 融合核的覆盖**不放在这里：触发 K=4 要求 `rangeX >= FUSED_K_MIN_COLS(200000)`，
    # 而这么宽的窗口即使只留 32 行也会产出数亿命中行（实测 thr=0 时排序+哈希把磁盘写满、
    # 留下 50 个共 4.2 GB 的 spill 段）。正确做法是用**强制 K=4 的编译期变体**
    # （`-DSLIME_K4_MIN_BLOCKS=0`，见下方 k4 目标）跑全部形状——它便宜、覆盖全、且能
    # 与参考逐字节比对。默认矩阵只保留能便宜执行的形状。
    # 几何尾列定向形状（merge4 的 bundle 排布约定）：
    # 「4 邻列 bundle」要求 outW = TILE_W - sizeX + 1 ≡ 0 (mod 4) ⇔ sizeX ≡ 1 (mod 4)。
    # sizeX ∈ [1,32] 只有 {1,5,9,13,17,21,25,29} 满足；其余 24 个形态的尾列必须另有覆盖路径。
    # 这里的 sizeX=16 就是③给出的最小反例形态（outW=1009，尾列 1008 裸露 ⇒ 丢 32/37521 命中）；
    # sizeX=17 是对照（17≡1 mod 4，恰好无缺口，也是参照实现可以不管它的原因）。
    # sizeX%4==0 尾列裸露
    echo "shape_16x17_tail||cmp||114514 -200000 -200000 -191800 -191800 16 17 45|fast"
    # 对照：17≡1 mod 4 无缺口
    echo "shape_17x17_algn||cmp||114514 -200000 -200000 -191800 -191800 17 17 45|fast"
    # 单次抽取上限 29（≡1 mod 4）
    echo "shape_29x17_algn||cmp||114514 -200000 -200000 -191800 -191800 29 17 45|fast"
    # 30>29 -> 必须走回退核
    echo "shape_30x17_fb||cmp||114514 -200000 -200000 -191800 -191800 30 17 45|fast"
    # 回退路径
    echo "shape_100x17||cmp||114514 0 0 255 255 100 17 0|"
    # 形状上限
    echo "shape_255x32||cmp||114514 0 0 255 255 255 32 0|fast"

    # ---- (b) threshold 扫描：X -300..300 / Z -300..300 ----
    echo "thr_0||cmp||114514 -300 -300 300 300 17 17 0|fast"
    echo "thr_1|g_sort_fused|cmp||114514 -300 -300 300 300 17 17 1|fast"
    echo "thr_55||cmp||114514 -300 -300 300 300 17 17 55|fast"
    # > 窗口上限 -> 只有表头
    echo "thr_hi||cmp||114514 -300 -300 300 300 17 17 100000|fast"
    echo "thr_hi_fb||cmp||114514 -300 -300 300 300 40 17 100000|"

    # ---- (c) 跨多个 X tile / 多个 Z 块的大范围：X ±6000 / Z ±6000 ----
    echo "big_fused_45|g_sort_big|cmp||114514 -6000 -6000 6000 6000 17 17 45|fast"
    echo "big_fb_90||cmp||114514 -6000 -6000 6000 6000 40 17 90|fast,scan"

    # ---- (d) 正负坐标 / 贴近 ±2,000,000 硬界 ----
    echo "coord_neg||cmp||114514 -1024 -1024 -769 -769 17 17 45|fast"
    echo "coord_pos||cmp||114514 769 769 1024 1024 17 17 45|"
    echo "coord_mixed||cmp||114514 -128 -128 127 127 17 17 55|"
    echo "coord_nearmax||cmp||114514 1999900 1999900 1999999 1999999 17 17 55|fast,scan"
    echo "coord_nearmin||cmp||114514 -2000000 -2000000 -1999901 -1999901 17 17 55|"
    echo "coord_nearmax_fb||cmp||114514 1999900 1999900 1999999 1999999 40 17 55|"
    # 非法输入必须被拒绝（负向对照，两边退出码相同且非零才算 PASS）
    echo "reject_xbound||reject||114514 2000001 0 2000010 10 17 17 55|fast"
    echo "reject_sizeX||reject||114514 0 0 255 255 256 1 55|"
    echo "reject_sizeZ||reject||114514 0 0 255 255 1 33 55|"
    echo "reject_size0||reject||114514 0 0 255 255 0 1 55|"
    echo "reject_sortarg||reject||114514 0 0 255 255 17 17 55 --sort=maybe|"

    # ---- (d2) 定向拒绝形状：Java 拒绝采样分支概率 ~4e-9，随机区域永远碰不到 ----
    # seed 1100064637205 的 chunk(5,7) 是一个「必然可区分」的格子：
    # 跳过重抽会得到 slime=1，Java 正确答案是 0（可用 tools/oracle.c --scan-reject 重新搜索）。
    # 没有这两项时，一个把重抽删掉的实现能通过整个矩阵 —— 已实测。
    echo "rej_directed_1x1||cmp||1100064637205 5 7 5 7 1 1 1|fast,scan"
    echo "rej_directed_17x17||cmp||1100064637205 -20 -20 30 30 17 17 5|"
    echo "rej_directed_2||cmp||2 3728 -10177 3728 -10177 1 1 1|"
    echo "rej_directed_2b||cmp||2 3700 -10200 3760 -10150 17 17 3|"

    # ---- (e) --sort=on / --sort=off 孪生对照（同 group 必须完全一致）----
    echo "srt_fused_off|g_sort_fused|cmp||114514 -300 -300 300 300 17 17 1 --sort=off|fast"
    echo "srt_fb_on|g_sort_fb|cmp||114514 0 0 4095 255 40 17 0|"
    echo "srt_fb_off|g_sort_fb|cmp||114514 0 0 4095 255 40 17 0 --sort=off|scan"
    echo "srt_big_off|g_sort_big|cmp||114514 -6000 -6000 6000 6000 17 17 45 --sort=off|fast"

    # ---- (f) SLIME_CAP_SLOTS：压小结果缓冲，强制「缓冲满 -> 丢弃重试、缩小 chunk」----
    # 注意它会被 clamp 到至少 rangeX，所以必须让扫描宽度足够小、命中足够密才会真的咬住
    echo "cap_4096||cmp|SLIME_CAP_SLOTS=4096|114514 0 0 4095 255 17 17 0|fast,scan"
    echo "cap_65536||cmp|SLIME_CAP_SLOTS=65536|114514 0 0 1023 1023 17 17 0|"
    # 回退路径分批重试
    echo "cap_4096_fb||cmp|SLIME_CAP_SLOTS=4096|114514 0 0 4095 255 40 17 0|"
    # 宽区 -> 多次折半
    echo "cap_4096_fbwide||cmp|SLIME_CAP_SLOTS=4096|114514 0 0 8191 255 40 17 0|"
    # 大区 -> 深重试
    echo "cap_16384_big||cmp|SLIME_CAP_SLOTS=16384|114514 -6000 -6000 6000 6000 17 17 45|"
    # C2 定向形状：缓冲区必须**在 chunk 中途打满**，才能区分「停手生效的时机」。
    # 上面那些 cap_* 项都是在第 0 行就溢出（缓冲区秒满），从来没有区分度地执行过
    # 「worker 置位 stop -> 其余 warp 算完本行、并在下一个 64 行边界才 break」这条路径。
    # 构造：threshold=0 时每行每个候选列都命中 ⇒ 每行命中 = rangeX = 16320 列。
    # 取 cap=65536 ⇒ 第 4 行末累计 65280、第 5 行中途越过 cap ⇒ 溢出点远在 64 行边界之前，
    # 于是"多算到边界"这段（≤63 行）真的被执行；随后主机丢弃本轮并折半重试。
    # 中途打满
    echo "cap_mid_fill||cmp|SLIME_CAP_SLOTS=65536|114514 0 0 16319 1023 17 17 0|fast"

    # ---- (g) SLIME_SORT_BLOCKS=2：限制内存块数，强制 spill 落盘 + 多路归并 ----
    # 2080 万行 -> 必然 spill
    echo "spill_2blk||cmp|SLIME_SORT_BLOCKS=2|114514 0 0 8191 2559 17 17 0|spill"
    echo "spill_2blk_small||cmp|SLIME_SORT_BLOCKS=2|114514 -6000 -6000 6000 6000 17 17 45|fast"

    # ---- (h) 两个种子（含 int64 最小值）----
    echo "seed_neg_min||cmp||-9223372036854775808 0 0 255 255 17 17 0|fast"
    echo "seed_neg_12345||cmp||-12345 0 0 255 255 17 17 0|"
}

# 计划行是否落在「融合路径」内（sizeX <= MAX_FUSED_SIZE_X、sizeZ <= MAX_SIZE_Z）。
# 强制 K=4 变体的选路守卫组合比默认更严：默认路径下 `rangeX >= FUSED_K_MIN_COLS`
# 的条件不会满足，于是每个 block 都要各自证明 `tiles4probe*zSubMax >= 408`，
# 窄形状（如 shape_1x1，1 tile × 1 zSub）会自动落回 K=1 的 ballot 核 ⇒ 对 k4 目标
# 这些项没有意义（而且该组合还会被程序按「不合法输入」拒绝），必须过滤掉。
shape_is_fused() {
    local args="$3" sizeX sizeZ
    sizeX=$(echo "$args" | awk '{print $(NF-2)}')
    sizeZ=$(echo "$args" | awk '{print $(NF-1)}')
    [[ "$sizeX" =~ ^[0-9]+$ && "$sizeZ" =~ ^[0-9]+$ && $sizeX -ge 1 && $sizeX -le 32 && $sizeZ -le 32 ]]
}

# 生成某目标的私有计划：k4 走过滤后的子集，其余目标用全量计划。
emit_plan_for() {
    if [[ "$1" == "k4" ]]; then
        while IFS='|' read -r id group kind env args flags; do
            [[ -z "${id:-}" ]] && continue
            shape_is_fused "$id" "$group" "$args" || continue
            printf '%s|%s|%s|%s|%s|%s\n' "$id" "$group" "$kind" "$env" "$args" "$flags"
        done < "$PLAN"
    else
        cat "$PLAN"
    fi
}

if [[ $FAST -eq 1 ]]; then
    plan_emit | awk -F'|' 'NF && $6 ~ /fast/' > "$PLAN"
else
    plan_emit > "$PLAN"
fi
NPLAN=$(grep -c . "$PLAN")

# --plan-only：只打印矩阵后退出 —— 不碰 GPU，也不需要 ref/new 任何二进制。
if [[ $PLAN_ONLY -eq 1 ]]; then
    echo "== matrix ($NPLAN entries, fast=$FAST)"
    awk -F'|' '{printf "%-18s group=%-12s kind=%-6s env=%-22s args=%s   [%s]\n",$1,$2,$3,$4,$5,$6}' "$PLAN"
    exit 0
fi

# 走到这里才需要二进制：先给出可操作的提示，再检查可执行性。
[[ -n "$REF" ]] || {
    cat >&2 <<'EOM'
error: --ref is required -- pass the path of the frozen reference binary.
  This script compares the current build/slime_main against an older executable over the
  same matrix, byte for byte. That old binary is not published, so supply one yourself:
    tools/verify.sh --ref /path/to/old_slime_main            # full matrix
    tools/verify.sh --ref /path/to/old_slime_main --fast     # reduced subset
    tools/verify.sh --plan-only                              # print the matrix only (no GPU)
EOM
    exit 2
}
[[ -x "$REF" ]] || { echo "error: reference binary missing or not executable: $REF" >&2; exit 2; }
[[ -x "$NEW" ]] || { echo "error: binary under test missing or not executable: $NEW" >&2; exit 2; }



# ---------------------------------------------------------------- 编译期变体
ZSUB1="$VDIR/slime_zsub1"
SCANONLY="$VDIR/slime_scan_only"
K4ONLY="$VDIR/slime_k4"
SRC="$ROOT/src/slime_main.cu"

build_one() {   # build_one <输出> <额外宏...>
    local out="$1"; shift
    if [[ -x "$out" && "$out" -nt "$SRC" ]]; then
        echo "  [cache] $out is up to date (source unchanged)"
        return 0
    fi
    echo "  [nvcc ] $out $*  (CPU heavy, please wait)"
    if ! nvcc -o "$out" "$SRC" -O3 -use_fast_math -arch=sm_89 "$@" > "$VDIR/build_extras.log" 2>&1; then
        echo "  [warn ] $out failed to compile; skipping this variant:" >&2
        tail -5 "$VDIR/build_extras.log" >&2
        rm -f "$out"
        return 1
    fi
    return 0
}

HAVE_ZSUB1=0
HAVE_SCAN=0
HAVE_K4=0
BUILD_FAIL=0
if [[ $FAST -eq 1 || $NOEXTRA -eq 1 ]]; then
    echo "== compile-time variants: skipped (--fast/--no-extra)"
else
    echo "== compile-time variants (pure CPU, no GPU lock taken)"
    # 历史 bug 路径：强制关闭 Z 切分（zSub 恒为 1）
    build_one "$ZSUB1" -DSLIME_Z_SUB_ROWS=100000000 -DSLIME_Z_SPLIT_MIN_TILES=100000000 && HAVE_ZSUB1=1 || BUILD_FAIL=1
    # 无结果登记变体：只验证能跑完、无 CUDA 错误
    build_one "$SCANONLY" -DSLIME_SCAN_ONLY && HAVE_SCAN=1 || BUILD_FAIL=1
    # 强制 K=4 变体：默认选路要 `rangeX >= FUSED_K_MIN_COLS(200000)` 或
    # `tiles4probe*zSubMax >= FUSED_K4_MIN_BLOCKS(408)`，而矩阵里最大的形状也只有 65 tiles
    # ⇒ **默认矩阵从来不走 K=4 融合核**（`scanRectFusedKernelCount` / 改造后的 merge4 核）。
    # 这是生产负载实际走的路径，必须纳入位精确矩阵，否则「54 项全绿」对它是零覆盖。
    build_one "$K4ONLY" -DSLIME_K4_MIN_BLOCKS=0 && HAVE_K4=1 || BUILD_FAIL=1
fi

# ---------------------------------------------------------------- 目标清单
TARGETS="$VDIR/targets.tsv"
: > "$TARGETS"
printf 'ref|%s|cmp\n'    "$(readlink -f "$REF")" >> "$TARGETS"
printf 'new|%s|cmp\n'    "$(readlink -f "$NEW")" >> "$TARGETS"
[[ $HAVE_ZSUB1 -eq 1 ]] && printf 'zsub1|%s|cmp\n' "$(readlink -f "$ZSUB1")" >> "$TARGETS"
[[ $HAVE_K4 -eq 1 ]] && printf 'k4|%s|cmp\n' "$(readlink -f "$K4ONLY")" >> "$TARGETS"
[[ $HAVE_SCAN -eq 1 ]] && printf 'scanonly|%s|runonly\n' "$(readlink -f "$SCANONLY")" >> "$TARGETS"

# 每目标私有计划：k4 只跑融合路径形状（见 emit_plan_for），其余目标用全量计划。
while IFS='|' read -r tname tbin tmode; do
    [[ -z "${tname:-}" ]] && continue
    emit_plan_for "$tname" > "$VDIR/plan_$tname.tsv"
done < "$TARGETS"

# ---------------------------------------------------------------- worker（在 GPU 锁内执行）
WORKER="$VDIR/_worker.sh"
cat > "$WORKER" <<'WORKER_EOF'
#!/bin/bash
# 由 tools/verify.sh 生成：跑完整个矩阵，并把每项结果写成 TSV。
# 每个 CSV 算完 md5/去重计数后立即删除（磁盘纪律），日志保留供事后分析。
set -uo pipefail
PLAN="$1"; RUN="$2"; TARGETS="$3"; VDIR="$4"
LOGDIR="$RUN/logs"; CSVDIR="$RUN/csv"
# 清掉上一轮的逐目标 TSV：本轮目标是动态的（--fast 会跳过编译期变体，targets.tsv 里
# 就没有 zsub1/k4），若不清，报告会照着**上一轮残留**的 zsub1.tsv 逐项比对，产生
# 一批假 FAIL（本目录实测：5 个）。只清 *.tsv，日志与日志目录保留供排查。
rm -f "$RUN"/*.tsv
date +%s%N > "$RUN/worker_start.txt"   # 此刻已持有 flock 且温度到位
mkdir -p "$LOGDIR" "$CSVDIR"

poll_runs() {   # 轮询 .runN（spill 落盘段）文件数，记录观察到的峰值
    local csv="$1" flag="$2" out="$3" m=0 n
    while [[ -e "$flag" ]]; do
        n=$(ls "$csv".run* 2>/dev/null | wc -l)
        if [[ "$n" -gt "$m" ]]; then m="$n"; echo "$m" > "$out"; fi
        sleep 0.05
    done
}

run_one() {
    local tname="$1" bin="$2" id="$3" kind="$4" env="$5" args="$6" flags="$7"
    local csv="$CSVDIR/${tname}__${id}.csv" log="$LOGDIR/${tname}__${id}.log"
    local sorted="$CSVDIR/${tname}__${id}.sorted" data="$CSVDIR/${tname}__${id}.data"
    local runcnt="$CSVDIR/${tname}__${id}.runcnt" runflag="$CSVDIR/${tname}__${id}.flag"
    local -a argv=() envv=()
    read -r -a argv <<< "$args"
    read -r -a envv <<< "$env"
    rm -f "$csv" "$sorted" "$data" "$runcnt" "$runflag" "$csv".run*

    local poller="" t0 t1 rc wall_ms
    if [[ "$flags" == *spill* ]]; then
        : > "$runflag"; echo 0 > "$runcnt"
        poll_runs "$csv" "$runflag" "$runcnt" &
        poller=$!
    fi
    # 硬超时：万一日后改动引入死锁，绝不能让一个挂住的进程一直占着全局 GPU 锁
    local -a TO=(timeout -k 5 "${SLIME_VERIFY_TIMEOUT:-900}")
    t0=$(date +%s%N)
    if (( ${#envv[@]} > 0 )); then
        env "${envv[@]}" "${TO[@]}" "$bin" "${argv[@]}" "$csv" >"$log" 2>&1
    else
        "${TO[@]}" "$bin" "${argv[@]}" "$csv" >"$log" 2>&1
    fi
    rc=$?
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    if [[ -n "$poller" ]]; then rm -f "$runflag"; wait "$poller" 2>/dev/null; fi

    # ---- 解析日志 ----
    local valid rows=0 uniq=0 md5=- gx0=- gx1=- gz0=- gz1=- runfiles=0 hastv=0 redo=0
    local dx=- dz=- minx=- maxx=- minz=- maxz=- bad16=- blocks=0 chunks=0
    valid=$(sed -n 's/^Total valid: \([0-9]*\).*/\1/p' "$log" | tail -1)
    [[ -n "$valid" ]] && hastv=1 || valid=-
    read -r gx0 gx1 gz0 gz1 < <(sed -n \
        's/^Global range: X\[[^]]*\] Z\[[^]]*\] -> \[\(-\{0,1\}[0-9]*\),\(-\{0,1\}[0-9]*\)\] Z\[\(-\{0,1\}[0-9]*\),\(-\{0,1\}[0-9]*\)\].*/\1 \2 \3 \4/p' \
        "$log" | tail -1)
    local cudaerr
    cudaerr=$(grep -c -E 'CUDA error|cudaError|an illegal memory access|internal: chunk retry limit|out of memory|OOM:' "$log")
    [[ -f "$runcnt" ]] && runfiles=$(cat "$runcnt")
    redo=$(grep -o '[0-9][0-9]* redo' "$log" | awk '{s+=$1} END{print s+0}')
    blocks=$(grep -c '^GPU:' "$log")                                        # Z 大块迭代次数
    chunks=$(grep -o '[0-9][0-9]* chunk(s)' "$log" | awk '{s+=$1} END{print s+0}')  # 批次总数

    # ---- 哈希（必须排序后比，且必须在真实磁盘上排序）----
    # 性能纪律：锁内时间越长，别的 Agent 等得越久。所以 md5 用 sort+md5sum（不可省），
    # 而不变式统计只在真的会用到它的目标（ref/new）上算，且全部用 C 工具（cut/uniq/sed）。
    if [[ -f "$csv" ]]; then
        rows=$(wc -l < "$csv")
        if LC_ALL=C sort -S 1G -o "$sorted" "$csv" 2>>"$log"; then
            md5=$(md5sum < "$sorted" | cut -d' ' -f1)
            if [[ "${NEED_STATS:-0}" == "1" ]]; then
                # 排序后表头恒为最后一行（'x' > 数字和 '-'），sed '$d' 即可去掉
                sed '$d' "$sorted" > "$data"
                # 注意：文件是按整行字符串排序的，所以「相同的 x」「相同的 (x,z) 对」天然相邻，
                # 可以直接 uniq 去重；而 z 不是主键（每个 x 段都会重来一遍），uniq 数不出不同的 z。
                # 不去数 z 也不会漏检：下面的检查里
                #   pairs == rangeX*validZ 且 dx == rangeX 且 Z 全部 16 对齐且落在 [z0,z1]
                # 已经蕴含「每个 x 都恰好取满 validZ 个 z」——否则总数必然 < rangeX*validZ。
                uniq=$(cut -d, -f1,2 "$data" | uniq | wc -l)   # 去重 (x,z) 对
                dx=$(cut -d, -f1 "$data" | uniq | wc -l)       # 去重 X 个数（x 是主键，可 uniq）
                # 各轴极值 + 非 16 倍数（区块坐标 ×16）坏行数：单趟、无数组，尽量省
                read -r minx maxx minz maxz bad16 < <(awk -F, '
                    { if (n++ == 0) { minx = maxx = $1; minz = maxz = $2 }
                      if ($1 < minx) minx = $1;  if ($1 > maxx) maxx = $1
                      if ($2 < minz) minz = $2;  if ($2 > maxz) maxz = $2
                      if ($1 % 16 != 0 || $2 % 16 != 0) bad++ }
                    END { if (n == 0) print "- - - - -"; else print minx, maxx, minz, maxz, bad + 0 }' "$data")
            fi
        else
            md5="SORT_FAIL"
        fi
    fi
    rm -f "$csv" "$sorted" "$data" "$csv".run* "$runcnt" "$runflag"

    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "$id" "$rc" "$valid" "$rows" "$uniq" "$md5" "$wall_ms" \
        "$gx0" "$gx1" "$gz0" "$gz1" "$runfiles" "$cudaerr" "$hastv" "$redo" \
        "$dx" "$dz" "$minx" "$maxx" "$minz" "$maxz" "$bad16" "$blocks" "$chunks" >> "$RUN/$tname.tsv"
}

while IFS='|' read -r tname bin tmode; do
    [[ -z "${tname:-}" ]] && continue
    echo "[$tname] $bin ($tmode)" >&2
    : > "$RUN/$tname.tsv"
    while IFS='|' read -r id group kind env args flags; do
        [[ -z "${id:-}" ]] && continue
        if [[ "$tmode" == "runonly" ]]; then
            [[ "$flags" == *scan* && "$kind" != "reject" ]] || continue
        fi
        NEED_STATS=0; [[ "$tname" == "ref" || "$tname" == "new" ]] && NEED_STATS=1
        run_one "$tname" "$bin" "$id" "$kind" "$env" "$args" "$flags"
    done < "$VDIR/plan_$tname.tsv"
done < "$TARGETS"
echo "worker done" >&2
WORKER_EOF
chmod +x "$WORKER"

# ---------------------------------------------------------------- 执行（GPU 锁内）
echo "== matrix: $NPLAN entries; targets: $(cut -d'|' -f1 "$TARGETS" | tr '\n' ' ')"
echo "== reference: $REF  ($(md5sum < "$(readlink -f "$REF")" | cut -c1-12))"
echo "== under test: $NEW  ($(md5sum < "$(readlink -f "$NEW")" | cut -c1-12))"
REF_MD5_FILE="$(dirname "$REF")/$(basename "$REF").md5"
if [[ -f "$REF_MD5_FILE" ]]; then
    want=$(awk '{print $1; exit}' "$REF_MD5_FILE")
    got=$(md5sum < "$(readlink -f "$REF")" | cut -d' ' -f1)
    [[ "$want" == "$got" ]] && echo "== reference integrity: OK ($got)" \
        || echo "== reference integrity: !! md5 mismatch (recorded $want != actual $got)" >&2
else
    echo "== reference integrity: skipped (no $REF_MD5_FILE)"
fi

T_ALL0=$(date +%s%N)
bash "$WORKER" "$PLAN" "$RUN" "$TARGETS" "$VDIR" || echo "worker exited non-zero (parsing whatever results exist)" >&2
T_ALL1=$(date +%s%N)

# ---------------------------------------------------------------- 报告
declare -A V
declare -A G_MD5 G_VALID G_ID
load() {
    local t="$1" f="$RUN/$t.tsv"
    [[ -f "$f" ]] || return 0
    local id rc valid rows uniq md5 wall gx0 gx1 gz0 gz1 runfiles cudaerr hastv redo
    local dx dz minx maxx minz maxz bad16 blocks chunks
    while IFS='|' read -r id rc valid rows uniq md5 wall gx0 gx1 gz0 gz1 runfiles cudaerr hastv redo \
                          dx dz minx maxx minz maxz bad16 blocks chunks; do
        [[ -z "${id:-}" ]] && continue
        V["$t|$id|rc"]="$rc";       V["$t|$id|valid"]="$valid"; V["$t|$id|rows"]="$rows"
        V["$t|$id|uniq"]="$uniq";   V["$t|$id|md5"]="$md5";     V["$t|$id|wall"]="$wall"
        V["$t|$id|gx0"]="$gx0";     V["$t|$id|gx1"]="$gx1"
        V["$t|$id|gz0"]="$gz0";     V["$t|$id|gz1"]="$gz1"
        V["$t|$id|runs"]="$runfiles"; V["$t|$id|cuda"]="$cudaerr"; V["$t|$id|tv"]="$hastv"
        V["$t|$id|redo"]="$redo"
        V["$t|$id|dx"]="$dx";     V["$t|$id|dz"]="$dz"
        V["$t|$id|minx"]="$minx"; V["$t|$id|maxx"]="$maxx"
        V["$t|$id|minz"]="$minz"; V["$t|$id|maxz"]="$maxz"
        V["$t|$id|bad16"]="$bad16"
        V["$t|$id|blocks"]="$blocks"; V["$t|$id|chunks"]="$chunks"
    done < "$f"
}
for t in ref new zsub1 scanonly; do load "$t"; done

FAILS=0; NPASS=0; NORACLE=0      # NORACLE = 不变式告警数（默认 WARN，--strict-oracle 时计入失败）
GPU_MS=0
echo
printf '%-16s %-5s %-5s %-12s %-12s %-10s %-5s %s\n' \
       config   ref_rc new_rc ref_valid   new_valid   md5[0:8]  verdict note
printf -- '-------------------------------------------------------------------------------------------------\n'

while IFS='|' read -r id group kind env args flags; do
    [[ -z "${id:-}" ]] && continue
    local_rc_r="${V[ref|$id|rc]:--}"; local_rc_n="${V[new|$id|rc]:--}"
    vr="${V[ref|$id|valid]:--}";     vn="${V[new|$id|valid]:--}"
    mr="${V[ref|$id|md5]:--}";       mn="${V[new|$id|md5]:--}"
    cerr=$(( ${V[ref|$id|cuda]:-0} + ${V[new|$id|cuda]:-0} ))
    tvr="${V[ref|$id|tv]:-0}";       tvn="${V[new|$id|tv]:-0}"
    GPU_MS=$(( GPU_MS + ${V[ref|$id|wall]:-0} + ${V[new|$id|wall]:-0} ))

    verdict="PASS"; note=""
    if [[ -z "${V[ref|$id|rc]:-}" || -z "${V[new|$id|rc]:-}" ]]; then
        verdict="FAIL"; note="missing run result"
    elif [[ "$kind" == "reject" ]]; then
        if [[ "$local_rc_r" != "0" && "$local_rc_n" != "0" && "$local_rc_r" == "$local_rc_n" ]]; then
            verdict="PASS"; note="invalid input rejected consistently (rc=$local_rc_r)"
        else
            verdict="FAIL"; note="rejection mismatch ref_rc=$local_rc_r new_rc=$local_rc_n"
        fi
    else
        [[ "$local_rc_r" != "0" ]] && { verdict="FAIL"; note+="ref exit=$local_rc_r "; }
        [[ "$local_rc_n" != "0" ]] && { verdict="FAIL"; note+="new exit=$local_rc_n "; }
        [[ "$tvr" == "1" && "$tvn" != "1" ]] && { verdict="FAIL"; note+="new has no Total valid line "; }
        [[ "$tvn" == "1" && "$tvr" != "1" ]] && { verdict="FAIL"; note+="ref has no Total valid line "; }
        [[ "$vr" != "$vn" ]] && { verdict="FAIL"; note+="Total valid mismatch ($vr vs $vn) "; }
        [[ "$mr" != "$mn" ]] && { verdict="FAIL"; note+="CSV md5 mismatch ($mr vs $mn) "; }
        # 退出码为 0 但没产出可比对的 CSV（缺文件 / 排序失败 / 连表头都没有）同样是 FAIL，
        # 否则「两边都没写文件」会以 md5 都为 "-" 的形式骗过比对
        for t in ref new; do
            [[ "$t" == "ref" ]] && tm="$mr" || tm="$mn"
            [[ "$t" == "ref" ]] && tr_="${V[ref|$id|rows]:-0}" || tr_="${V[new|$id|rows]:-0}"
            if [[ "$tm" == "-" || "$tm" == "SORT_FAIL" || ! "$tr_" =~ ^[0-9]+$ || "$tr_" -lt 1 ]]; then
                verdict="FAIL"; note+="$t produced no comparable CSV (md5=$tm rows=$tr_) "
            fi
        done
        [[ "$cerr" != "0" ]] && { verdict="FAIL"; note+="log contains a CUDA error marker "; }
        [[ "$local_rc_r" == "124" || "$local_rc_n" == "124" ]] && note+="(rc=124 = killed by the ${SLIME_VERIFY_TIMEOUT:-900}s timeout; suspect deadlock/hang) "
        # ---- oracle：坐标网格 + 命中数必须与扫描范围自洽 ----
        #   thr=0 时每个窗口都命中，于是命中集合必须恰好是扫描范围内的整张窗口网格：
        #     行数 == rangeX*validZ、去重坐标数 == 行数、去重 X/Z 个数 == rangeX/validZ、
        #     且 X/Z 极值必须精确落在扫描范围两端（这条能抓住「计数对但坐标整体错位」的 bug）。
        #   非 thr=0 时退化为较弱的边界检查：所有坐标必须是 16 的倍数且落在扫描范围内。
        set -- $args
        local_sx="${6:-0}"; local_sz="${7:-0}"; local_thr="${8:-0}"
        for t in ref new; do
            g0="${V[$t|$id|gx0]:--}"; g1="${V[$t|$id|gx1]:--}"
            h0="${V[$t|$id|gz0]:--}"; h1="${V[$t|$id|gz1]:--}"
            got_valid="${V[$t|$id|valid]:--}"; got_rows="${V[$t|$id|rows]:-0}"
            got_uniq="${V[$t|$id|uniq]:-0}"
            gdx="${V[$t|$id|dx]:--}"; gdz="${V[$t|$id|dz]:--}"
            mnx="${V[$t|$id|minx]:--}"; mxx="${V[$t|$id|maxx]:--}"
            mnz="${V[$t|$id|minz]:--}"; mxz="${V[$t|$id|maxz]:--}"
            bad16="${V[$t|$id|bad16]:--}"
            if [[ "$g0" == "-" || "$local_sx" == "0" ]]; then
                continue
            elif [[ "$local_thr" == "0" ]]; then
                exp_dx=$(( g1 - g0 + 1 - local_sx + 1 )); exp_dz=$(( h1 - h0 + 1 - local_sz + 1 ))
                exp=$(( exp_dx * exp_dz ))
                if [[ "$got_valid" != "$exp" || "$got_rows" != "$((exp + 1))" || "$got_uniq" != "$exp" \
                   || "$gdx" != "$exp_dx" \
                   || "$mnx" != "$((g0 * 16))" || "$mxx" != "$(((g1 - local_sx + 1) * 16))" \
                   || "$mnz" != "$((h0 * 16))" || "$mxz" != "$(((h1 - local_sz + 1) * 16))" \
                   || "$bad16" != "0" ]]; then
                    NORACLE=$((NORACLE+1))
                    note+="ORACLE($t: grid non-conforming valid=$got_valid/$exp rows=$got_rows uniq=$got_uniq " \
                    note+="X[$mnx,$mxx] expect[$((g0*16)),$(((g1-local_sx+1)*16))] " \
                    note+="Z[$mnz,$mxz] expect[$((h0*16)),$(((h1-local_sz+1)*16))] dx=$gdx/$exp_dx bad16=$bad16) "
                fi
            elif [[ "$got_rows" =~ ^[0-9]+$ && "$got_rows" -gt 1 ]]; then   # 有空数据行才做弱边界检查
                bad=""
                [[ "$bad16" != "0" ]] && bad+="rows not multiple of 16=$bad16 "
                [[ "$mnx" =~ ^-?[0-9]+$ && "$mnx" -lt $((g0 * 16)) ]] && bad+="X below range($mnx) "
                [[ "$mxx" =~ ^-?[0-9]+$ && "$mxx" -gt $(((g1 - local_sx + 1) * 16)) ]] && bad+="X above range($mxx) "
                [[ "$mnz" =~ ^-?[0-9]+$ && "$mnz" -lt $((h0 * 16)) ]] && bad+="Z below range($mnz) "
                [[ "$mxz" =~ ^-?[0-9]+$ && "$mxz" -gt $(((h1 - local_sz + 1) * 16)) ]] && bad+="Z above range($mxz) "
                if [[ -n "$bad" ]]; then
                    NORACLE=$((NORACLE+1))
                    note+="ORACLE($t: coordinate out of range: $bad) "
                fi
            fi
            if [[ "$local_thr" -ge $(( local_sx * local_sz )) && "$local_thr" != "0" ]]; then
                if [[ "$got_valid" != "0" || "$got_rows" != "1" ]]; then
                    NORACLE=$((NORACLE+1))
                    note+="ORACLE($t: threshold above the window max but output is non-empty valid=$got_valid rows=$got_rows) "
                fi
            fi
        done
    fi
    # ---- 附加目标：zsub1 / k4（与参考逐字节比对）与 scanonly（只要求跑完）----
    # 新增编译期变体时必须在这里加分支，否则该目标跑了却不被判定（= 零覆盖）。
    for xt in zsub1 k4 scanonly; do
        [[ -n "${V[$xt|$id|rc]:-}" ]] || continue
        # 本轮目标里没有这个变体（例如 --fast 跳过了编译期变体）时跳过，不要拿别的轮次
        # 或别的目标的残留数据来判定。
        grep -q "^$xt|" "$TARGETS" || continue
        xr="${V[$xt|$id|rc]}"; xv="${V[$xt|$id|valid]:--}"; xm="${V[$xt|$id|md5]:--}"
        xc="${V[$xt|$id|cuda]:-0}"
        GPU_MS=$(( GPU_MS + ${V[$xt|$id|wall]:-0} ))
        if [[ "$xt" == "scanonly" ]]; then
            if [[ "$xr" == "0" && "$xc" == "0" ]]; then note+="scanonly=OK "
            else verdict="FAIL"; note+="scan_only failed (rc=$xr cudaerr=$xc) "; fi
        elif [[ "$xt" == "k4" ]]; then
            # 强制 K=4：与参考逐字节比对（项的取舍见 shape_is_fused）
            if [[ "$xr" == "0" && "$xv" == "$vr" && "$xm" == "$mr" ]]; then note+="k4=OK "
            else verdict="FAIL"; note+="k4 mismatch (rc=$xr valid=$xv md5=$xm vs ref $vr/$mr) "; fi
        elif [[ "$kind" == "reject" ]]; then
            if [[ "$xr" != "0" && "$xr" == "$local_rc_r" ]]; then note+="zsub1=OK "
            else verdict="FAIL"; note+="zsub1 rejection mismatch (rc=$xr, ref=$local_rc_r) "; fi
        elif [[ "$xr" == "0" && "$xv" == "$vr" && "$xm" == "$mr" ]]; then
            note+="zsub1=OK "
        else
            verdict="FAIL"; note+="zsub1 mismatch (rc=$xr valid=$xv md5=$xm) "
        fi
    done
    # ---- 分组自检：同 group 的配置（--sort=on/off 孪生）必须完全一致 ----
    if [[ -n "${group:-}" ]]; then
        gkey="${group}"
        if [[ -z "${G_MD5[$gkey]:-}" ]]; then
            G_MD5[$gkey]="$mr"; G_VALID[$gkey]="$vr"; G_ID[$gkey]="$id"
        elif [[ "${G_MD5[$gkey]}" != "$mr" || "${G_VALID[$gkey]}" != "$vr" ]]; then
            verdict="FAIL"
            note+="group[$gkey] differs from ${G_ID[$gkey]} (md5 $mr vs ${G_MD5[$gkey]}) "
        else
            note+="group[$gkey]=same as ${G_ID[$gkey]} "
        fi
    fi
    [[ "$verdict" == "PASS" ]] && NPASS=$((NPASS+1)) || FAILS=$((FAILS+1))
    printf '%-16s %-5s %-5s %-12s %-12s %-10s %-5s %s\n' \
        "$id" "$local_rc_r" "$local_rc_n" "$vr" "$vn" "${mr:0:8}" "$verdict" "$note"
done < "$PLAN"

# 覆盖率：证明那些「藏 bug 的路径」真的被走过了，而不是配置写了却没生效
echo
echo "== path-coverage evidence"
echo "   -- buffer full -> discard and retry / spill to disk (redo = retries; runs = peak .runN segments)"
while IFS='|' read -r id group kind env args flags; do
    [[ -n "${env:-}${flags:-}" ]] || continue
    line="   $(printf '%-16s' "$id")"
    hit=0
    for t in ref new zsub1; do
        r="${V[$t|$id|redo]:-}"; n="${V[$t|$id|runs]:-}"
        [[ -z "$r" ]] && continue
        line+=" $t(redo=$r,runs=$n)"
        [[ "$r" != "0" || "$n" != "0" ]] && hit=1
    done
    [[ $hit -eq 1 ]] && echo "$line"
done < "$PLAN"
echo "   -- multiple Z blocks / multiple batches (blocks = kernel Z-block iterations; chunks = total batches)"
while IFS='|' read -r id group kind env args flags; do
    [[ "$kind" != "reject" ]] || continue
    b="${V[ref|$id|blocks]:-0}"; c="${V[ref|$id|chunks]:-0}"
    [[ "$b" =~ ^[0-9]+$ ]] || continue
    if [[ "$b" -gt 1 || "$c" -gt 1 ]]; then
        printf '   %-16s ref(blocks=%s,chunks=%s)' "$id" "$b" "$c"
        [[ -n "${V[new|$id|blocks]:-}" ]] && printf ' new(blocks=%s,chunks=%s)' "${V[new|$id|blocks]}" "${V[new|$id|chunks]}"
        echo
    fi
done < "$PLAN"

TOT_MS=$(( (T_ALL1 - T_ALL0) / 1000000 ))
WSTART=$(cat "$RUN/worker_start.txt" 2>/dev/null || echo "")
if [[ -n "$WSTART" ]]; then
    LOCK_MS=$(( (WSTART - T_ALL0) / 1000000 )); EXEC_MS=$(( (T_ALL1 - WSTART) / 1000000 ))
else
    LOCK_MS=0; EXEC_MS=$TOT_MS
fi
echo
echo "== summary"
echo "   entries       : $NPLAN (targets: $(cut -d'|' -f1 "$TARGETS" | tr '\n' ' '))"
if [[ $BUILD_FAIL -eq 1 ]]; then
    echo "   variants      : compile failure -> matrix incomplete, judged FAIL (see $VDIR/build_extras.log)"
    FAILS=$((FAILS+1))
fi
echo "   passed/failed : $NPASS / $FAILS"
echo "   oracle warns  : $NORACLE$([[ $STRICT -eq 1 && $NORACLE -gt 0 ]] && echo '  (--strict-oracle -> counted as failure)')"
echo "   program time  : $(( GPU_MS / 1000 )).$(( (GPU_MS % 1000) / 100 )) s (sum over all targets/entries, excl. sort and hash)"
echo "   lock+cooldown : $(( LOCK_MS / 1000 )).$(( (LOCK_MS % 1000) / 100 )) s (longer when another job holds the lock -- expected)"
echo "   matrix+hash   : $(( EXEC_MS / 1000 )).$(( (EXEC_MS % 1000) / 100 )) s (inside the lock: run + sort + md5)"
echo "   total wall    : $(( TOT_MS / 1000 )).$(( (TOT_MS % 1000) / 100 )) s"
echo "   log directory : ${LOGDIR#$ROOT/}"
if [[ $STRICT -eq 1 && $NORACLE -gt 0 ]]; then FAILS=$((FAILS+NORACLE)); fi
if [[ $FAILS -gt 0 ]]; then
    echo "   verdict       : FAIL"
    exit 1
fi
echo "   verdict       : PASS (every configuration matches the reference byte for byte)"
exit 0
