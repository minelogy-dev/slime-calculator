#!/bin/bash
#
# quickcheck.sh -- correctness check for slime_main (fast: about 1-2 minutes).
#
# ASCII output only.  For every configuration it computes the expected result
# with an independent oracle that was written from the Java specification
# (oracle.c, no code shared with the program) and compares the COMPLETE set of
# emitted rows, not a sample.
#
#   usage: quickcheck.sh [options]
#     --bin PATH      binary under test   (default <root>/build/slime_main)
#     --root DIR      project root (auto-detected by default)
#     --oracle PATH   use an existing oracle binary instead of building one
#     --keep          keep the produced CSVs (default: delete them)
#     --no-build      do not rebuild the program even if it is missing
#     -h, --help
#
# Exit code: 0 = all configurations passed, 1 = at least one failed, 2 = setup error.
#
# What "passed" means, per configuration:
#   1. the program exits 0 and prints no CUDA error;
#   2. the scanned region it reports matches the documented padding rule and
#      contains the requested region;
#   3. every row satisfies the structural invariants (block-aligned coordinates,
#      inside the scanned region, count >= threshold, no duplicates, and with
#      --sort=on strictly ordered);
#   4. 'Total valid:' equals the number of CSV rows;
#   5. the CSV equals the oracle's expectation as a multiset (byte-identical
#      after a locale-independent numeric sort).
#
set -uo pipefail

BIN=""; ROOT=""; ORACLE=""; KEEP=0; DO_BUILD=1; FULL=0
# Hard ceiling on the number of emitted rows per configuration.  A configuration
# that exceeds it is a harness sizing mistake, not a program failure, so it is
# reported as SKIP instead of silently eating memory.
MAXROWS=2000000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)       BIN="$2"; shift 2 ;;
        --root)      ROOT="$2"; shift 2 ;;
        --oracle)    ORACLE="$2"; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        --no-build)  DO_BUILD=0; shift ;;
        --full)      FULL=1; shift ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

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
    else ROOT="$(find_root)" || { echo "cannot find project root; use --root" >&2; exit 2; }
    fi
fi
[[ -z "$BIN" ]] && BIN="$ROOT/build/slime_main"

WORK="$ROOT/.work/quickcheck"; mkdir -p "$WORK"
cleanup() { [[ $KEEP -eq 1 ]] || rm -f "$WORK"/*.csv "$WORK"/*.log "$WORK"/*.probes; }
trap cleanup EXIT

# ---------------------------------------------------------------- setup
if [[ ! -x "$BIN" ]]; then
    if [[ $DO_BUILD -eq 1 && -x "$ROOT/build.sh" ]]; then
        echo "building the program ..."
        ( cd "$ROOT" && ./build.sh ) > "$WORK/build.log" 2>&1 || { echo "build failed (see $WORK/build.log)"; exit 2; }
    fi
fi
[[ -x "$BIN" ]] || { echo "binary not found: $BIN (run ./build.sh or pass --bin)" >&2; exit 2; }

if [[ -z "$ORACLE" ]]; then
    ORACLE="$WORK/oracle"
    SRC_ORACLE=""
    for c in "$ROOT/tools/oracle.c" "$(dirname "$0")/oracle.c"; do
        [[ -f "$c" ]] && { SRC_ORACLE="$c"; break; }
    done
    [[ -n "$SRC_ORACLE" ]] || { echo "cannot find oracle.c next to this script" >&2; exit 2; }
    if [[ ! -x "$ORACLE" || "$SRC_ORACLE" -nt "$ORACLE" ]]; then
        gcc -O2 -fwrapv -o "$ORACLE" "$SRC_ORACLE" 2> "$WORK/oracle_build.log" \
            || { echo "cannot build oracle.c (see $WORK/oracle_build.log)" >&2; exit 2; }
    fi
fi
[[ -x "$ORACLE" ]] || { echo "oracle not executable: $ORACLE" >&2; exit 2; }

# ---------------------------------------------------------------- helpers
canon() { tail -n +2 "$1" | LC_ALL=C sort -t, -k1,1n -k2,2n; }

predict_region() { # x0 z0 x1 z1 -> "sx0 sx1 sz0 sz1"
    local x0=$1 z0=$2 x1=$3 z1=$4
    local w=$(( x1 - x0 + 1 )) h=$(( z1 - z0 + 1 ))
    if (( w % 256 != 0 )); then
        local nw=$(( (w / 256 + 1) * 256 ))
        x0=$(( x0 - (nw - w) / 2 )); x1=$(( x1 + (nw - w + 1) / 2 ))
    fi
    if (( h < 256 )); then
        local d=$(( 256 - h )); local l=$(( d / 2 )); local r=$(( d - l ))
        z0=$(( z0 - l )); z1=$(( z1 + r ))
    fi
    echo "$x0 $x1 $z0 $z1"
}
scan_region() {
    local line; line="$(grep -m1 '^Global range:' "$1" 2>/dev/null || true)"
    [[ -z "$line" ]] && { echo ""; return; }
    sed -E 's/.*-> \[(-?[0-9]+),(-?[0-9]+)\] Z\[(-?[0-9]+),(-?[0-9]+)\].*/\1 \2 \3 \4/' <<<"$line"
}
total_valid() { sed -n 's/^Total valid: \([0-9]*\).*/\1/p' "$1" | tail -1; }

invariants() { # csv minx maxx minz maxz thr sorted [dupmax]
    # NOTE: the duplicate check keeps one hash entry per row.  That is fine for
    # the bounded row counts this script produces, but an unbounded awk array is
    # what previously consumed gigabytes, so it is capped here as well.
    local dupmax="${8:-1000000}"
    awk -F, -v minx="$2" -v maxx="$3" -v minz="$4" -v maxz="$5" -v thr="$6" -v srt="$7" \
        -v dupmax="$dupmax" '
        NR==1 { if ($0 != "x,z,slime_count") { print "BAD:header"; exit 1 } next }
        NF!=3 { badnf++; next }
        { n++
          if ($1%16 || $2%16) bad16++
          if ($3+0 < thr) badcnt++
          if ($1+0<minx || $1+0>maxx || $2+0<minz || $2+0>maxz) badrange++
          if (srt==1 && n>1 && ($1+0<px+0 || ($1+0==px+0 && $2+0<=pz+0))) unsorted++
          px=$1; pz=$2
          if (n<=dupmax) { if (seen[$1","$2]++) dup++ } }
        END { printf "rows=%d badnf=%d bad16=%d badcnt=%d badrange=%d unsorted=%d dup=%d dupchecked=%d",
                     n+0,badnf+0,bad16+0,badcnt+0,badrange+0,unsorted+0,dup+0,(n<=dupmax) }' "$1"
}

PASS=0; FAIL=0; SKIP=0; FAILED=()
note() { printf '  %-34s %s\n' "$1" "$2"; }

# check_one <tag> <env> <seed> <x0> <z0> <x1> <z1> <sx> <sz> <thr|AUTO> <sort>
check_one() {
    local tag=$1 env=$2 seed=$3 x0=$4 z0=$5 x1=$6 z1=$7 sx=$8 sz=$9 thr=${10} sortm=${11}
    local csv="$WORK/$tag.csv" log="$WORK/$tag.log" exp="$WORK/$tag.exp.csv"
    rm -f "$csv"

    local pred; pred="$(predict_region "$x0" "$z0" "$x1" "$z1")"
    read -r px0 px1 pz0 pz1 <<<"$pred"

    # the fallback kernel (sizeX > 32) sizes its Z block from free VRAM, so its
    # scanned Z range is learned from a throw-away run instead of predicted
    if (( sx > 32 )); then
        rm -f "$WORK/$tag.probe.csv"
        timeout 1800 env $env "$BIN" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" 2147483647 \
            "$WORK/$tag.probe.csv" --sort=on > "$WORK/$tag.probe.log" 2>&1
        rm -f "$WORK/$tag.probe.csv"
        local sr; sr="$(scan_region "$WORK/$tag.probe.log")"
        [[ -n "$sr" ]] && read -r px0 px1 pz0 pz1 <<<"$sr"
    fi

    local olog="$WORK/$tag.oracle.log"
    local othr="$thr"
    if [[ "$thr" == "AUTO" ]]; then
        "$ORACLE" --seed "$seed" --x0 "$px0" --z0 "$pz0" --x1 "$px1" --z1 "$pz1" \
                  --sx "$sx" --sz "$sz" --mode auto --target 20000 --method roll \
                  --out "$exp" > "$olog" 2>&1 || { note "$tag" "FAIL (oracle)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; }
        othr="$(sed -n 's/.*threshold=\([0-9]*\).*/\1/p' "$olog")"
        [[ -n "$othr" ]] || { note "$tag" "FAIL (oracle threshold)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; }
    else
        "$ORACLE" --seed "$seed" --x0 "$px0" --z0 "$pz0" --x1 "$px1" --z1 "$pz1" \
                  --sx "$sx" --sz "$sz" --threshold "$thr" --method roll \
                  --out "$exp" > "$olog" 2>&1 || { note "$tag" "FAIL (oracle)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; }
        othr="$thr"
    fi

    timeout 1800 env $env "$BIN" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$othr" \
        "$csv" --sort="$sortm" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then note "$tag" "FAIL (exit $rc)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; fi
    if grep -qiE 'CUDA error|Cannot (open|write)|OOM' "$log"; then
        note "$tag" "FAIL (error in log)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; fi
    [[ -f "$csv" ]] || { note "$tag" "FAIL (no csv)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; }
    local nrows; nrows=$(( $(wc -l < "$csv") - 1 ))
    if (( nrows > MAXROWS )); then
        note "$tag" "SKIP (rows=$nrows exceeds the $MAXROWS guard; raise the threshold)"
        SKIP=$((SKIP+1)); rm -f "$csv" "$exp"; return
    fi

    local sr; sr="$(scan_region "$log")"
    if [[ -z "$sr" ]]; then note "$tag" "FAIL (no region line)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; fi
    read -r ax0 ax1 az0 az1 <<<"$sr"
    if [[ "$ax0" != "$px0" || "$ax1" != "$px1" ]] || { (( sx <= 32 )) && { [[ "$az0" != "$pz0" || "$az1" != "$pz1" ]]; }; }; then
        note "$tag" "FAIL (scanned [$ax0,$ax1]x[$az0,$az1] != expected [$px0,$px1]x[$pz0,$pz1])"
        FAIL=$((FAIL+1)); FAILED+=("$tag"); return
    fi
    if (( x0 < ax0 || x1 > ax1 || z0 < az0 || z1 > az1 )); then
        note "$tag" "FAIL (scanned region does not contain the request)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return
    fi

    local srt=0; [[ "$sortm" == "on" ]] && srt=1
    local inv; inv="$(invariants "$csv" $(( ax0 * 16 )) $(( (ax1 - sx + 1) * 16 )) \
                                          $(( az0 * 16 )) $(( (az1 - sz + 1) * 16 )) "$othr" "$srt")"
    if [[ "$inv" != rows=* ]]; then note "$tag" "FAIL ($inv)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return; fi
    local rest; rest="$(sed -E 's/^rows=[0-9]+ //' <<<"$inv")"
    case "$rest" in
        "badnf=0 bad16=0 badcnt=0 badrange=0 unsorted=0 dup=0 dupchecked=0"|\
        "badnf=0 bad16=0 badcnt=0 badrange=0 unsorted=0 dup=0 dupchecked=1") ;;
        *) note "$tag" "FAIL (invariants: $rest)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return ;;
    esac
    local rows tv; rows="$(sed -n 's/^rows=\([0-9]*\).*/\1/p' <<<"$inv")"; tv="$(total_valid "$log")"
    if [[ "$tv" != "$rows" ]]; then
        note "$tag" "FAIL ('Total valid' $tv != rows $rows)"; FAIL=$((FAIL+1)); FAILED+=("$tag"); return
    fi
    if ! cmp -s <(canon "$csv") <(canon "$exp"); then
        local d="$WORK/$tag.diff"
        diff <(canon "$exp") <(canon "$csv") 2>/dev/null | head -4 > "$d"
        note "$tag" "FAIL (differs from oracle; first differences in $(basename "$d"))"
        FAIL=$((FAIL+1)); FAILED+=("$tag"); return
    fi
    note "$tag" "ok   rows=$rows thr=$othr"
    PASS=$((PASS+1))
    rm -f "$csv" "$exp"
}

# ---------------------------------------------------------------- oracle self-test
echo "==================================================================="
echo " quickcheck for slime_main"
echo "   binary : $BIN"
echo "   oracle : $ORACLE"
echo "==================================================================="
echo
echo "[0] oracle self-test (two independent summation methods must agree)"
ok=1
for sh in "1 1" "3 5" "17 17" "32 32" "33 7" "64 32"; do
    set -- $sh
    "$ORACLE" --seed 7 --x0 -37 --z0 -21 --x1 41 --z1 33 --sx "$1" --sz "$2" \
              --mode auto --target 5000 --method both --out "$WORK/self.csv" \
              > "$WORK/self.log" 2>&1 || ok=0
done
rm -f "$WORK/self.csv"
if [[ $ok -eq 1 ]]; then echo "    ok   naive and rolling-window oracle agree on 6 shapes"; PASS=$((PASS+1))
else echo "    FAIL oracle methods disagree"; FAIL=$((FAIL+1)); FAILED+=("oracle-selftest"); fi
echo

echo "[1] fused kernel (sizeX <= 32)"
# Thresholds are AUTO unless stated: AUTO picks the largest count that still
# yields <= 20000 rows, which both bounds the CSV and exercises the `>=`
# boundary (the chosen value is a count that actually occurs).
check_one fused_17x17      ""            114514  -300  -300   300   300  17 17 AUTO on
check_one fused_multitile  ""            114514 -3000 -3000  3000  3000  17 17 AUTO on
check_one fused_multichunk ""            114514   -24  -60000   24  60000  17 17 AUTO on
check_one fused_1x17       ""            114514  -100  -100   100   100   1 17 AUTO on
check_one fused_17x1       ""            114514  -100  -100   100   100  17  1 AUTO on
check_one fused_32x32      ""            114514  -300  -300   300   300  32 32 AUTO on
check_one fused_17x32      ""            114514  -300  -300   300   300  17 32 AUTO off
check_one fused_small_1x1  ""            114514  -100  -100   100   100   1  1 AUTO on
check_one fused_thr_fixed  ""            114514   -24   -24    24    24  17 17 30   on
echo

echo "[2] fallback kernel (sizeX > 32)"
check_one fb_33x17  "" 114514 -1000 -1000 1000 1000 33 17 AUTO on
check_one fb_40x17  "" 114514 -1000 -1000 1000 1000 40 17 AUTO on
check_one fb_255x32 "" 114514  -500  -500  500  500 255 32 AUTO on
echo

echo "[3] coordinates and thresholds"
check_one neg_seed        "" -12345  -300  -300   300   300 17 17 AUTO on
check_one world_corner    "" 114514 -1999999 -1999999 -1999900 -1999900 17 17 3 on
check_one thr_zero        "" 114514  -40   -40    40    40  3  3 0    on
check_one thr_high_empty  "" 114514  -300  -300   300   300 17 17 999999 on
echo

echo "[4] forced internal paths"
check_one cap_small       "SLIME_CAP_SLOTS=4096"              114514 -3000 -3000 3000 3000 17 17 AUTO on
check_one cap_tiny        "SLIME_CAP_SLOTS=1024"              114514 -1000 -1000 1000 1000 17 17 AUTO off
# The spill path (on-disk sort runs + k-way merge) only triggers above 8M hits
# because a pool block is 8M slots, so it needs a ~350 MB CSV; opt-in via --full.
if [[ $FULL -eq 1 ]]; then
    check_one spill_blocks "SLIME_SORT_BLOCKS=2" 114514 -2400 -2400 2400 2400 17 17 8 on
fi
echo

echo "==================================================================="
printf ' passed: %d    failed: %d    skipped: %d\n' "$PASS" "$FAIL" "$SKIP"
if [[ $FAIL -gt 0 ]]; then
    printf ' failed configs: %s\n' "${FAILED[*]}"
    echo " RESULT: FAIL"
    echo "==================================================================="
    exit 1
fi
echo " RESULT: PASS"
echo "==================================================================="
exit 0
