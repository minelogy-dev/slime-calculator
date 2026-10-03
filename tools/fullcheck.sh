#!/bin/bash
#
# tools/fullcheck.sh -- slow but exhaustive correctness soak test for slime_main.
#
#   The point of this harness is to be RIGHT, not fast.  It spends hours of
#   exclusive GPU time trying to falsify the program, using an oracle that was
#   written from the Java specification (tools/oracle.c) and shares no code
#   with the program under test.
#
#   Chain of trust:
#     Java spec  ->  tools/oracle.c        (independent implementation)
#                ->  tools/oracle_pycheck.py (third implementation, P0 only)
#                ->  slime_main             (the program under test)
#   If the oracle is wrong the whole thing is worthless, so P0 cross-checks the
#   oracle against a Python implementation of the same spec and the harness
#   includes a negative control (a deliberately broken binary MUST be caught).
#
#   What is compared:
#     * the complete set of emitted rows (canonicalised, numerically sorted)
#       against the oracle for the SCANNED region the program reports;
#     * the scanned region itself against the documented padding rule;
#     * per-row invariants (block alignment, range, threshold, uniqueness,
#       ordering) for every row of every run, including huge regions that the
#       oracle cannot materialise;
#     * for huge regions, random single-candidate probes (oracle --probe) that
#       catch both spurious rows and MISSING rows.
#
#   Usage:
#     tools/fullcheck.sh [--hours N] [--bin PATH] [--phases "0 1 2 3 4"]
#                        [--workdir DIR] [--target N] [--resume] [--no-build]
#                        [--timeout SEC]
#
#   Output is pure ASCII (this is meant to run on a headless machine).
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="build/slime_main"
HOURS=6
WORK=".fullcheck"
PHASES="0 1 2 3 4 5"
DO_BUILD=1
RESUME=0
TARGET=20000
TIMEOUT=3600
LIMIT=0
CFG_COUNT=0
SEED_MAIN=114514
SEED_ALT=-12345

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours)    HOURS="$2"; shift 2 ;;
        --bin)      BIN="$2"; shift 2 ;;
        --phases)   PHASES="$2"; shift 2 ;;
        --workdir)  WORK="$2"; shift 2 ;;
        --target)   TARGET="$2"; shift 2 ;;
        --timeout)  TIMEOUT="$2"; shift 2 ;;
        --limit)    LIMIT="$2"; shift 2 ;;
        --resume)   RESUME=1; shift ;;
        --no-build) DO_BUILD=0; shift ;;
        -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$WORK/bin" "$WORK/log" "$WORK/csv"
RESULTS="$WORK/results.tsv"
FAILURES="$WORK/failures.txt"
DONE="$WORK/done.txt"
: > "$RESULTS"; : > "$FAILURES"; [[ -f "$DONE" ]] || : > "$DONE"

T0=$(date +%s)
DEADLINE=$(( T0 + HOURS * 3600 ))
NPASS=0; NFAIL=0; NSKIP=0; NRUN=0
declare -A PH_PASS PH_FAIL PH_TOTAL

ts()   { date '+%H:%M:%S'; }
say()  { printf '[%s] %s\n' "$(ts)" "$*"; }
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
out_of_time() { [[ $(date +%s) -ge $DEADLINE ]]; }
# true when a phase should stop: out of time budget, or --limit configs reached
phase_stop() { out_of_time && return 0; [[ $LIMIT -gt 0 && $CFG_COUNT -ge $LIMIT ]] && return 0; return 1; }
new_cfg() { CFG_COUNT=$((CFG_COUNT+1)); }

DRY=0
rec() { # rec <phase> <tag> <status> <detail>
    [[ $DRY -eq 1 ]] && return 0
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$(ts)" >> "$RESULTS"
    PH_TOTAL[$1]=$(( ${PH_TOTAL[$1]:-0} + 1 ))
    case "$3" in
        PASS) NPASS=$((NPASS+1)); PH_PASS[$1]=$(( ${PH_PASS[$1]:-0} + 1 )) ;;
        SKIP) NSKIP=$((NSKIP+1)) ;;
        *)    NFAIL=$((NFAIL+1)); PH_FAIL[$1]=$(( ${PH_FAIL[$1]:-0} + 1 ))
              printf 'phase=%s tag=%s status=%s detail=%s\n' "$1" "$2" "$3" "$4" >> "$FAILURES" ;;
    esac
}

# ---------------------------------------------------------------- helpers

# canonical form: rows only, numerically sorted by (x, z), locale independent
canon() { tail -n +2 "$1" | LC_ALL=C sort -t, -k1,1n -k2,2n; }

# documented padding rule, replicated exactly (integer division like C)
predict_region() { # <x0> <z0> <x1> <z1>  ->  echoes "sx0 sx1 sz0 sz1"
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

scan_region() { # <log> -> echoes "sx0 sx1 sz0 sz1", empty if not found
    local line
    line="$(grep -m1 '^Global range:' "$1" 2>/dev/null || true)"
    [[ -z "$line" ]] && { echo ""; return; }
    sed -E 's/.*-> \[(-?[0-9]+),(-?[0-9]+)\] Z\[(-?[0-9]+),(-?[0-9]+)\].*/\1 \2 \3 \4/' <<<"$line"
}

total_valid() { sed -n 's/^Total valid: \([0-9]*\).*/\1/p' "$1" | tail -1; }

# validate one CSV against structural invariants; echoes "rows=.. bad..=.." or BAD:<reason>
csv_invariants() { # <csv> <minx> <maxx> <minz> <maxz> <threshold> <sorted:0|1> [dupmax]
    local dupmax="${8:-2000000}"
    awk -F, -v minx="$2" -v maxx="$3" -v minz="$4" -v maxz="$5" -v thr="$6" -v need_sorted="$7" \
        -v dupmax="$dupmax" '
        NR == 1 { if ($0 != "x,z,slime_count") { print "BAD:header"; exit 1 } next }
        NF != 3 { badnf++; next }
        {
            n++
            if ($1 % 16 != 0 || $2 % 16 != 0) bad16++
            if ($3 + 0 < thr) badcnt++
            if ($1 + 0 < minx || $1 + 0 > maxx || $2 + 0 < minz || $2 + 0 > maxz) badrange++
            if (n > 1 && need_sorted == 1) {
                if ($1 + 0 < px + 0 || ($1 + 0 == px + 0 && $2 + 0 <= pz + 0)) unsorted++
            }
            px = $1; pz = $2
            if (n <= dupmax) { if (seen[$1 "," $2]++) dup++ }
        }
        END {
            printf "rows=%d badnf=%d bad16=%d badcnt=%d badrange=%d unsorted=%d dup=%d dupchecked=%d\n",
                   n+0, badnf+0, bad16+0, badcnt+0, badrange+0, unsorted+0, dup+0, (n<=dupmax)
        }' "$1"
}

# ---------------------------------------------------------------- run one program
# run_prog <bin> <env> <seed> <x0> <z0> <x1> <z1> <sx> <sz> <thr> <csv> <sort> <log>
run_prog() {
    local bin=$1 env=$2 seed=$3 x0=$4 z0=$5 x1=$6 z1=$7 sx=$8 sz=$9 thr=${10} csv=${11} sortm=${12} log=${13}
    rm -f "$csv"
    # shellcheck disable=SC2086
    timeout "$TIMEOUT" env $env "$bin" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr" \
        "$csv" --sort="$sortm" > "$log" 2>&1
    return $?
}

# ---------------------------------------------------------------- one full check
# check_one <phase> <tagprefix> <bin> <env> <seed> <x0> <z0> <x1> <z1> <sx> <sz>
#           <sort> <expcsv> <thr-or-AUTO> <sx0> <sx1> <sz0> <sz1>
check_one() {
    local phase=$1 tag=$2 bin=$3 env=$4 seed=$5 x0=$6 z0=$7 x1=$8 z1=$9
    local sx=${10} sz=${11} sortm=${12} expcsv=${13} thr=${14}
    local psx0=${15} psx1=${16} psz0=${17} psz1=${18}
    local tagfull="$tag"
    local csv="$WORK/csv/${tag}.csv" log="$WORK/log/${tag}.log"

    if [[ $RESUME -eq 1 ]] && grep -qxF "$tagfull" "$DONE" 2>/dev/null; then
        rec "$phase" "$tagfull" SKIP "already done"
        return 0
    fi
    NRUN=$((NRUN+1))
    local _t0=$SECONDS

    run_prog "$bin" "$env" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr" "$csv" "$sortm" "$log"
    local rc=$?
    [[ ${FC_DEBUG:-0} == 1 ]] && echo "    [dbg] run_prog=$((SECONDS-_t0))s" >&2
    local _t1=$SECONDS
    if [[ $rc -ne 0 ]]; then
        rec "$phase" "$tagfull" FAIL "exit=$rc (see $log)"
        return 1
    fi
    if grep -qiE 'CUDA error|internal: chunk retry|Cannot (open|write)|OOM' "$log"; then
        rec "$phase" "$tagfull" FAIL "error marker in log"
        return 1
    fi
    [[ -f "$csv" ]] || { rec "$phase" "$tagfull" FAIL "no csv produced"; return 1; }

    # scanned region must match the documented rule (X always; Z for fused path)
    local sr; sr="$(scan_region "$log")"
    if [[ -z "$sr" ]]; then rec "$phase" "$tagfull" FAIL "no 'Global range' line"; return 1; fi
    read -r asx0 asx1 asz0 asz1 <<<"$sr"
    if [[ "$asx0" != "$psx0" || "$asx1" != "$psx1" ]]; then
        rec "$phase" "$tagfull" FAIL "scanned X [$asx0,$asx1] != predicted [$psx0,$psx1]"
        return 1
    fi
    if (( sx <= 32 )) && [[ "$asz0" != "$psz0" || "$asz1" != "$psz1" ]]; then
        rec "$phase" "$tagfull" FAIL "scanned Z [$asz0,$asz1] != predicted [$psz0,$psz1] (fused path)"
        return 1
    fi
    # requested region must be contained in the scanned region
    if (( x0 < asx0 || x1 > asx1 || z0 < asz0 || z1 > asz1 )); then
        rec "$phase" "$tagfull" FAIL "scanned region does not contain the requested one"
        return 1
    fi

    # structural invariants
    local minx=$(( asx0 * 16 )) maxx=$(( (asx1 - sx + 1) * 16 ))
    local minz=$(( asz0 * 16 )) maxz=$(( (asz1 - sz + 1) * 16 ))
    local sortedflag=0; [[ "$sortm" == "on" ]] && sortedflag=1
    local inv; inv="$(csv_invariants "$csv" "$minx" "$maxx" "$minz" "$maxz" "$thr" "$sortedflag")"
    if [[ "$inv" != rows=* ]]; then rec "$phase" "$tagfull" FAIL "$inv"; return 1; fi
    local rows; rows="$(sed -n 's/^rows=\([0-9]*\).*/\1/p' <<<"$inv")"
    local rest; rest="$(sed -E 's/^rows=[0-9]+ //' <<<"$inv")"
    case "$rest" in
        "badnf=0 bad16=0 badcnt=0 badrange=0 unsorted=0 dup=0 dupchecked=0"|\
        "badnf=0 bad16=0 badcnt=0 badrange=0 unsorted=0 dup=0 dupchecked=1") ;;
        *) rec "$phase" "$tagfull" FAIL "invariant violation: $rest"; return 1 ;;
    esac
    local tv; tv="$(total_valid "$log")"
    if [[ -z "$tv" || "$tv" != "$rows" ]]; then
        rec "$phase" "$tagfull" FAIL "'Total valid' ($tv) != csv rows ($rows)"
        return 1
    fi

    # full oracle comparison
    if [[ -n "$expcsv" && -f "$expcsv" ]]; then
        if ! cmp -s <(canon "$csv") <(canon "$expcsv"); then
            local n1 n2
            n1="$(canon "$csv" | md5sum | cut -c1-12)"
            n2="$(canon "$expcsv" | md5sum | cut -c1-12)"
            diff <(canon "$expcsv") <(canon "$csv") | head -5 > "$WORK/log/${tag}.diff" 2>&1
            rec "$phase" "$tagfull" FAIL "content differs from oracle ($n1 vs $n2, see ${tag}.diff)"
            return 1
        fi
    fi

    [[ ${FC_DEBUG:-0} == 1 ]] && echo "    [dbg] checks=$((SECONDS-_t1))s rows=$rows" >&2
    rec "$phase" "$tagfull" PASS "rows=$rows"
    echo "$tagfull" >> "$DONE"
    rm -f "$csv"
    return 0
}

# The number of rows the oracle emits is roughly candidates * P(count >= t).
# For tiny rects that probability is large (a 1x1 rect hits ~10% of the time), so
# the region has to shrink or the CSV becomes millions of rows.
pick_region() { # <sx> <sz> -> echoes "x0 z0 x1 z1"
    local area=$(( $1 * $2 ))
    if (( area <= 1 )); then echo "-256 -64 255 63"
    elif (( area <= 4 )); then echo "-1024 -128 1023 127"
    else echo "-4096 -128 4095 127"
    fi
}

# binaries worth running for a given rect width
bins_for() { # <sx>  -> list
    if (( $1 <= 32 )); then echo "$VAR_LIST"; else echo "default"; fi
}

# prepare an expectation: predict region, run oracle, set EXP_THR/EXP_CSV/PRED_*
#   prep_expectation <tag> <seed> <x0> <z0> <x1> <z1> <sx> <sz> [method]
prep_expectation() {
    local tag=$1 seed=$2 x0=$3 z0=$4 x1=$5 z1=$6 sx=$7 sz=$8 method=${9:-roll}
    local pred; pred="$(predict_region "$x0" "$z0" "$x1" "$z1")"
    read -r PRED_X0 PRED_X1 PRED_Z0 PRED_Z1 <<<"$pred"
    if (( sx > 32 )); then
        # The fallback kernel (sizeX > 32) sizes H_max from free VRAM and then
        # truncates it to a multiple of 256, so the scanned Z range cannot be
        # predicted from the request alone.  Ask the program once (threshold is
        # unreachable, so this run itself is also the 'empty result' test).
        local plog="$WORK/log/probe_${tag}.log"
        run_prog "$BIN" "" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" 2147483647 \
                 "$WORK/csv/probe_${tag}.csv" on "$plog"
        local sr; sr="$(scan_region "$plog")"
        rm -f "$WORK/csv/probe_${tag}.csv"
        [[ -n "$sr" ]] && read -r PRED_X0 PRED_X1 PRED_Z0 PRED_Z1 <<<"$sr"
    fi
    EXP_CSV="$WORK/csv/exp_${tag}.csv"
    local ol="$WORK/log/oracle_${tag}.log"
    if ! "$ORACLE" --seed "$seed" --x0 "$PRED_X0" --z0 "$PRED_Z0" --x1 "$PRED_X1" --z1 "$PRED_Z1" \
            --sx "$sx" --sz "$sz" --mode auto --target "$TARGET" --method "$method" \
            --out "$EXP_CSV" > "$ol" 2>&1; then
        return 1
    fi
    EXP_THR="$(sed -n 's/.*threshold=\([0-9]*\).*/\1/p' "$ol")"
    EXP_HITS="$(sed -n 's/.*hits=\([0-9]*\).*/\1/p' "$ol")"
    [[ -n "$EXP_THR" ]] || return 1
    return 0
}

# ================================================================ phases

build_all() {
    say "building: program + oracle + knob variants"
    if [[ $DO_BUILD -eq 1 ]]; then
        ./build.sh > "$WORK/log/build_main.log" 2>&1 || { say "BUILD FAILED (main)"; return 1; }
    fi
    gcc -O2 -fwrapv -o build/oracle tools/oracle.c > "$WORK/log/build_oracle.log" 2>&1 \
        || { say "BUILD FAILED (oracle)"; return 1; }
    ORACLE=build/oracle
    [[ -x "$ORACLE" ]] || { say "oracle missing"; return 1; }

    # knob variants: every one must produce byte-identical results
    declare -gA VAR_BIN VAR_ENV
    local -a names=(default k4 k1 zsub1 zfine znosplit)
    local -a flags=("" \
        "-DSLIME_K4_MIN_BLOCKS=0" \
        "-DSLIME_K4_MIN_BLOCKS=2147483647" \
        "-DSLIME_Z_SUB_ROWS=1000000000 -DSLIME_Z_SPLIT_MIN_TILES=1000000000" \
        "-DSLIME_Z_SUB_ROWS=64 -DSLIME_Z_SPLIT_MIN_TILES=1000000000" \
        "-DSLIME_Z_SUB_ROWS=1000000000 -DSLIME_Z_SPLIT_MIN_TILES=1")
    local i
    for i in "${!names[@]}"; do
        local n="${names[$i]}" f="${flags[$i]}"
        if [[ "$n" == "default" ]]; then
            cp -f "$BIN" "$WORK/bin/$n" || return 1
        elif [[ $DO_BUILD -eq 1 || ! -x "$WORK/bin/$n" ]]; then
            # shellcheck disable=SC2086
            nvcc -o "$WORK/bin/$n" src/slime_main.cu -O3 -use_fast_math -arch=native \
                 -Xcompiler="-O3" $f > "$WORK/log/build_$n.log" 2>&1 || { say "BUILD FAILED ($n)"; return 1; }
        fi
    done
    VAR_LIST="${names[*]}"
    return 0
}

phase0() {
    say "== P0: sanity, oracle self-check, negative control =="
    # 0a. oracle vs python (third implementation)
    if command -v python3 >/dev/null 2>&1 && [[ -f tools/oracle_pycheck.py ]]; then
        if python3 tools/oracle_pycheck.py "$ORACLE" > "$WORK/log/pycheck.log" 2>&1; then
            rec 0 "oracle-vs-python" PASS "$(tail -1 "$WORK/log/pycheck.log")"
        else
            rec 0 "oracle-vs-python" FAIL "see log/pycheck.log"
        fi
    else
        rec 0 "oracle-vs-python" SKIP "python3 or checker missing"
    fi
    # 0b. oracle self-agreement: naive vs roll on a few shapes
    local ok=1
    for sh in "1 1" "3 5" "17 17" "32 32" "33 7" "64 32"; do
        set -- $sh
        if ! "$ORACLE" --seed 7 --x0 -37 --z0 -21 --x1 41 --z1 33 --sx "$1" --sz "$2" \
                --mode auto --target 5000 --method both --out "$WORK/csv/p0self.csv" \
                > "$WORK/log/p0self.log" 2>&1; then ok=0; fi
    done
    rm -f "$WORK/csv/p0self.csv"
    if [[ $ok -eq 1 ]]; then rec 0 "oracle-naive-vs-roll" PASS "6 shapes agree"
    else rec 0 "oracle-naive-vs-roll" FAIL "roll and naive disagree"; fi

    # 0c. known-answer: threshold 0 on a tiny region must emit exactly every candidate
    local csv="$WORK/csv/p0zero.csv"
    if run_prog "$BIN" "" 0 -8 -8 8 8 3 3 0 "$csv" on "$WORK/log/p0zero.log"; then
        local rows; rows=$(( $(wc -l < "$csv") - 1 ))
        # requested 17x17 chunks -> padded X to 256 -> 240 valid cols; Z 17 -> padded to 256 -> 238 rows
        local pred; pred="$(predict_region -8 -8 8 8)"; read -r a b c d <<<"$pred"
        local want=$(( (b - a + 1 - 3 + 1) * (d - c + 1 - 3 + 1) ))
        if [[ "$rows" == "$want" ]]; then rec 0 "known-answer-thr0" PASS "rows=$rows"
        else rec 0 "known-answer-thr0" FAIL "rows=$rows want=$want"; fi
    else
        rec 0 "known-answer-thr0" FAIL "run failed"
    fi
    rm -f "$csv"

    # 0d. negative control: a deliberately broken binary MUST be caught
    sed 's/d_results\[pos\]\.z = z \* 16;/d_results[pos].z = (z + 1) * 16;/' src/slime_main.cu \
        > "$WORK/bin/broken.cu"
    if grep -q "z + 1" "$WORK/bin/broken.cu" && \
       nvcc -o "$WORK/bin/broken" "$WORK/bin/broken.cu" -O3 -use_fast_math -arch=native \
            -Xcompiler="-O3" > "$WORK/log/build_broken.log" 2>&1; then
        local tag="p0neg"
        if prep_expectation "$tag" 114514 -64 -64 64 64 17 17 roll; then
            DRY=1
            if check_one 0 "$tag" "$WORK/bin/broken" "" 114514 -64 -64 64 64 17 17 on \
                    "$EXP_CSV" "$EXP_THR" "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" \
                    >/dev/null 2>&1; then
                DRY=0
                rec 0 "negative-control" FAIL "broken binary was NOT caught"
            else
                DRY=0
                rec 0 "negative-control" PASS "broken binary correctly rejected"
            fi
        else
            rec 0 "negative-control" FAIL "oracle prep failed"
        fi
        rm -f "$WORK/csv/exp_p0neg.csv"
    else
        rec 0 "negative-control" FAIL "could not build broken binary"
    fi
    rm -f "$WORK/csv/p0neg.csv"

    # 0e. directed Java-rejection shapes.  These belong to the sanity phase, not
    # to the soak: the branch has probability ~4e-9 and the sweep in P1 would
    # almost never execute it with any discriminating power, while a time-budgeted
    # run can spend its whole budget inside P1 and never reach P2 at all.
    for spec in "1 1 1" "17 17 5"; do
        set -- $spec
        local dtag="p0rej_${1}x${2}"
        if prep_expectation "$dtag" 1100064637205 5 7 5 7 "$1" "$2" roll; then
            for n in $(bins_for "$1"); do
                check_one 0 "${dtag}_${n}" "$WORK/bin/$n" "" 1100064637205 5 7 5 7 \
                    "$1" "$2" on "$EXP_CSV" "$EXP_THR" \
                    "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
            done
        else rec 0 "$dtag" FAIL "oracle failed"; fi
        rm -f "$EXP_CSV"
    done
}

# exhaustive (sizeX, sizeZ) sweep over the whole legal domain
phase1() {
    CFG_COUNT=0
    say "== P1: exhaustive rect sweep (sizeX 1..255 x sizeZ 1..32) =="
    local seeds=(114514 -12345 0) si sx sz
    for si in "${!seeds[@]}"; do
      local seed="${seeds[$si]}" xoff=$(( si * 1237 )) zoff=$(( si * 37 ))
      for (( sx = 1; sx <= 255; sx++ )); do
        for (( sz = 1; sz <= 32; sz++ )); do
            phase_stop && { say "P1: stopping at seed=$seed sx=$sx sz=$sz"; return; }
            new_cfg
            local X0 Z0 X1 Z1
            read -r X0 Z0 X1 Z1 <<<"$(pick_region "$sx" "$sz")"
            X0=$(( X0 + xoff )); X1=$(( X1 + xoff ))
            Z0=$(( Z0 + zoff )); Z1=$(( Z1 + zoff ))
            local tag="p1_s${si}_${sx}_${sz}"
            local sortm=on; (( sz % 2 == 1 )) && sortm=off   # cover both writers for free
            if ! prep_expectation "$tag" "$seed" $X0 $Z0 $X1 $Z1 "$sx" "$sz" roll; then
                rec 1 "$tag" FAIL "oracle failed"; continue
            fi
            local n
            for n in $(bins_for "$sx"); do
                check_one 1 "${tag}_${n}" "$WORK/bin/$n" "" "$seed" $X0 $Z0 $X1 $Z1 \
                    "$sx" "$sz" "$sortm" "$EXP_CSV" "$EXP_THR" \
                    "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
            done
            rm -f "$EXP_CSV"
        done
        (( sx % 32 == 0 )) && say "P1 progress: seed=$seed sizeX=$sx/255  pass=$NPASS fail=$NFAIL  left=$(left)min"
      done
    done
}

# second exhaustive pass on a TALLER region so that the Z sub-splitting path
# (zSub > 1) is exercised for every fused rect size
phase1b() {
    CFG_COUNT=0
    say "== P1b: exhaustive rect sweep on a tall region (zSub > 1) =="
    local sx sz
    for (( sx = 1; sx <= 32; sx++ )); do
        for (( sz = 1; sz <= 32; sz++ )); do
            phase_stop && { say "P1b: stopping at sx=$sx sz=$sz"; return; }
            new_cfg
            local X0=-512 Z0=-8192 X1=511 Z1=8191
            if (( sx * sz <= 4 )); then X0=-128; Z0=-4096; X1=127; Z1=4095; fi
            local tag="p1b_${sx}_${sz}"
            local sortm=on; (( sz % 2 == 0 )) && sortm=off
            if ! prep_expectation "$tag" "$SEED_ALT" $X0 $Z0 $X1 $Z1 "$sx" "$sz" roll; then
                rec 1 "$tag" FAIL "oracle failed"; continue
            fi
            local n
            for n in $(bins_for "$sx"); do
                check_one 1 "${tag}_${n}" "$WORK/bin/$n" "" "$SEED_ALT" $X0 $Z0 $X1 $Z1 \
                    "$sx" "$sz" "$sortm" "$EXP_CSV" "$EXP_THR" \
                    "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
            done
            rm -f "$EXP_CSV"
        done
    done
}

# geometry: widths/heights crossing every internal boundary, odd coordinates,
# both seeds, sort on/off, thresholds 0/1/max+1
phase2() {
    CFG_COUNT=0
    say "== P2: geometry / boundary sweep =="
    # note: |chunk coord| <= 2,000,000 is a hard limit of the program, so the
    # far-positive origin gets a 1-wide request instead of one that would be
    # rejected before it ever runs.
    local -a xs=(-2000000 -1999999 -4097 -1025 -513 -257 -1 0 1 255 511 1023 2047 1999999)
    local -a xw=(300 300 300 300 300 300 300 300 300 300 300 300 300 1)
    local -a widths=(1 2 17 255 256 257 511 512 513 1008 1009 1024 1025 2047 2048 2049 4096)
    local -a heights=(1 17 255 256 257 511 512 2047 2048 2049 4096 65535 65536 65537)
    local -a sxs=(1 2 3 15 16 17 31 32 33 34 64 255)
    local -a szs=(1 2 3 16 17 31 32)
    local i j tag seed sortm

    # (a) width sweep at fixed narrow height
    for i in "${widths[@]}"; do
        phase_stop && return
        new_cfg
        tag="p2w_${i}"
        if prep_expectation "$tag" "$SEED_MAIN" -600 -600 $(( -600 + i - 1 )) 599 17 17 roll; then
            for n in $(bins_for 17); do
                check_one 2 "${tag}_${n}" "$WORK/bin/$n" "" "$SEED_MAIN" -600 -600 \
                    $(( -600 + i - 1 )) 599 17 17 on "$EXP_CSV" "$EXP_THR" \
                    "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
            done
        else rec 2 "$tag" FAIL "oracle failed"; fi
        rm -f "$EXP_CSV"
    done

    # (b) height sweep at fixed narrow width (keeps the oracle affordable)
    for j in "${heights[@]}"; do
        phase_stop && return
        new_cfg
        tag="p2h_${j}"
        if prep_expectation "$tag" "$SEED_MAIN" -24 -300 23 $(( -300 + j - 1 )) 17 17 roll; then
            for n in $(bins_for 17); do
                check_one 2 "${tag}_${n}" "$WORK/bin/$n" "" "$SEED_MAIN" -24 -300 23 \
                    $(( -300 + j - 1 )) 17 17 on "$EXP_CSV" "$EXP_THR" \
                    "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
            done
        else rec 2 "$tag" FAIL "oracle failed"; fi
        rm -f "$EXP_CSV"
    done

    # (c) coordinates: extreme origins, zero crossing, both seeds.
    # The far-positive origin gets a 1-wide request because |chunk coord| <=
    # 2,000,000 is a hard limit of the program (it would reject the run otherwise).
    local xi
    for xi in "${!xs[@]}"; do
        i="${xs[$xi]}"
        local xw=300 zw=200
        (( i + xw > 2000000 )) && xw=1
        (( i + zw > 2000000 )) && zw=1
        (( i + zw < -2000000 )) && zw=1
        phase_stop && return
        new_cfg
        for seed in "$SEED_MAIN" "$SEED_ALT"; do
            tag="p2c_${i}_${seed}"
            if prep_expectation "$tag" "$seed" "$i" "$i" $(( i + xw )) $(( i + zw )) 17 32 roll; then
                for n in $(bins_for 17); do
                    check_one 2 "${tag}_${n}" "$WORK/bin/$n" "" "$seed" "$i" "$i" \
                        $(( i + xw )) $(( i + zw )) 17 32 on "$EXP_CSV" "$EXP_THR" \
                        "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
                done
            else rec 2 "$tag" FAIL "oracle failed"; fi
            rm -f "$EXP_CSV"
        done
    done

    # (d) rect-size cross product at a few representative regions
    for i in "${sxs[@]}"; do
        for j in "${szs[@]}"; do
            phase_stop && return
            new_cfg
            tag="p2r_${i}_${j}"
            if prep_expectation "$tag" "$SEED_MAIN" -300 -60 299 59 "$i" "$j" roll; then
                for sortm in on off; do
                    check_one 2 "${tag}_${sortm}" "$WORK/bin/default" "" "$SEED_MAIN" -300 -60 299 59 \
                        "$i" "$j" "$sortm" "$EXP_CSV" "$EXP_THR" \
                        "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
                done
            else rec 2 "$tag" FAIL "oracle failed"; fi
            rm -f "$EXP_CSV"
        done
    done

    # (d2) directed Java-rejection shapes (probability ~4e-9, see tools/oracle.c
    # --scan-reject): without these a kernel that skips the re-draw passes the
    # whole matrix, because no ordinary region contains a distinguishable cell.
    for spec in "1 1 1" "17 17 5"; do
        set -- $spec
        phase_stop && return
        new_cfg
        tag="p2rej_${1}x${2}"
        if prep_expectation "$tag" 1100064637205 5 7 5 7 "$1" "$2" roll; then
            for n in $(bins_for "$1"); do
                check_one 2 "${tag}_${n}" "$WORK/bin/$n" "" 1100064637205 5 7 5 7 \
                    "$1" "$2" on "$EXP_CSV" "$EXP_THR" \
                    "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
            done
        else rec 2 "$tag" FAIL "oracle failed"; fi
        rm -f "$EXP_CSV"
    done

    # (e) threshold edges: 0 (all candidates), 1, maxcount, maxcount+1
    local -a tr=(1 32 100 255)
    for i in "${tr[@]}"; do
        phase_stop && return
        new_cfg
        # maxcount probe
        local mc
        # NB: the program scans the PADDED region (X rounded up to a multiple of
        # 256, Z up to at least 256 rows), so the oracle must be given that region,
        # not the requested one -- otherwise the expectation is missing rows and
        # the comparison fails for a harness reason, not a program bug.
        local p2treg; p2treg="$(predict_region -100 -100 99 99)"
        read -r t_x0 t_x1 t_z0 t_z1 <<<"$p2treg"   # predict_region echoes x0 x1 z0 z1
        mc="$("$ORACLE" --seed "$SEED_MAIN" --x0 "$t_x0" --z0 "$t_z0" --x1 "$t_x1" --z1 "$t_z1" \
                --sx "$i" --sz 17 --mode auto --target 1 --out "$WORK/csv/p2t_probe.csv" 2>/dev/null \
                | sed -n 's/.*maxcount=\([0-9]*\).*/\1/p')"
        rm -f "$WORK/csv/p2t_probe.csv"
        [[ -z "$mc" ]] && { rec 2 "p2t_$i" FAIL "maxcount probe failed"; continue; }
        for thr in 0 1 "$mc" $(( mc + 1 )); do
            tag="p2t_${i}_${thr}"
            local exp="$WORK/csv/exp_${tag}.csv"
            if ! "$ORACLE" --seed "$SEED_MAIN" --x0 "$t_x0" --z0 "$t_z0" --x1 "$t_x1" --z1 "$t_z1" \
                    --sx "$i" --sz 17 \
                    --threshold "$thr" --method roll --out "$exp" > "$WORK/log/oracle_${tag}.log" 2>&1; then
                rec 2 "$tag" FAIL "oracle failed"; continue
            fi
            for sortm in on off; do
                check_one 2 "${tag}_${sortm}" "$WORK/bin/default" "" "$SEED_MAIN" -100 -100 99 99 \
                    "$i" 17 "$sortm" "$exp" "$thr" "$t_x0" "$t_x1" "$t_z0" "$t_z1" || true
            done
            rm -f "$exp"
        done
    done
}

# knobs: device result buffer refills and sort-run spilling
phase3() {
    CFG_COUNT=0
    say "== P3: cap / spill / sort-mode coverage =="
    local -a caps=("" "4096" "65536" "4194304")
    local -a blocks=("" "1" "2" "3" "8")
    local -a cfgs=("-300 -300 300 300 17 17" "-1500 -1500 1500 1500 32 32" \
                   "-40 -20000 40 20000 17 17" "-120 -120 120 120 40 17")
    local c b tag
    for c in "${cfgs[@]}"; do
        set -- $c
        local x0=$1 z0=$2 x1=$3 z1=$4 sx=$5 sz=$6
        tag="p3_${x0}_${sx}_${sz}"
        phase_stop && return
        new_cfg
        if ! prep_expectation "$tag" "$SEED_MAIN" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" roll; then
            rec 3 "$tag" FAIL "oracle failed"; continue
        fi
        for b in "${blocks[@]}"; do
            for c2 in "${caps[@]}"; do
                [[ -z "$b" && -z "$c2" ]] && continue
                local env=""
                [[ -n "$b" ]] && env="SLIME_SORT_BLOCKS=$b"
                [[ -n "$c2" ]] && env="$env SLIME_CAP_SLOTS=$c2"
                local lbl="b${b:-x}_c${c2:-x}"
                for sortm in on off; do
                    check_one 3 "${tag}_${lbl}_${sortm}" "$WORK/bin/default" "$env" "$SEED_MAIN" \
                        "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$sortm" "$EXP_CSV" "$EXP_THR" \
                        "$PRED_X0" "$PRED_X1" "$PRED_Z0" "$PRED_Z1" || true
                done
            done
        done
        rm -f "$EXP_CSV"
    done
}

# huge regions: full-row invariants + random single-candidate probes
phase4() {
    CFG_COUNT=0
    say "== P4: huge regions (invariants + random probes) =="
    local -a cfgs=(
        "114514 -1875000 -200000 1875000 200000 17 17 55"
        "114514 -200000 -200000 200000 200000 17 17 55"
        "114514 -1000000 -1000000 1000000 1000000 32 32 300"
        "-12345 -500000 -500000 500000 500000 17 32 200"
        "114514 -1999999 -1999999 -1990000 -1990000 17 17 3"
        "114514 -2 -2 2 2 1 1 0"
    )
    local spec
    for spec in "${cfgs[@]}"; do
        phase_stop && return
        new_cfg
        set -- $spec
        local seed=$1 x0=$2 z0=$3 x1=$4 z1=$5 sx=$6 sz=$7 thr=$8
        local tag="p4_${x0}_${z0}_${sx}_${sz}_${thr}"
        local csv="$WORK/csv/${tag}.csv" log="$WORK/log/${tag}.log"
        local pred; pred="$(predict_region "$x0" "$z0" "$x1" "$z1")"
        read -r px0 px1 pz0 pz1 <<<"$pred"
        NRUN=$((NRUN+1))
        if ! run_prog "$WORK/bin/default" "" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$thr" \
                "$csv" on "$log"; then
            rec 4 "$tag" FAIL "run failed (see $log)"; continue
        fi
        local sr; sr="$(scan_region "$log")"; read -r asx0 asx1 asz0 asz1 <<<"$sr"
        local minx=$(( asx0 * 16 )) maxx=$(( (asx1 - sx + 1) * 16 ))
        local minz=$(( asz0 * 16 )) maxz=$(( (asz1 - sz + 1) * 16 ))
        local inv; inv="$(csv_invariants "$csv" "$minx" "$maxx" "$minz" "$maxz" "$thr" 1)"
        local rows; rows="$(sed -n 's/^rows=\([0-9]*\).*/\1/p' <<<"$inv")"
        local rest; rest="$(sed -E 's/^rows=[0-9]+ //' <<<"$inv")"
        local tv; tv="$(total_valid "$log")"
        case "$rest" in
            "badnf=0 bad16=0 badcnt=0 badrange=0 unsorted=0 dup=0 dupchecked=0"|\
            "badnf=0 bad16=0 badcnt=0 badrange=0 unsorted=0 dup=0 dupchecked=1") ;;
            *) rec 4 "$tag" FAIL "invariants: $rest (Total valid=$tv rows=$rows)"
               rm -f "$csv"; continue ;;
        esac
        if [[ "$tv" != "$rows" ]]; then
            rec 4 "$tag" FAIL "'Total valid' ($tv) != csv rows ($rows)"
            rm -f "$csv"; continue
        fi
        # random probes: candidates the oracle says ARE hits and candidates it
        # says are not -- catches both spurious rows and MISSING rows on regions
        # far too large to materialise.  One awk pass over the CSV for all of them.
        local plist="$WORK/csv/${tag}.probes"
        : > "$plist"
        # Number of candidate top-left positions, NOT (asx1 - sx + 2): with a
        # negative origin that expression goes negative, bash's % then returns a
        # negative value and every probe lands outside the scanned region.
        local ncol=$(( asx1 - asx0 + 2 - sx ))
        local nrow=$(( asz1 - asz0 + 2 - sz ))
        (( ncol < 1 )) && ncol=1
        (( nrow < 1 )) && nrow=1
        local seedr=$(( RANDOM * 32768 + RANDOM ))
        local k
        for (( k = 0; k < 300; k++ )); do
            local rx=$(( asx0 + (seedr + k * 2654435761) % ncol ))
            local rz=$(( asz0 + (seedr / 7 + k * 40503) % nrow ))
            local pc
            pc="$("$ORACLE" --probe --seed "$seed" --cx "$rx" --cz "$rz" --sx "$sx" --sz "$sz" \
                    | sed -n 's/.*count=\([0-9]*\).*/\1/p')"
            [[ -n "$pc" ]] && printf '%d,%d,%d\n' $(( rx * 16 )) $(( rz * 16 )) "$pc" >> "$plist"
        done
        local pres
        pres="$(awk -F, -v thr="$thr" '
            NR == FNR { want[$1 "," $2] = $3; next }
            FNR == 1 { next }
            { key = $1 "," $2; if (key in want) got[key] = 1 }
            END {
                for (k in want) {
                    n++
                    e = (want[k] + 0 >= thr) ? 1 : 0
                    a = (k in got) ? 1 : 0
                    if (e != a) bad++
                }
                printf "probes=%d bad=%d", n + 0, bad + 0
            }' "$plist" "$csv")"
        rm -f "$plist"
        local bad; bad="$(sed -n 's/.*bad=\([0-9]*\).*/\1/p' <<<"$pres")"
        local got_in; got_in="$(sed -n 's/probes=\([0-9]*\).*/\1/p' <<<"$pres")"
        local want_in=0 got_out=0
        if [[ "$bad" == "0" && "$got_in" -gt 0 ]]; then
            rec 4 "$tag" PASS "rows=$rows probes=$got_in all correct"
        else
            rec 4 "$tag" FAIL "probe mismatches=$bad of $got_in (rows=$rows)"
        fi
        rm -f "$csv"
    done
}

# determinism + monotonicity + sort-mode equivalence on medium regions
phase5() {
    CFG_COUNT=0
    say "== P5: determinism, monotonicity, sort-mode equivalence =="
    local -a cfgs=(
        "114514 -2000 -2000 2000 2000 17 17"
        "114514 -600 -20000 600 20000 17 17"
        "-12345 -3000 -3000 3000 3000 32 32"
        "114514 -800 -800 800 800 40 17"
    )
    local spec
    for spec in "${cfgs[@]}"; do
        phase_stop && return
        new_cfg
        set -- $spec
        local seed=$1 x0=$2 z0=$3 x1=$4 z1=$5 sx=$6 sz=$7
        local tag="p5_${x0}_${sx}_${sz}"
        # get a threshold with a decent number of hits
        if ! prep_expectation "$tag" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" roll; then
            rec 5 "$tag" FAIL "oracle failed"; continue
        fi
        local a="$WORK/csv/${tag}_a.csv" b="$WORK/csv/${tag}_b.csv" c="$WORK/csv/${tag}_c.csv"
        run_prog "$WORK/bin/default" "" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$EXP_THR" "$a" on  "$WORK/log/${tag}_a.log"
        run_prog "$WORK/bin/default" "" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$EXP_THR" "$b" on  "$WORK/log/${tag}_b.log"
        run_prog "$WORK/bin/default" "" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" "$EXP_THR" "$c" off "$WORK/log/${tag}_c.log"
        local ok=1 why=""
        cmp -s "$a" "$b" || { ok=0; why="rerun differs"; }
        cmp -s <(canon "$a") <(canon "$c") || { ok=0; why="sort=on vs off differ"; }
        cmp -s <(canon "$a") <(canon "$EXP_CSV") || { ok=0; why="oracle mismatch"; }
        # monotonicity: thr+1 must be a subset of thr
        local m="$WORK/csv/${tag}_m.csv"
        if [[ $ok -eq 1 ]]; then
            run_prog "$WORK/bin/default" "" "$seed" "$x0" "$z0" "$x1" "$z1" "$sx" "$sz" \
                $(( EXP_THR + 1 )) "$m" on "$WORK/log/${tag}_m.log"
            if ! awk -F, 'NR==FNR { if (FNR>1) seen[$1","$2]=1; next }
                          FNR>1 { if (!(($1","$2) in seen)) bad++ }
                          END { exit (bad?1:0) }' "$a" "$m"; then
                ok=0; why="thr+1 not a subset of thr"
            fi
        fi
        if [[ $ok -eq 1 ]]; then rec 5 "$tag" PASS "thr=$EXP_THR hits=$EXP_HITS deterministic+monotone"
        else rec 5 "$tag" FAIL "$why"; fi
        rm -f "$a" "$b" "$c" "$m" "$EXP_CSV"
    done
}

# ================================================================ main

say "slime-calculator full correctness check"
say "binary=$BIN  workdir=$WORK  budget=${HOURS}h  target_hits=$TARGET"
say "phases: $PHASES"
echo "build: $(git rev-parse --short HEAD 2>/dev/null || echo unknown) $(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    > "$WORK/summary.txt"

ORACLE=build/oracle
if ! build_all; then say "build failed, aborting"; exit 2; fi
have_phase() { [[ " $PHASES " == *" $1 "* ]]; }

# Phase order is deliberate: the cheap, high-value phases run BEFORE the
# exhaustive sweep, so a time-budgeted run always covers them.  (Measured the
# hard way: a 2 h budget was entirely consumed by P1, leaving P2-P5 -- including
# the directed rejection shapes -- never executed.)
have_phase 0 && phase0
have_phase 2 && phase2
have_phase 3 && phase3
have_phase 4 && phase4
have_phase 5 && phase5
have_phase 1 && phase1
have_phase 1 && phase1b

echo
echo "================ SUMMARY ================"
printf '%-8s %8s %8s %8s\n' phase total pass fail
{
printf '%-8s %8s %8s %8s\n' phase total pass fail
} >> "$WORK/summary.txt"
for p in 0 1 2 3 4 5; do
    t=${PH_TOTAL[$p]:-0}; [[ $t -eq 0 ]] && continue
    printf '%-8s %8d %8d %8d\n' "$p" "$t" "${PH_PASS[$p]:-0}" "${PH_FAIL[$p]:-0}"
    printf '%-8s %8d %8d %8d\n' "$p" "$t" "${PH_PASS[$p]:-0}" "${PH_FAIL[$p]:-0}" >> "$WORK/summary.txt"
done
printf '%-8s %8d %8d %8d\n' TOTAL "$((NPASS+NFAIL+NSKIP))" "$NPASS" "$NFAIL"
printf '%-8s %8d %8d %8d\n' TOTAL "$((NPASS+NFAIL+NSKIP))" "$NPASS" "$NFAIL" >> "$WORK/summary.txt"
echo "skipped (resume): $NSKIP   program runs: $NRUN"
echo "elapsed: $(( ($(date +%s) - T0) / 60 )) min   budget: ${HOURS}h   configs: $CFG_COUNT   program runs: $NRUN"
if [[ $NFAIL -eq 0 ]]; then
    echo "RESULT: PASS"
    echo "RESULT: PASS" >> "$WORK/summary.txt"
    exit 0
else
    echo "RESULT: FAIL ($NFAIL failures; see $FAILURES)"
    echo "RESULT: FAIL ($NFAIL failures)" >> "$WORK/summary.txt"
    head -20 "$FAILURES"
    exit 1
fi
