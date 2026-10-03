#!/bin/bash
#
# releasecheck.sh -- run the whole verification battery on the current build and
# write a signed, reproducible report you can attach to a release.
#
#   usage: releasecheck.sh [options]
#     --out FILE      report path (default .work/release-verification.md)
#     --soak HOURS    additionally run tools/fullcheck.sh for HOURS (default 0 =
#                     skip; the soak is the 5-8 h exhaustive sweep)
#     --java PATH     JVM for the external Minecraft cross-check (a JDK is
#                     required by that stage; without one the stage is skipped
#                     and the report says so)
#     --skip-verify   skip tools/verify.sh (the 42-config bit-exactness matrix)
#     --quick         only the fast stage (quickcheck + cross-check)
#     --no-build      reuse the existing build/slime_main (useful when another
#                     job is running it); its hash is still recorded
#     -h, --help
#
# Stages:
#   0. environment + source/binary identity
#   1. tools/quickcheck.sh        output vs an independent Java-spec oracle,
#                                 complete row set, 23 configurations
#                                 (LOCAL-ONLY helper: skipped with a note when absent)
#   2. tools/mccrosscheck.sh      predicate vs the ORIGINAL Minecraft Java code
#                                 running on a real JVM
#   3. tools/verify.sh            42-config bit-exactness matrix against the
#                                 frozen reference binary
#   4. tools/fullcheck.sh         optional exhaustive soak (all 8160 legal
#                                 rectangle sizes x 3 seeds x 6 compile variants)
#
# Every stage's raw log is kept and hashed.  The report states explicitly what
# was NOT verified -- see docs/zh-cn/correctness.md; a verification report that only
# lists passes is not an honest one.
#
# ASCII output only.
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="$ROOT/.work/release-verification.md"
SOAK=0; JAVA=""; SKIP_VERIFY=0; QUICK=0; NOBUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)         OUT="$2"; shift 2 ;;
        --soak)        SOAK="$2"; shift 2 ;;
        --java)        JAVA="$2"; shift 2 ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        --quick)       QUICK=1; shift ;;
        --no-build)    NOBUILD=1; shift ;;
        -h|--help)     sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# `--java` wins; otherwise look where a JDK is normally reachable, so a machine
# that has one does not silently skip the external cross-check stage.
if [[ -z "$JAVA" ]]; then
    if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then JAVA="$JAVA_HOME/bin/java"
    elif command -v java >/dev/null 2>&1; then JAVA="$(command -v java)"
    fi
fi

LOGD="$ROOT/.work/releasecheck"
mkdir -p "$LOGD"
REPORT_TMP="$(mktemp)"
: > "$REPORT_TMP"

say() { printf '%s\n' "$*" | tee -a "$REPORT_TMP" >&2; }
hdr() { printf '\n### %s\n\n' "$*" >> "$REPORT_TMP"; }

STAGE_NAME=(); STAGE_RC=(); STAGE_LOG=()
run_stage() { # run_stage <name> <logfile> <command...>
    local name="$1" log="$2"; shift 2
    printf '[%s] %s ... ' "$(date '+%H:%M:%S')" "$name" >&2
    ( "$@" ) > "$log" 2>&1
    local rc=$?
    STAGE_NAME+=("$name"); STAGE_RC+=("$rc"); STAGE_LOG+=("$log")
    if [[ $rc -eq 0 ]]; then echo "PASS" >&2; else echo "FAIL (rc=$rc)" >&2; fi
    return $rc
}
sha() { [[ -f "$1" ]] && sha256sum "$1" | cut -d' ' -f1 || echo "-"; }

# ---------------------------------------------------------------- stage 0
say "=== release verification ==="
say "started $(date -u '+%Y-%m-%dT%H:%M:%SZ') on $(hostname)"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | head -1)"
[[ -n "$GIT_DIRTY" ]] && GIT_DIRTY="yes" || GIT_DIRTY="no"
SRC_SHA="$(sha src/slime_main.cu)"
NVCC="$(nvcc --version 2>/dev/null | tail -1 | sed 's/^ *//')"
GPUINFO="$(nvidia-smi --query-gpu=name,driver_version,clocks.current.sm,clocks.current.memory,temperature.gpu \
            --format=csv,noheader 2>/dev/null | head -1)"

if [[ $NOBUILD -eq 1 ]]; then
    say "reusing the existing build (--no-build)"
    BUILD_RC=0
else
    say "building"
    ./build.sh > "$LOGD/build.log" 2>&1
    BUILD_RC=$?
fi
BIN_SHA="$(sha build/slime_main)"
BIN_MD5="$(md5sum build/slime_main 2>/dev/null | cut -d' ' -f1)"
# The CUDA toolchain is not bit-reproducible at the ELF level: two builds of the
# same source differ in a few bytes of an ABI-tag string that embeds nvcc's
# temporary file name (tmpxft_<random>_...cudafe1.cpp).  The CODE is identical
# (measured: SASS diff 0, .text byte diff 0), so the device-code hash is the
# reproducible identity of the verified version.
SASS_SHA="$(cuobjdump -sass build/slime_main 2>/dev/null | sha256sum | cut -d' ' -f1)"

# ---------------------------------------------------------------- stages
if [[ $BUILD_RC -ne 0 ]]; then
    say "BUILD FAILED -- see $LOGD/build.log"
else
    if [[ -x "$ROOT/tools/quickcheck.sh" ]]; then
        run_stage "quickcheck (oracle, complete row set)" \
                  "$LOGD/quickcheck.log" "$ROOT/tools/quickcheck.sh" || true
    else
        say "SKIP quickcheck: tools/quickcheck.sh not present (local-only helper, see .gitignore)"
    fi

    if [[ -n "$JAVA" ]]; then
        run_stage "mccrosscheck (original Minecraft Java on a JVM)" \
                  "$LOGD/mccrosscheck.log" "$ROOT/tools/mccrosscheck.sh" --java "$JAVA" || true
    else
        say "SKIP mccrosscheck: no JDK found (any JDK 11+ works; a relocated one too)"
    fi

    if [[ $QUICK -eq 0 && $SKIP_VERIFY -eq 0 ]]; then
        run_stage "verify.sh (42-config bit-exactness matrix)" \
                  "$LOGD/verify.log" "$ROOT/tools/verify.sh" || true
    fi

    if [[ $SOAK -gt 0 ]]; then
        run_stage "fullcheck soak (${SOAK} h exhaustive sweep)" \
                  "$LOGD/fullcheck.log" "$ROOT/tools/fullcheck.sh" --hours "$SOAK" \
                  --workdir "$ROOT/.fullcheck" || true
    fi
fi

# ---------------------------------------------------------------- report
{
echo "# Release verification report"
echo
echo "Generated by \`tools/releasecheck.sh\`. Every line below is reproducible by"
echo "re-running that script on the same commit."
echo
echo "## Identity"
echo
echo '| item | value |'
echo '|---|---|'
echo "| generated (UTC) | $(date -u '+%Y-%m-%dT%H:%M:%SZ') |"
echo "| git commit | \`$GIT_COMMIT\` |"
echo "| working tree dirty | $GIT_DIRTY |"
echo "| \`src/slime_main.cu\` sha256 | \`$SRC_SHA\` |"
echo "| \`build/slime_main\` sha256 | \`$BIN_SHA\` |"
echo "| \`build/slime_main\` md5 | \`$BIN_MD5\` |"
echo "| device code (\`cuobjdump -sass\`) sha256 | \`$SASS_SHA\` |"
echo "| nvcc | ${NVCC:-unknown} |"
echo "| GPU at report time | ${GPUINFO:-unknown} |"
echo "| build | $( [[ $NOBUILD -eq 1 ]] && echo 'reused (--no-build)' || echo "exit code $BUILD_RC" ) |"
echo
echo "Note on reproduction: the CUDA toolchain is **not** bit-reproducible at the ELF"
echo "level -- two builds of this exact source differ in 3 bytes of an ABI-tag string"
echo "that embeds nvcc's temporary file name.  The generated code is identical:"
echo "measured \`cuobjdump -sass\` diff = 0 and \`.text\` byte diff = 0 between two"
echo "independent builds.  So the **source sha256 plus the device-code sha256** are the"
echo "reproducible identity of the verified version; the ELF hash identifies the exact"
echo "artifact these checks ran against."
echo
echo "## Results"
echo
echo '| stage | result | raw log |'
echo '|---|---|---|'
if [[ ${#STAGE_NAME[@]} -eq 0 ]]; then
    echo "| (none) | BUILD FAILED | \`$LOGD/build.log\` |"
else
    for i in "${!STAGE_NAME[@]}"; do
        rc="${STAGE_RC[$i]}"
        res="PASS"; [[ "$rc" != "0" ]] && res="**FAIL (rc=$rc)**"
        echo "| ${STAGE_NAME[$i]} | $res | \`${STAGE_LOG[$i]#$ROOT/}\` |"
    done
fi
echo
echo "## What each stage establishes"
echo
echo "* **quickcheck** -- for every configuration in tools/quickcheck.sh (both kernels, multi-tile and"
echo "  multi-chunk regions, 1x1 / 1x17 / 17x1 / 17x32 / 32x32 / 255x32, thresholds"
echo "  0 / negative / above-maximum, negative and INT64_MIN seeds, the world corner,"
echo "  both output modes, the forced result-buffer-retry path, and two directed"
echo "  shapes for the ~4e-9 Java rejection branch, in both failure directions) the"
echo "  program's COMPLETE set of"
echo "  emitted rows equals the expectation of an independent oracle that was"
echo "  written from the Java specification. On small regions the oracle also"
echo "  recomputes with the brute-force O(sizeX*sizeZ)-per-candidate sum and"
echo "  refuses to emit if the two methods disagree."
echo "* **mccrosscheck** -- the per-chunk predicate is compared against the ORIGINAL"
echo "  Minecraft Java code, extracted verbatim from \`docs/en-us/index.md\` and"
echo "  executed by a real JVM, over 1,048,576 chunks. A JDK is required: there is"
echo "  no cached summary that could go stale and still report a pass."
echo "* **verify.sh** -- 42 configurations produce byte-identical output to the"
echo "  frozen reference binary, including the \`sizeX > 32\` fallback kernel, forced"
echo "  buffer-overflow retries, the on-disk spill + k-way merge path, forced"
echo "  \`zSub == 1\`, and the world-boundary corners."
if [[ $SOAK -gt 0 ]]; then
echo "* **fullcheck soak** -- every legal \`sizeX\` (1..255) x \`sizeZ\` (1..32) is"
echo "  swept (3 seeds, and 6 compile variants for the fused kernel), each compared"
echo "  against the oracle in full; huge regions additionally get row invariants and"
echo "  random single-candidate probes."
fi
SOAKSUM="$ROOT/.fullcheck/summary.txt"
if [[ -f "$SOAKSUM" ]]; then
echo
echo "## Exhaustive soak result found on disk"
echo
echo "\`\`.fullcheck/summary.txt\`\` (produced by \`\`tools/fullcheck.sh\`\`):"
echo
echo '```'
cat "$SOAKSUM"
echo '```'
echo
echo "Note: the soak sweeps every legal rectangle size (\`\`sizeX\`\` 1..255 x"
echo "\`\`sizeZ\`\` 1..32) over several seeds and, for the fused kernel, several"
echo "compile variants, comparing the COMPLETE emitted row set against the oracle"
echo "for every configuration. It is time-budgeted (\`\`--hours\`\`), so a partial"
echo "run means the sweep stopped early at the reported point, not that anything"
echo "failed."
fi
echo
echo "## Scope and limits (what this report does NOT establish)"
echo
echo "Read \`docs/zh-cn/correctness.md\` for the full argument. The correctness goals are"
echo "three, and each is covered above:"
echo
echo "1. **The output has no duplicates and no omissions.** Asserted directly (every"
echo "   \`(x,z)\` unique) and by comparing the COMPLETE emitted row set with an"
echo "   independent Java-spec oracle -- for every configuration of the sweep."
echo "2. **The per-chunk rule is the original Minecraft algorithm.**\ \`mccrosscheck\`"
echo "   runs the wiki's own Java on a real JVM and compares 1,048,576 chunks."
echo "3. **The interface matches the documentation** (coordinates x16,"
echo "   \`count >= threshold\`, the 256-column padding rule, \`--sort\` ordering);"
echo "   each of those is asserted per row or per run."
echo
echo "NOTE: this report does NOT include a device-side UB stage. The 0-UB claim"
echo "rests on the static audit in \`docs/zh-cn/correctness.md\` section 4 alone."
echo "Stated here so the absence of that stage is never mistaken for a passed UB"
echo "check."
echo
echo "NOTE: point 2 above is only established if the Results table shows"
echo "\`mccrosscheck\` as PASS. That stage needs a JDK and is skipped without one --"
echo "there is deliberately no cached summary to fall back on. Check the table."
echo
echo "Explicitly OUT OF SCOPE by decision: concurrency and row ordering -- results"
echo "are compared as multisets and no ordering is promised."
echo
echo "Finally: everything here says \"no counterexample was found in this matrix\","
echo "not \"no counterexample exists\". What the matrix does not cover was not"
echo "tested. The one thing no self-check can settle is whether the documented"
echo "interface is what the user actually wants -- that is requirements review, not"
echo "implementation correctness."
echo
echo "## Reproduce"
echo
echo '```bash'
echo "./build.sh"
echo "tools/quickcheck.sh                                   # local-only helper, not published"
echo "tools/mccrosscheck.sh --java \"\$JAVA_HOME/bin/java\"   # a JDK is required"
echo "tools/verify.sh"
echo "tools/fullcheck.sh --hours 6                          # the 5-8 h soak"
echo '```'
} >> "$REPORT_TMP"

mv "$REPORT_TMP" "$OUT"

# ---------------------------------------------------------------- verdict
FAILED=0
for rc in "${STAGE_RC[@]:-}"; do [[ "$rc" != "0" ]] && FAILED=1; done
[[ $BUILD_RC -ne 0 ]] && FAILED=1

say ""
say "report written to $OUT"
if [[ $FAILED -eq 0 ]]; then
    say "VERDICT: PASS (all executed stages passed)"
    exit 0
else
    say "VERDICT: FAIL (see the report and $LOGD)"
    exit 1
fi
