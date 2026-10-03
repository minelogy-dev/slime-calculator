#!/bin/bash
#
# mccrosscheck.sh -- validate the per-chunk predicate against the ORIGINAL
# Minecraft logic, executed by a real JVM.
#
# Why this exists: every other check in this repo compares the program against
# tools/oracle.c, which is an independent implementation but still *this repo's*
# reading of the rule.  The one thing self-comparison cannot catch is a shared
# misunderstanding of the specification.  docs/en-us/index.md quotes the
# Minecraft wiki's own Java code, so this script
#
#   1. extracts that code block VERBATIM from the markdown (and proves the
#      extraction is byte-identical to the block, so nothing is retyped),
#   2. compiles and runs it on a real JDK,
#   3. runs slime_main with a 1x1 rectangle so its output is exactly the set of
#      slime chunks, and
#   4. compares the two sets.
#
#   usage: mccrosscheck.sh [options]
#     --java PATH     java binary (default: $JAVA_HOME/bin/java, else `java`)
#     --javac PATH    javac binary (default: alongside --java)
#     --x0/--z0/--x1/--z1 N   chunk region to compare (default -512 -512 511 511;
#                     keep the width/height multiples of 256 so the program's
#                     padding is a no-op and the two regions coincide exactly)
#     --doc FILE      markdown to extract from (default docs/en-us/index.md)
#     --workdir DIR   scratch directory (default .work/mccrosscheck)
#     -h, --help
#
# Exit code: 0 = identical, 1 = mismatch, 2 = setup problem.
#
# A JDK is REQUIRED.  This script used to fall back to a recorded "golden
# summary" (chunk count + sha256) when no JVM was found, but such a file cannot
# check its own currency: the summary is a function of seed, region and the
# Java block, so once the spec drifts the cached numbers keep reporting PASS.
# That is a silent failure in the one layer whose whole job is to catch a
# shared misreading of the spec, so the fallback was removed rather than
# maintained.  Install any JDK 11+ (a relocated one works too), or pass
# --java / set JAVA_HOME.
#
# ASCII output only.
#
set -uo pipefail

JAVA=""; JAVAC=""; DOC=""; BIN=""
# Default seed chosen deliberately: its chunk (5,7) takes the Java rejection
# branch AND the re-draw changes the answer, so this region is the only kind of
# input that can tell a correct build from one that skips the re-draw.  Seed
# 114514 has ZERO such cells in the same region, which was measured -- using it
# would make this cross-check blind to that whole code path.  Re-derive with:
#   ./build/oracle --scan-reject --seed S --x0 -512 --z0 -512 --x1 511 --z1 511
SEED=1100064637205
X0=-512; Z0=-512; X1=511; Z1=511
WORK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --java)    JAVA="$2"; shift 2 ;;
        --javac)   JAVAC="$2"; shift 2 ;;
        --doc)     DOC="$2"; shift 2 ;;
        --bin)     BIN="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        --workdir) WORK="$2"; shift 2 ;;
        --x0)      X0="$2"; shift 2 ;;
        --z0)      Z0="$2"; shift 2 ;;
        --x1)      X1="$2"; shift 2 ;;
        --z1)      Z1="$2"; shift 2 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
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
ROOT="${SLIME_ROOT:-$(find_root)}" || { echo "cannot find project root; set SLIME_ROOT" >&2; exit 2; }
[[ -n "$DOC" ]]    || DOC="$ROOT/docs/en-us/index.md"
[[ -n "$WORK" ]]   || WORK="$ROOT/.work/mccrosscheck"
[[ -n "$BIN" ]] || BIN="$ROOT/build/slime_main"
mkdir -p "$WORK"

# ---------------------------------------------------------------- JDK lookup
if [[ -z "$JAVA" ]]; then
    if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then JAVA="$JAVA_HOME/bin/java"
    elif command -v java >/dev/null 2>&1; then JAVA="$(command -v java)"
    fi
fi
if [[ -z "$JAVAC" && -n "$JAVA" ]]; then
    JAVAC="$(dirname "$JAVA")/javac"
    [[ -x "$JAVAC" ]] || JAVAC=""
fi
HAVE_JDK=0
[[ -n "$JAVA" && -x "$JAVA" && -n "$JAVAC" && -x "$JAVAC" ]] && HAVE_JDK=1
if [[ $HAVE_JDK -eq 0 ]]; then
    echo "no JDK found -- this cross-check cannot be faked by a recorded summary." >&2
    echo "install a JDK (any 11+; a relocated one works too), or pass --java / set JAVA_HOME" >&2
    exit 2
fi
# Support a JDK that was merely unpacked somewhere (e.g. `dpkg-deb -x` of
# openjdk-*-jdk-headless + -jre-headless): its launchers need libjli/libjvm,
# which live outside bin/.
_jl="$(cd "$(dirname "$JAVA")/../lib" 2>/dev/null && pwd)"
if [[ -n "$_jl" && -e "$_jl/libjli.so" ]]; then
    export LD_LIBRARY_PATH="$_jl:$_jl/server${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "==================================================================="
echo " cross-check against the original Minecraft logic"
echo "   doc      : $DOC"
echo "   region   : chunks [$X0,$X1] x [$Z0,$Z1]"
echo "   binary   : $BIN"
echo "   jvm      : $JAVA"
echo "==================================================================="

# ---------------------------------------------------------------- 1. extract
# The Java code block is taken VERBATIM; the file must keep the class name.
JAVA_SRC="$WORK/CheckSlimeChunk.java"
python3 - "$DOC" "$JAVA_SRC" <<'PY'
import re, sys
doc, out = sys.argv[1], sys.argv[2]
text = open(doc, encoding="utf-8").read()
m = re.search(r"```java\n(.*?)```", text, re.S)
if not m:
    print("no ```java block found in", doc, file=sys.stderr); sys.exit(3)
code = m.group(1)
if "class CheckSlimeChunk" not in code:
    print("the java block does not define CheckSlimeChunk", file=sys.stderr); sys.exit(3)
open(out, "w", encoding="utf-8").write(code)
print("   extracted %d lines from the markdown, verbatim" % code.count("\n"))
PY
rc=$?
[[ $rc -ne 0 ]] && exit 2

# ---------------------------------------------------------------- 2. driver
cat > "$WORK/CrossCheck.java" <<'JAVA'
/*
 * Driver for the verbatim CheckSlimeChunk class extracted from docs/.
 * It prints one "chunkX,chunkZ" line per slime chunk in the requested region.
 */
public class CrossCheck {
    public static void main(String[] args) {
        long seed = Long.parseLong(args[0]);
        int x0 = Integer.parseInt(args[1]), z0 = Integer.parseInt(args[2]);
        int x1 = Integer.parseInt(args[3]), z1 = Integer.parseInt(args[4]);
        StringBuilder sb = new StringBuilder(1 << 20);
        long n = 0;
        for (int z = z0; z <= z1; z++) {
            for (int x = x0; x <= x1; x++) {
                if (CheckSlimeChunk.isSlimeChunk(seed, x, z)) {
                    sb.append(x).append(',').append(z).append('\n');
                    n++;
                    if (sb.length() > (1 << 20)) { System.out.print(sb); sb.setLength(0); }
                }
            }
        }
        System.out.print(sb);
        System.err.println("java: " + n + " slime chunks in the region");
    }
}
JAVA

GOT="$WORK/java_set.txt"

echo "[1/3] compiling and running the original Java logic"
( cd "$WORK" && "$JAVAC" -d . CheckSlimeChunk.java CrossCheck.java ) 2>&1 | head -5
"$JAVA" -cp "$WORK" CrossCheck "$SEED" "$X0" "$Z0" "$X1" "$Z1" > "$GOT" 2> "$WORK/java.err"
rc=$?
if [[ $rc -ne 0 ]]; then echo "JVM run failed:"; head -3 "$WORK/java.err"; exit 2; fi
cat "$WORK/java.err"
LC_ALL=C sort "$GOT" -o "$GOT"
NJ=$(wc -l < "$GOT")
HJ=$(sha256sum < "$GOT" | cut -d' ' -f1)
echo "      $NJ chunks from the JVM, sha256 ${HJ:0:16}"

# ---------------------------------------------------------------- 2. program
echo "[2/3] running the binary with a 1x1 rectangle (its output IS the slime-chunk set)"
[[ -x "$BIN" ]] || { echo "binary missing: $BIN (run ./build.sh)" >&2; exit 2; }
CSV="$WORK/prog.csv"
rm -f "$CSV"
"$BIN" "$SEED" "$X0" "$Z0" "$X1" "$Z1" 1 1 1 "$CSV" --sort=on > "$WORK/prog.log" 2>&1
rc=$?
[[ $rc -ne 0 ]] && { echo "run failed:"; tail -5 "$WORK/prog.log"; exit 2; }
grep -m1 '^Global range:' "$WORK/prog.log"
# NB: both sides must be sorted in the SAME collation -- under a UTF-8 locale
# `sort` ignores ',' at the primary level, which reorders the same multiset and
# makes cmp/comm report bogus differences.
tail -n +2 "$CSV" | awk -F, '{ if ($3+0 >= 1) printf "%d,%d\n", $1/16, $2/16 }' \
    | LC_ALL=C sort > "$WORK/prog_set.txt"
NP=$(wc -l < "$WORK/prog_set.txt")
HP=$(sha256sum < "$WORK/prog_set.txt" | cut -d' ' -f1)
echo "      $NP chunks from $(basename "$BIN"), sha256 ${HP:0:16}"

# ---------------------------------------------------------------- 3. compare
echo "[3/3] comparing"
if cmp -s "$GOT" "$WORK/prog_set.txt"; then
    echo "      IDENTICAL to the JVM's set (${NJ} chunks)"
    echo; echo "RESULT: PASS"; echo "==================================================================="
    exit 0
fi
echo "      MISMATCH: java=$NJ program=$NP"
echo "      only in java     : $(comm -23 "$GOT" "$WORK/prog_set.txt" | wc -l)"
echo "      only in the binary: $(comm -13 "$GOT" "$WORK/prog_set.txt" | wc -l)"
comm -3 "$GOT" "$WORK/prog_set.txt" 2>/dev/null | head -10 | sed 's/^/        /'
echo; echo "RESULT: FAIL"; echo "==================================================================="
exit 1
