#!/usr/bin/env python3
"""
tools/oracle_pycheck.py -- validate build/oracle against a THIRD, independent
implementation of the same Java specification, written in Python.

This exists because the whole correctness argument rests on the oracle being
right.  Python's arbitrary-precision integers make the Java semantics explicit:
every intermediate is wrapped to 32 or 64 bits by hand.

Usage:  python3 tools/oracle_pycheck.py [path-to-oracle]
Exit code 0 = agreement, 1 = mismatch, 2 = usage error.
Output is pure ASCII.
"""
import subprocess
import sys

MASK32 = (1 << 32) - 1
MASK48 = (1 << 48) - 1
MASK64 = (1 << 64) - 1
MULT = 0x5DEECE66D
ADD = 0xB


def i32(v):
    v &= MASK32
    return v - (1 << 32) if v >= (1 << 31) else v


def i64(v):
    v &= MASK64
    return v - (1 << 64) if v >= (1 << 63) else v


class JavaRandom:
    """java.util.Random, transcribed from the JDK source."""

    def __init__(self, seed):
        self.seed = (seed ^ MULT) & MASK48

    def next(self, bits):
        self.seed = (self.seed * MULT + ADD) & MASK48
        value = self.seed >> (48 - bits)          # Java >>> on a 48-bit value
        return i32(value)                          # cast to int

    def next_int(self, bound):
        r = self.next(31)
        m = bound - 1
        if (bound & m) == 0:
            return i32((bound * r) >> 31)
        while True:
            u = r
            r = i32(u % bound)
            if i32(u - r + m) >= 0:                # Java int comparison
                return r

    def next_int10_is_zero(self):
        return self.next_int(10) == 0


def is_slime(seed, x, z):
    """Minecraft Java Edition slime chunk test."""
    s = i64(seed
            + i32(x * x * 4987142)
            + i32(x * 5947611)
            + i64(i32(z * z) * 4392871)
            + i32(z * 389711))
    arg = i64(s ^ 987234911)
    return JavaRandom(arg).next_int10_is_zero()


def main():
    oracle = sys.argv[1] if len(sys.argv) > 1 else "build/oracle"
    seed = 114514
    x0, z0, x1, z1 = -33, -17, 29, 41
    sx, sz = 17, 17
    out = "/tmp/oracle_pycheck.csv"
    proc = subprocess.run(
        [oracle, "--seed", str(seed), "--x0", str(x0), "--z0", str(z0),
         "--x1", str(x1), "--z1", str(z1), "--sx", str(sx), "--sz", str(sz),
         "--mode", "auto", "--target", str(10 ** 9), "--method", "both", "--out", out],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print("oracle failed:", proc.stderr.strip())
        return 1

    rows = {}
    with open(out) as f:
        for line in f.read().splitlines()[1:]:
            a, b, c = line.split(",")
            rows[(int(a), int(b))] = int(c)

    checked = 0
    bad = 0
    for cx in range(x0, x1 - sx + 2):
        for cz in range(z0, z1 - sz + 2):
            want = 0
            for dx in range(sx):
                for dz in range(sz):
                    if is_slime(seed, cx + dx, cz + dz):
                        want += 1
            checked += 1
            got = rows.get((cx * 16, cz * 16), 0)
            if got != want:
                bad += 1
                if bad <= 5:
                    print("MISMATCH cx=%d cz=%d oracle=%d python=%d" % (cx, cz, got, want))

    # a few explicit spot values, printed so the log carries concrete evidence
    for (px, pz) in ((0, 0), (1, 1), (-1, -1), (5, 7), (1000000, -1000000), (1999999, 1999999)):
        vals = [is_slime(seed, px + dx, pz + dz) for dx in range(3) for dz in range(3)]
        print("SPOT seed=%d chunk=(%d,%d) 3x3=%s" % (seed, px, pz, "".join("1" if v else "0" for v in vals)))

    print("PYCHECK checked=%d rows=%d mismatches=%d" % (checked, len(rows), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
