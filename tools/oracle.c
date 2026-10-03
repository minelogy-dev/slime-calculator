/*
 * tools/oracle.c -- independent reference implementation for slime-calculator.
 *
 * This file is deliberately written from the *Java specification* and from
 * java.util.Random's documented behaviour, NOT from src/slime_main.cu.  It is
 * the golden model that tools/fullcheck.sh compares the GPU program against.
 * Nothing here is shared with the program under test.
 *
 * Java reference (Minecraft, Java Edition):
 *
 *     Random rnd = new Random(
 *           seed
 *         + (long)(x * x * 4987142)          // int arithmetic, then widened
 *         + (long)(x * 5947611)              // int arithmetic, then widened
 *         + (long)(z * z) * 4392871L         // widened first, then long multiply
 *         + (long)(z * 389711)               // int arithmetic, then widened
 *         ^ 987234911L);                     // '^' binds looser than '+'
 *     boolean slime = rnd.nextInt(10) == 0;
 *
 * java.util.Random:
 *     Random(long s)  { this.seed = (s ^ 0x5DEECE66DL) & ((1L << 48) - 1); }
 *     int next(int bits){ this.seed = (this.seed * 0x5DEECE66DL + 0xBL)
 *                                      & ((1L << 48) - 1);
 *                         return (int)(this.seed >>> (48 - bits)); }
 *     int nextInt(int bound) {
 *         int r = next(31);
 *         int m = bound - 1;
 *         if ((bound & m) == 0) { r = (int)((bound * (long)r) >> 31); }
 *         else { for (int u = r; u - (r = u % bound) + m < 0; u = next(31)); }
 *         return r;
 *     }
 *
 * 'nextInt(10)' therefore rejects when (u - u % 10 + 9) overflows int, i.e.
 * when u >= 2147483640; by the project's own DFS proof at most one rejection
 * can happen for this generator, but the loop below is written generically.
 *
 * Build:  gcc -O2 -fwrapv -o build/oracle tools/oracle.c
 * (the -fwrapv is belt-and-braces; all wrap-sensitive arithmetic below is done
 *  in unsigned types so the result does not depend on it)
 *
 * Output is pure ASCII by design (the check is meant to run on a headless
 * machine where non-ASCII output may not render).
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MASK48 0xFFFFFFFFFFFFULL
#define JMULT 0x5DEECE66DULL
#define JADD 0xBULL

/* ------------------------------------------------------------------ */
/* Java-literal predicate                                             */
/* ------------------------------------------------------------------ */

/* true iff  new java.util.Random(seed_arg).nextInt(10) == 0  */
static int java_next_int10_is_zero(int64_t seed_arg) {
    uint64_t seed = ((uint64_t)seed_arg ^ JMULT) & MASK48;
    for (int guard = 0; guard < 8; guard++) {
        seed = (seed * JMULT + JADD) & MASK48;
        int32_t u = (int32_t)(uint32_t)(seed >> 17); /* next(31) */
        int32_t r = (int32_t)((int64_t)u % 10);      /* u >= 0, so >%> and % agree */
        uint32_t cond = (uint32_t)u - (uint32_t)r + 9u; /* Java int wrap */
        if ((int32_t)cond >= 0) return r == 0;       /* accepted */
        /* rejected: loop re-draws next(31) */
    }
    return 0; /* not reachable for bound = 10 */
}

/* true iff chunk (cx, cz) is a slime chunk for world_seed */
static int is_slime_chunk(int64_t world_seed, int32_t cx, int32_t cz) {
    uint32_t ux = (uint32_t)cx;
    uint32_t uz = (uint32_t)cz;
    /* int arithmetic (wraps mod 2^32), then sign-extended to long */
    int64_t t1 = (int64_t)(int32_t)((ux * ux) * 4987142u);
    int64_t t2 = (int64_t)(int32_t)(ux * 5947611u);
    int64_t t4 = (int64_t)(int32_t)(uz * 389711u);
    /* (long)(z*z) is widened *before* the long multiply */
    int64_t t3 = (int64_t)(int32_t)(uz * uz) * 4392871LL;

    /* Java long addition wraps mod 2^64: do it in uint64_t */
    uint64_t sum = (uint64_t)world_seed + (uint64_t)t1 + (uint64_t)t2 + (uint64_t)t3 + (uint64_t)t4;
    uint64_t arg = sum ^ 987234911ULL; /* '^' applies to the whole long sum */
    /* new java.util.Random(arg) */
    return java_next_int10_is_zero((int64_t)arg);
}

/* bits % 10 == 0 for a 31-bit draw, without any rejection handling: this is the
 * answer an implementation gets if it forgets to re-draw.  Together with the
 * correct answer it tells us whether a rejection cell can distinguish the two. */
static int mod10_is_zero(uint32_t bits) {
    uint32_t q = bits * 0xCCCCCCCDu;              /* ceil(2^35/10) */
    q = (q >> 1) | (q << 31);                     /* rotate right by 1 */
    return q <= 0x19999999u;
}

/* ------------------------------------------------------------------ */
/* counting methods                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    int64_t seed;
    int32_t x0, x1, z0, z1; /* scanned region, chunk coordinates (inclusive) */
    int32_t sx, sz;         /* window size */
} Region;

static inline int64_t out_rows(const Region *r) { return (int64_t)r->z1 - r->z0 + 1 - (r->sz - 1); }
static inline int64_t out_cols(const Region *r) { return (int64_t)r->x1 - r->x0 + 1 - (r->sx - 1); }

/* Brute force: literally sum sx*sz booleans per candidate.  Obvious, slow,
 * and therefore the most trustworthy method. */
static void run_naive(const Region *r, void (*emit)(void *, int32_t, int32_t, int32_t), void *ctx,
                      int64_t *hist, int *maxcount) {
    int64_t ocols = out_cols(r), orows = out_rows(r);
    for (int64_t i = 0; i < orows; i++) {
        int32_t cz = r->z0 + (int32_t)i;
        for (int64_t j = 0; j < ocols; j++) {
            int32_t cx = r->x0 + (int32_t)j;
            int32_t cnt = 0;
            for (int32_t dz = 0; dz < r->sz; dz++)
                for (int32_t dx = 0; dx < r->sx; dx++)
                    cnt += is_slime_chunk(r->seed, cx + dx, cz + dz);
            hist[cnt]++;
            if (cnt > *maxcount) *maxcount = cnt;
            if (emit) emit(ctx, cx, cz, cnt);
        }
    }
}

/* Rolling column sums: O(W*H) predicate evaluations, O(sz*W) memory.
 * Structurally different from both the naive method and from the program
 * (which slides X with ballot bitmaps and Z with a shared ring). */
static void run_roll(const Region *r, void (*emit)(void *, int32_t, int32_t, int32_t), void *ctx,
                     int64_t *hist, int *maxcount) {
    int64_t W = (int64_t)r->x1 - r->x0 + 1;
    int64_t ocols = out_cols(r), orows = out_rows(r);
    int32_t sz = r->sz;

    uint8_t *ring = (uint8_t *)malloc((size_t)sz * (size_t)W);
    int32_t *colsum = (int32_t *)malloc((size_t)W * sizeof(int32_t));
    if (!ring || !colsum) { fprintf(stderr, "oracle: out of memory\n"); exit(3); }
    memset(colsum, 0, (size_t)W * sizeof(int32_t));

    for (int32_t dz = 0; dz < sz; dz++) {
        int32_t cz = r->z0 + dz;
        uint8_t *row = ring + (size_t)dz * (size_t)W;
        for (int64_t j = 0; j < W; j++) {
            uint8_t v = (uint8_t)is_slime_chunk(r->seed, r->x0 + (int32_t)j, cz);
            row[j] = v;
            colsum[j] += v;
        }
    }

    for (int64_t i = 0; i < orows; i++) {
        int32_t cz = r->z0 + (int32_t)i;
        /* running window sum over the current column sums */
        int32_t run = 0;
        for (int32_t dx = 0; dx < r->sx; dx++) run += colsum[dx];
        for (int64_t j = 0; j < ocols; j++) {
            int32_t cnt = run;
            hist[cnt]++;
            if (cnt > *maxcount) *maxcount = cnt;
            if (emit) emit(ctx, r->x0 + (int32_t)j, cz, cnt);
            if (j + 1 < ocols) {
                run += colsum[j + r->sx] - colsum[j];
            }
        }
        /* slide Z: drop the oldest row, add the next one */
        uint8_t *slot = ring + (size_t)(i % sz) * (size_t)W;
        int32_t nz = cz + sz;
        if (i + 1 < orows) {
            for (int64_t j = 0; j < W; j++) {
                uint8_t v = (uint8_t)is_slime_chunk(r->seed, r->x0 + (int32_t)j, nz);
                colsum[j] += (int32_t)v - (int32_t)slot[j];
                slot[j] = v;
            }
        }
    }
    free(ring);
    free(colsum);
}

/* histogram only (no emission) for a given method */
static void histogram(const Region *r, int method, int64_t *hist, int *maxcount) {
    if (method == 0)
        run_naive(r, NULL, NULL, hist, maxcount);
    else
        run_roll(r, NULL, NULL, hist, maxcount);
}

/* emission context */
typedef struct {
    FILE *f;
    int32_t threshold;
    int64_t written;
} Sink;

static void emit_row(void *ctx, int32_t cx, int32_t cz, int32_t cnt) {
    Sink *s = (Sink *)ctx;
    if (cnt >= s->threshold) {
        /* slime_main reports block coordinates (chunk * 16) */
        fprintf(s->f, "%" PRId32 ",%" PRId32 ",%" PRId32 "\n", cx * 16, cz * 16, cnt);
        s->written++;
    }
}

/* ------------------------------------------------------------------ */
/* CLI                                                                */
/* ------------------------------------------------------------------ */

static void usage(void) {
    fprintf(stderr,
            "usage: oracle --seed S --x0 A --z0 B --x1 C --z1 D --sx U --sz V\n"
            "              [--mode auto|t] [--threshold T] [--target N]\n"
            "              [--method roll|naive|both] --out FILE\n"
            "  region is the SCANNED region in chunk coordinates, inclusive\n"
            "  --sx/--sz are the window (rectangle) sizes\n"
            "  --mode auto picks the largest threshold with <= --target hits\n");
}

int main(int argc, char **argv) {
    int64_t seed = 0;
    int32_t x0 = 0, z0 = 0, x1 = 0, z1 = 0, sx = 0, sz = 0;
    int mode_auto = 1;
    int64_t threshold = 0;
    int64_t target = 2000;
    int method = 1; /* 1 = roll, 0 = naive, 2 = both */
    const char *out = NULL;
    int probe = 0;
    int scan_reject = 0;
    int32_t pcx = 0, pcz = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--seed") && i + 1 < argc) seed = strtoll(argv[++i], NULL, 10);
        else if (!strcmp(a, "--x0") && i + 1 < argc) x0 = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--z0") && i + 1 < argc) z0 = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--x1") && i + 1 < argc) x1 = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--z1") && i + 1 < argc) z1 = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--sx") && i + 1 < argc) sx = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--sz") && i + 1 < argc) sz = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--threshold") && i + 1 < argc) { threshold = strtoll(argv[++i], NULL, 10); mode_auto = 0; }
        else if (!strcmp(a, "--mode") && i + 1 < argc) mode_auto = !strcmp(argv[++i], "auto");
        else if (!strcmp(a, "--target") && i + 1 < argc) target = strtoll(argv[++i], NULL, 10);
        else if (!strcmp(a, "--probe")) probe = 1;
        else if (!strcmp(a, "--scan-reject")) scan_reject = 1;
        else if (!strcmp(a, "--cx") && i + 1 < argc) pcx = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--cz") && i + 1 < argc) pcz = (int32_t)strtol(argv[++i], NULL, 10);
        else if (!strcmp(a, "--method") && i + 1 < argc) {
            const char *m = argv[++i];
            method = !strcmp(m, "naive") ? 0 : (!strcmp(m, "both") ? 2 : 1);
        } else if (!strcmp(a, "--out") && i + 1 < argc) out = argv[++i];
        else { usage(); return 2; }
    }
    if (scan_reject) {
        /*
         * Directed search for cells that take the Java rejection branch and
         * where the branch actually CHANGES the answer.  Such a cell is the
         * only kind of input that can distinguish a correct implementation
         * from one that skips the re-draw, and the branch has probability
         * ~4e-9, so a random region essentially never contains one.  This mode
         * exists so the directed test shape can be re-derived from scratch if
         * the hash tables ever change.
         */
        const uint64_t MASK = MASK48, MU = JMULT, AD = JADD;
        const uint64_t XA = 987234911ULL ^ JMULT;
        int64_t cells = 0, nrej = 0, nsens = 0;
        for (int32_t cz = z0; cz <= z1; cz++) {
            uint32_t uz = (uint32_t)cz;
            int64_t bz = (int64_t)(int32_t)(uz * uz) * 4392871LL + (int64_t)(int32_t)(uz * 389711u);
            for (int32_t cx = x0; cx <= x1; cx++) {
                uint32_t ux = (uint32_t)cx;
                int64_t bx = (int64_t)(int32_t)((ux * ux) * 4987142u)
                           + (int64_t)(int32_t)(ux * 5947611u) + (int64_t)seed;
                cells++;
                uint64_t u = ((uint64_t)bx + (uint64_t)bz) & MASK;
                uint64_t r = ((u ^ XA) * MU + AD) & MASK;
                uint32_t bits = (uint32_t)(r >> 17);
                if (bits < 2147483640u) continue;
                nrej++;
                uint64_t r2 = (r * MU + AD) & MASK;
                uint32_t bits2 = (uint32_t)(r2 >> 17);
                int fast = mod10_is_zero(bits);       /* what a buggy fast path gives */
                int correct = mod10_is_zero(bits2);   /* what Java gives */
                if (fast != correct) {
                    nsens++;
                    printf("REJECTCELL seed=%" PRId64 " cx=%" PRId32 " cz=%" PRId32
                           " fast=%d correct=%d\n", seed, cx, cz, fast, correct);
                }
            }
        }
        printf("SCAN seed=%" PRId64 " cells=%" PRId64 " reject=%" PRId64 " sensitive=%" PRId64 "\n",
               seed, cells, nrej, nsens);
        return 0;
    }

    if (sx < 1 || sz < 1) { usage(); return 2; }

    if (probe) {
        /* single-candidate count: used to spot-check huge regions without
         * materialising the whole grid */
        int32_t cnt = 0;
        for (int32_t dz = 0; dz < sz; dz++)
            for (int32_t dx = 0; dx < sx; dx++)
                cnt += is_slime_chunk(seed, pcx + dx, pcz + dz);
        printf("PROBE seed=%" PRId64 " cx=%" PRId32 " cz=%" PRId32 " sx=%" PRId32 " sz=%" PRId32
               " count=%" PRId32 "\n",
               seed, pcx, pcz, sx, sz, cnt);
        return 0;
    }

    if (!out || x1 < x0 || z1 < z0) { usage(); return 2; }

    Region r = {seed, x0, x1, z0, z1, sx, sz};
    int64_t ocols = out_cols(&r), orows = out_rows(&r);
    if (ocols <= 0 || orows <= 0) {
        fprintf(stderr, "oracle: window larger than region\n");
        return 2;
    }

    int64_t *hist = (int64_t *)calloc((size_t)sx * (size_t)sz + 2, sizeof(int64_t));
    if (!hist) { fprintf(stderr, "oracle: out of memory\n"); return 3; }
    int maxcount = 0;
    histogram(&r, method == 0 ? 0 : 1, hist, &maxcount);

    int64_t chosen = threshold;
    int64_t hits = 0;
    if (mode_auto) {
        int64_t acc = 0;
        chosen = maxcount + 1;
        for (int c = maxcount; c >= 0; --c) {
            if (acc + hist[c] > target) break;
            acc += hist[c];
            chosen = c;
        }
        if (acc == 0) {
            /* Even the most frequent count is more common than the target (this
             * happens for tiny rects, where ~10% of candidates are hits).  Use
             * the top bin -- NOT 0, which would emit every single candidate. */
            if (maxcount > 0 && hist[maxcount] > 0) { chosen = maxcount; acc = hist[maxcount]; }
            else chosen = 0; /* genuinely empty region */
        }
        hits = acc;
    } else {
        for (int c = (int)threshold; c <= maxcount; c++) hits += hist[c];
    }

    /* cross-check: on request, recompute the histogram with the other method */
    if (method == 2) {
        int64_t *hist2 = (int64_t *)calloc((size_t)sx * (size_t)sz + 2, sizeof(int64_t));
        int max2 = 0;
        histogram(&r, 0, hist2, &max2);
        int bad = 0;
        for (int c = 0; c <= maxcount || c <= max2; c++)
            if (hist[c] != hist2[c]) bad++;
        if (bad || max2 != maxcount) {
            fprintf(stderr, "oracle: SELF-CHECK FAILED (roll vs naive disagree)\n");
            return 4;
        }
        free(hist2);
    }

    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "oracle: cannot open %s\n", out); return 3; }
    fprintf(f, "x,z,slime_count\n");
    Sink sink = {f, (int32_t)chosen, 0};
    if (method == 0)
        run_naive(&r, emit_row, &sink, hist, &maxcount);
    else
        run_roll(&r, emit_row, &sink, hist, &maxcount);
    fclose(f);

    printf("ORACLE method=%s seed=%" PRId64 " region=[%" PRId32 ",%" PRId32 "]x[%" PRId32 ",%" PRId32
           "] rect=%" PRId32 "x%" PRId32 " cand=%" PRId64 " threshold=%" PRId64 " hits=%" PRId64
           " emitted=%" PRId64 " maxcount=%d\n",
           method == 0 ? "naive" : "roll", seed, x0, x1, z0, z1, sx, sz, ocols * orows, chosen, hits,
           sink.written, maxcount);
    free(hist);
    return 0;
}
