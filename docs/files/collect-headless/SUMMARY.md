# collect.sh results

timestamp   20261003-174158
host        kamov
src sha256  505a5930291a5688ed6b356199e0810f9ff8092b7bf182720409045dec98fe4e
bin md5     2412c2434d49599034819b9ed9e51b2a
reference   .work/verify/ref_head
repetitions 3

## key numbers

| metric | value | source |
|---|---|---|
| full map (14.06e12 candidates) | wall_min 37133.41 ms, gpu_min 35583.09 ms, B/s_wall 378.7 | 02-quickbench-full.txt |
| full-map candidates | 14062507.5 M cells | 02-quickbench-full.txt |

## workloads

```
workload             cells_M  reps    gpu_min   wall_min   B/s_gpu  B/s_wall    ratio   det
---------------- ----------- ----- ---------- ---------- --------- --------- -------- -----
rect17_300k          90000.6     3     222.87     327.50     403.8     274.8        -    ok
xwide_2Mx32k         64002.0     3     156.77     258.92     408.3     247.2        -    ok
ztall_32kx256k       16384.5     3      43.74     135.68     374.6     120.8        -    ok
rect32_100k          10000.2     3      31.17     121.29     320.8      82.4        -    ok
fullmap_14G       14062507.5     3   35583.09   37133.41     395.2     378.7        -    ok

```

## files

00-identity.txt                environment and identity (toolchain versions, src/bin hashes, dirty-file count)
01-gpuinfo.txt                 GPU profile (model, SM count, driver, clocks, temperature)
01-gpuinfo-oneline.txt         the same, single line
02-quickbench-full.txt         per-workload speed table (full-map wall clock and row count)
03-ncu.csv                     ncu metrics (inst/candidate numerator, cycles, registers, occupancy, pipe mix)
03-inst-per-candidate.txt      inst/candidate (frequency-independent)
03-ncu-report.ncu-repz         raw ncu report; open with ncu --import or ncu-ui
04-verify-full.txt             bit-exactness matrix against --ref (byte-for-byte)
05-mccrosscheck.txt            cross-check vs the wiki Java on a real JVM
06-quickcheck.txt              complete-output comparison against the independent oracle

## stage verdicts

04-verify-full.txt             (no verdict line)
05-mccrosscheck.txt            (no verdict line)
06-quickcheck.txt              RESULT: PASS
