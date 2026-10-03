# 工具与脚本

| 路径 | 用途 | 调用 |
|---|---|---|
| `build.sh` | 构建三个二进制到 `build/`。默认产出 **SASS(native) + 同架构 PTX**（更新的卡可让驱动 JIT） | `./build.sh`；`CUDA_ARCH=... ./build.sh` 覆盖架构 |
| `bench.sh` | 多负载速度回归：min/median/mean、B/s、重复间输出一致性（`det` 列） | `./bench.sh [--quick|--full] [--outdir DIR]` |
| `src/slime_main.cu` | 主程序（见 `docs/zh-cn/optimization-design.md`） | `<seed> <x0> <z0> <x1> <z1> <sizeX> <sizeZ> <thr> <out.csv> [--sort=on|off]` |
| `src/slime_circle.cu` | 二级筛选：圆盘分数 | `<in.csv> <radius> <sizeX> <sizeZ> <seed> <out.csv> <threshold>` |
| `src/slime_cmp.c` | 用户侧距离/计数过滤 | `<in.csv> <distance> <threshold> <out.csv>` |
| `src/slime_hash.cuh` | 史莱姆判定的**唯一实现**（上面三个共用） | 头文件 |
| `tools/oracle.c` | 从 Java 规范直译的**独立 oracle**（与 `src/` 零共享代码）。模式：出 CSV / `--probe` / `--scan-reject` | `./build/oracle --seed S --x0 .. --z0 .. --x1 .. --z1 .. --sx U --sz V [--probe|--scan-reject] --out FILE` |
| `tools/oracle_pycheck.py` | 第三个实现，用来校验 oracle 自己 | `python3 tools/oracle_pycheck.py` |
| `tools/collect.sh` | 一条命令采集性能与正确性数据（画像/耗时/ncu/位精确/对拍），写 `.work/collect/<时间戳>/` 并打成同名 `.tar.gz`，目录内另有 `SUMMARY.md` 汇总各项结论 | `./tools/collect.sh --ref 旧二进制 [--reps N]` |
| `tools/mccrosscheck.sh` | 把 `docs/en-us/index.md` 里 wiki 的原始 Java 逐字抽出来交给**真 JVM** 跑，与程序对拍 | `./tools/mccrosscheck.sh [--java PATH]`（**需 JDK**，没有就以 rc=2 退出） |
| `tools/verify.sh` | **位精确矩阵**：与 `--ref` 给出的旧可执行文件在同一套配置上逐字节比对 | `./tools/verify.sh --ref BIN [--fast]`；`--plan-only` 只看矩阵 |
| `tools/fullcheck.sh` | **穷举 soak**：合法域全跑（sizeX 1..255 × sizeZ 1..32 × 3 种子，sizeX<=32 再 ×6 编译变体）+ 负向对照 | `./tools/fullcheck.sh [--hours N]`，结果在 `.fullcheck/summary.txt` |
| `tools/releasecheck.sh` | 发布前总检查：跑上面几项并写出报告 | `./tools/releasecheck.sh [--out FILE] [--soak HOURS]` |
| `tools/gpuinfo.sh` | GPU 状态快照；`-w [秒] [-t 温度]` 等到目标温度，`-c` 判定是否处于标准状态 | `./tools/gpuinfo.sh [-1] [-w 900 -t 35] [-c]` |
| `tools/quickbench.sh` | 速度表；带 `--ref` 时给交错 A/B 比值 | `./tools/quickbench.sh [--full] [-n REPS] [--ref BIN]` |
| `tools/quickcheck.sh` | 约十秒内拿独立 Java oracle 校验输出 | `./tools/quickcheck.sh` |

**`verify.sh` 的 `--ref` 没有默认值**：冻结参考是某次构建的产物，且 CUDA 工具链在 ELF 层面
不可复现，所以它不入库，必须由调用方给出路径。`--ref` 同目录若有 `<名称>.md5` 会用于完整性校验。

## 运行期旋钮

| 变量 | 作用 |
|---|---|
| `SLIME_PRINT_TUNING=1` | 打印每张卡派生出的并行度阈值 |
| `SLIME_NO_AUTOTUNE=1` | 关闭运行期自适应，退回编译期常量 |
| `SLIME_FAKE_SM` / `SLIME_FAKE_BPSM` / `SLIME_FAKE_SHARED_PER_SM` / `SLIME_FAKE_SHARED_PER_BLOCK` | 测试钩子：模拟别的卡的画像 |
| `SLIME_CAP_SLOTS` | 覆盖结果缓冲区大小（压小可强制走「缓冲区满 → 折半重试」） |
| `SLIME_PROGRESS=1` / `SLIME_NO_PROGRESS=1` | 强制开/关进度条（默认只在 stderr 是终端时显示） |
| `SLIME_VERIFY_TIMEOUT` | `verify.sh` 每项的硬超时（默认 900 s） |

编译期旋钮（`-D`）：`SLIME_K4_MIN_BLOCKS`（`0` 强制 K=4 / `2147483647` 强制 K=1）、
`SLIME_Z_SUB_ROWS`、`SLIME_Z_SPLIT_MIN_TILES`（测试用它强制走不切分那条路径）、
`SLIME_PROFILE`（分阶段计时）、`SLIME_SCAN_ONLY`（只保留命中判定、不做登记）、
`SLIME_NO_PROGRESS_CODE`（把进度代码整段编译掉；用于验证它对设备侧零影响）。
