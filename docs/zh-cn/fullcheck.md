# `tools/fullcheck.sh` —— 长时间、只求正确性的正确性检查

这是一项以时间换覆盖率的检查：用 5~8 小时独占 GPU 去**证伪** `slime_main`，不测速度。
脚本、oracle 与报告的输出都是纯 ASCII，可在无头机器上直接运行。

```bash
./build.sh                      # 可选：fullcheck 自身也会重建
./tools/fullcheck.sh --hours 1 --phases 0     # 约 2 分钟：校验 oracle 与 harness 自身
./tools/fullcheck.sh --hours 6                # 正式运行，默认 6 小时预算
./tools/fullcheck.sh --hours 6 --resume       # 续跑，已 PASS 的配置跳过
./tools/fullcheck.sh --hours 8 --limit 0      # 不限配置数，跑到 8 小时预算用尽
```

需要的环境：`bash`、`gcc`、`nvcc`；`python3` 可选（只在 P0 用一次）。
不依赖任何外部数据或中间产物。

---

## 1. 为什么它可信：信任链

```
Java 规范  ──►  tools/oracle.c          （独立实现，从 JDK 语义逐条抄写）
           ──►  tools/oracle_pycheck.py （第三方实现，任意精度整数，手工按 32/64 位回绕）
           ──►  slime_main              （被测程序）
```

* **`tools/oracle.c` 与 `src/slime_main.cu` 不共享任何一行代码。** 它把
  `new java.util.Random(seed + (long)(x*x*4987142) + (long)(x*5947611) + (long)(z*z)*4392871L
  + (long)(z*389711) ^ 987234911L).nextInt(10) == 0` 按 Java 语义直译，
  包括 `nextInt` 的拒绝采样循环和 `int` 溢出规则。
* **oracle 自身也被校验**：P0 用第二个独立实现（Python）逐个候选比对，再用两种互不相同的
  求和方法（暴力逐格累加、滚动列和）互相校验。
* **harness 的负向对照**：P0 把源码里 `d_results[pos].z = z * 16;` 改成 `(z + 1) * 16`
  编出一个坏二进制，harness 必须报 FAIL；没报 FAIL 即说明 harness 本身失效。

## 2. 它到底比什么

每个配置都做五件事：

1. **扫描区间**：程序打印的 `Global range:` 第二段必须等于按文档补齐规则算出的区间
   （X 按 256 对齐；融合路径的 Z 可预测；回退路径的 `H_max` 取决于空闲显存，因此先探一次
   取真实区间）。同时断言**请求区间被扫描区间包含**，以排除少扫。
2. **逐行不变式**：块坐标必须是 16 的倍数、落在扫描区间内、`count >= threshold`、
   无重复、`--sort=on` 时严格按 (x,z) 递增；`Total valid:` 必须等于 CSV 行数。
3. **与 oracle 全量比对**：两边都做数值排序后 `cmp`，比的是**完整命中集合**，
   既抓多报也抓漏报。
4. **超区域（P4）**：区域太大没法物化时，改用**随机单点探针** ——
   取 300 个候选，oracle 逐个算真值，再在程序的 CSV 里查这个坐标在不在，双向都抓。
5. **确定性与单调性（P5）**：同一配置跑两遍必须逐字节相同；`--sort=on` 与 `--sort=off`
   必须是同一个多重集；`threshold+1` 的命中集合必须是 `threshold` 的子集。

## 3. 覆盖矩阵（阶段按价值排序，时间不够会自然截断）

| 阶段 | 内容 | 规模 | 大致耗时 |
|---|---|---|---|
| P0 | oracle 三方互验、已知答案、**负向对照** | 5 项 | ~30 s |
| P1 | **穷举 `sizeX` 1..255 × `sizeZ` 1..32**（合法域全遍历），×3 个种子，`sizeX<=32` 的配置再 ×6 种编译变体（默认 / 强制 K=4 / 强制 K=1 / 强制 zSub=1 / 细 Z 切分 / 强制不切 Z） | 24480 配置 | ~4.5 h |
| P1b | 同上但在**高 Z** 区域（16384 行，zSub>1 全路径） | 1024 配置 | ~40 min |
| P2 | 几何边界：X 宽 1..4096（跨 256/1024/2048）、Z 高 1..65537（跨 256/2048/65536）、坐标 ±2,000,000、两个种子、6 种 rect | ~170 配置 | ~12 min |
| P3 | 旋钮：`SLIME_CAP_SLOTS`（强制结果缓冲溢出重试）× `SLIME_SORT_BLOCKS`（强制落盘 spill + k 路归并）× `--sort on/off` | ~150 运行 | ~6 min |
| P4 | 超大区域（全图级）不变式 + 300 点随机探针 | 6 区域 | ~10 min |
| P5 | 确定性 / 单调性 / 两种输出模式等价 | 4 区域 | ~3 min |

**P1 取穷举而非抽样**：`sizeX` × `sizeZ` 的合法域共 8160 个点，全部跑完才能给出
「每个合法矩形尺寸都验过」这一可证伪的陈述。`sizeX > 32` 走另一套内核（行前缀和 +
滑动窗口），对应 223×32 个点尤其不能省。

## 3.5 发布时怎么用

不单独运行它，而是用 `tools/releasecheck.sh --soak 6`：该命令依次跑 quickcheck、
mccrosscheck（真 JVM 对拍 wiki 原始 Java）、`verify.sh`（位精确矩阵）与本次穷举 soak，
最后写出一份发布验证报告（`--out FILE`，默认 `.work/release-verification.md`），
记录 git commit、源码与二进制的 sha256、各阶段结果与原始日志路径，以及这份报告没有证明什么。

soak 按时间预算运行（`--hours`），因此「部分完成」表示预算用尽而不是失败，报告中亦如此记录。
实测速率约每分钟 58 个配置，一个种子跑完 8160 个矩形约 2.3 小时。跑满 5~8 小时用 `--soak 8`，
分次跑用 `--resume`。

## 4. 结果怎么读

* `.fullcheck/summary.txt` —— 各阶段通过/失败计数 + `RESULT: PASS|FAIL`
* `.fullcheck/results.tsv` —— 每个配置一行：`阶段 / 标签 / PASS|FAIL|SKIP / 详情 / 时间`
* `.fullcheck/failures.txt` —— 失败清单（前 20 条会同时打到 stdout）
* `.fullcheck/log/` —— 每次运行的完整程序输出与 oracle 输出（排查用）
* 退出码：全过 0，有失败 1，构建失败 2

**判据**：`RESULT: PASS` 表示在这套矩阵下程序输出与独立 oracle 逐字节一致，且所有结构
不变式成立。任何一条 FAIL 都按真问题排查。

## 5. 已知边界

* **温度与频率不影响正确性**，因此本检查不做起始温度对齐（那属于计时协议）。在独占 GPU 的
  机器上直接运行即可。
* **oracle 覆盖不到的部分**：`oracle.c` 是纯 CPU 的 Java 语义模型，验证的是计算结果是否正确，
  不验证计算是否真的全部发生在 GPU 上（后者属于性能问题）。
* **回退路径（`sizeX > 32`）的 Z 扫描区间**依赖运行时空闲显存，无法从请求预测；harness 用
  程序自报的区间比对，只强制「请求 ⊆ 扫描」。这是已记录的口径差异，不是缺陷。
* **超时**：单个配置默认 3600 s 硬超时（`--timeout` 可改），超时按 FAIL 记。
