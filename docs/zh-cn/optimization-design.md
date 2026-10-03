# slime_main 优化设计

## 1. 任务形状

对一批候选区域，逐个候选位置统计其 **sizeX × sizeZ 窗口**内的史莱姆区块数，
输出所有 `count >= threshold` 的位置。语义要求：范围内**一个不漏**，多输出不允许（按 `(x,z,count)` 去重后逐字节可比）。

融合路径的前提是 `sizeX <= MAX_FUSED_SIZE_X`：整个窗口位段必须能落在「相邻两个字拼成的 64 位」里。
`sizeX > 32` 走宽窗回退（整行 `uint8` 前缀和 + 两次查表）。

---

## 2. 计算核的结构

### 2.1 三层数据流

| 层 | 职责 |
|---|---|
| **写侧** | 每线程负责 K 个**跨步**列，逐格做判定，`__ballot_sync` 得到 32 列的行位图，由 `lane==0` 存入 shared 行环 |
| **X 向记账** | `scanRectFusedMerge4Kernel`：每 lane 负责 **4 个相邻候选列**（一个 bundle），热路径只维护**一个** bundle 级松上界 `U` |
| **闸门与登记** | `U >= threshold` 才对该 bundle 的 4 个候选从行位图环**按需精确重算**，命中则 `atomicAdd` 领槽登记 |

### 2.2 共享上界为什么不会漏结果

候选 `c` 的窗口 `[gx+c, gx+c+sizeX-1]` 对 `c ∈ [0,3]` 都是并集 `[gx, gx+sizeX+2]` 的子集，
行区间相同、计数非负 ⇒ 精确值 `E(c) <= U`。于是 `U < threshold` 的 bundle 里不可能有命中。

**闸门必须写 `U >= threshold`**：写成 `>` 会漏掉 `E == threshold` 的命中。

### 2.3 热路径唯一状态

每行对 `U` 做「加本行并集 popc、减 sizeZ 行前那一行并集 popc」，每 bundle 每行 2 次
`2×LDS + SHF + LOP3 + POPC + IADD3`，**与候选数无关**。

`merge4` **不保留 per-候选计数环**：那正是本方案要删掉的那次逐列 popc；保留它等于把收益买回去。

### 2.4 抽取宽度

并集位段宽 `sizeX + 3`，要一次 32 位 `__funnelshift_r` 取出 ⇒ `sizeX <= 29`（`MAX_MERGE4_SIZE_X`）。
`sizeX ∈ [30,32]` 回退到旧的计数环核，`sizeX > 32` 回退宽窗前缀和核。

---

## 3. 判定函数（`src/slime_hash.cuh`）

与 Java 规范逐位等价，热路径 SASS 从 18 条降到 15 条。四处变形：

1. `(s ^ 987234911) ^ 0x5DEECE66D` 合并成一个常量 XOR，省一次 64 位 XOR；
2. 48×35 的 LCG 乘法整体左移 16 位再做（「中段抽取」）：`Q = (X * (MULT<<16) + (0xB<<16)) mod 2^64 = W << 16`，
   `Q` 的高 32 位再 `>>1` 就是 `bits`（31 位，天然无脏位）⇒ 老实现的 `& MASK48` 与 64 位漏斗移位都不需要；
3. 求和后不需要 48 位掩码：`X` 高 32 位里 bit16 以上的脏数据乘的是 `L2 = 0xE66D0000`（16 个尾零），
   贡献落在 `2^32` 之外被截断丢掉；
4. `nextInt(10)` 的 `bits % 10 == 0` 用**一次 32 位乘 + 一次比较**：取 `MAGIC = 0x4CCCCCCD`
   （满足 `10*MAGIC ≡ 2 (mod 2^32)`），在真实定义域 `bits ∈ [0, 2^31)` 上
   `(bits*MAGIC) <= 0x19999999 ⟺ bits % 10 == 0`。已对全部 **2^31** 个取值穷举验证（mismatch 0）。
   注意这与「`0xCCCCCCCD` + 循环右移」是两条不同常量路线：前者旋转不可省，换到这个常量后才冗余。

拒绝采样走**行级 warp 均匀门控**（`bits >= 2147483640` 的概率约 3.7e-9），慢路径写成
`__noinline__` 真 CALL：让 ptxas 出真分支而不是谓词化，且慢路径完全不占热路径寄存器。

---

## 4. 关键参数与依据

| 参数 | 值 | 依据 |
|---|---|---|
| `MAX_SIZE_X` / `MAX_SIZE_Z` | 255 / 32 | 语义界 |
| `MAX_FUSED_SIZE_X` | 32 | 窗口位段须落在相邻两字的 64 位内 |
| `MAX_MERGE4_SIZE_X` | 29 | 并集位段宽 `sizeX+3 <= 32`（一次 32 位抽取） |
| `FUSED_K` | 4 | K 扫描实测：K=4 −4.9%、K=5 −4.6%、K=3 +4.2%、K=2 +13.5% |
| `WARPS_PER_BLOCK` | 8（256 线程） | 48 warp/SM ÷ 8 = 6 blocks，占满 |
| `Z_SUB_ROWS` | 2048 | halo 开销 `(sizeZ-1)/2048 <= 0.8%`，同时把 block 数乘上 `ceil(rows/2048)` |
| `RING` / `STRIDE` | 64 / 36 | `RING > MAX_SIZE_Z`；`STRIDE` 取 16 B 对齐以便一条 `STS.128` |
| `POISON_CNT` | `-(1<<30)` | 越界列毒值，闸门永远不会被它触发 |
| `BASE_X_PAD` | `TILE_W` | 无条件 `__ldg` 的越界读上界（`static_assert` 锁死） |

**并行度阈值不再写死，改为运行期按设备派生**（见 §6）。

---

## 5. 主机侧流水线

* **结果池**：一次 `cudaMalloc` 出 baseX/baseZ/results/pos；`cap >= rangeX` 是不变式。
* **越界写的结构性防护**：登记前先 `atomicAdd` 领槽再判界，`pos` 单调递增 ⇒ `[0,cap)` 每槽恰好写一次；
  `pos >= cap` 立即停手，主机丢弃整轮并缩小 chunk 重试 ⇒ 丢弃的命中不会丢结果。
* **排序与写盘**：有序段 + k 路归并，`--sort=on|off`；`on` 时不依赖外部 `sort` 即可逐字节比对。
* **chunk 自适应**：chunk 行数按实测密度推进，缓冲区满就折半重试。
* **进度条**：独立展示线程，读各 worker 已覆盖的候选 Z 行数（每个 chunk launch 前写一次
  原子，全图规模下每卡几十次），按固定间隔刷新单行。输出走 stderr，只在它是终端时默认开启
  （重定向到文件或管道时自动关闭，日志与 benchmark 输出保持纯净），`SLIME_PROGRESS=1` /
  `SLIME_NO_PROGRESS=1` 可强制开关。设备侧零影响：用 `-DSLIME_NO_PROGRESS_CODE` 编译掉整段
  进度代码后，内核 SASS 与开启版**逐字节相同**。

---

## 6. 运行期自适应

程序在 `cudaSetDevice` 之后、真正 launch 之前，按**当前设备**的真实属性派生一组
「全生命周期固定」的常量，并打印出来：

| 量 | 取法 |
|---|---|
| `blocksPerSm` | `cudaOccupancyMaxActiveBlocksPerMultiprocessor(merge4 核, 256 线程, 0)` |
| `slots` | `SM 数 × blocksPerSm`（SM 数用 `cudaGetDevice` + `cudaDeviceGetAttribute` 查**当前设备**） |
| K=4 启用阈值 | `2 × slots` 个 block（两个波形） |
| Z 切分阈值 | `10 × slots` 个 tile |
| `FUSED_K_MIN_COLS` 等价判据 | `tiles4probe >= slots` |
| 计数环 `sizeZ` 上限 | 扫 `sizeZ = 1..32`，取「blocks/SM 仍不低于 sizeZ=1 时」的最大值 |

* 编译期常量退化为**下界/兜底**；`-D` 显式覆盖**优先**（测试旋钮语义不变）。
* 逃生开关 `SLIME_NO_AUTOTUNE=1` 退回编译期常量。
* 测试钩子：`SLIME_FAKE_SM` / `SLIME_FAKE_BPSM` / `SLIME_FAKE_SHARED_PER_SM` / `SLIME_FAKE_SHARED_PER_BLOCK`
  可在没有目标卡时验证派生逻辑。
* `SLIME_PRINT_TUNING=1` 打印每个 worker 自己那张卡派生出的值。

**设备代码同时内嵌 SASS 与本机 compute capability 的 PTX**（`build.sh` 探测 `compute_cap` 生成
`-gencode arch=compute_XX,code=compute_XX`）：驱动优先用精确匹配的 SASS，
架构更新的卡用内嵌 PTX 做 JIT；探测不到 GPU 的构建机退回只出 SASS。

---

## 7. 当前性能

测量环境：RTX 4060 Ti 16 GB、驱动 595.91.07、nvcc 13.4、gcc 15.2、Ubuntu。
测量口径见第 9 节：与频率无关的稳定指标是 `inst/候选`，绝对时间只在同一状态、同一会话内可比。
原始采集放在 `docs/files/collect-headless/`：`collect.sh` 在无头环境完整跑一次的产物，
含身份信息、GPU 画像、速度表、位精确矩阵与 oracle 对拍结果。

各负载速度（`quickbench.sh --full --reps 3`，同一次会话内测得；`fullmap_14G` 与 `readme_3M75`
是同一负载：x,z ∈ [-1875000,1875000]、17×17、thr 55、14.06e12 个候选）：

| 负载 | 区域 | 候选数 | gpu_min | wall_min | B/s_wall |
|---|---|---|---|---|---|
| `rect17_300k` | x,z ∈ [-150000,150000] | 90,000.6 M | 222.87 ms | 327.50 ms | 274.8 |
| `xwide_2Mx32k` | x ∈ [-1000000,1000000], z ∈ [-16000,16000] | 64,002.0 M | 156.77 ms | 258.92 ms | 247.2 |
| `ztall_32kx256k` | x ∈ [-16000,16000], z ∈ [-256000,256000] | 16,384.5 M | 43.74 ms | 135.68 ms | 120.8 |
| `rect32_100k` | x,z ∈ [-50000,50000]，32×32 回退路径 | 10,000.2 M | 31.17 ms | 121.29 ms | 82.4 |
| `fullmap_14G` | x,z ∈ [-1875000,1875000] | 14,062,507.5 M | **35,583.09 ms** | **37,133.41 ms** | **378.7** |

`fullmap_14G` 输出 40,700,773 行（`--sort=on`）：程序自报 GPU 时间 35.58 s、墙钟 37.13 s，
相差 1.55 s，差额是 CSV 格式化与写盘。该次采集的输出目录落在 tmpfs 上，写入不成为瓶颈；
换到磁盘上输出时墙钟会明显变长（同一会话约定下实测 38.75 s），差额同样只在写入侧，
`gpu_min` 与 B/s 的口径不变。

指令级指标不在 `docs/files/collect-headless/` 的覆盖范围内：该次采集运行 ncu 的用户没有读
GPU 性能计数器的权限（`ERR_NVGPUCTRPERM`，见该目录的 `03-ncu-report.log`），
所以下列数据取自同机另一次采集 —— 两次的 `src sha256` 与设备侧 SASS 逐字节相同，
差别只在输出目录落在哪块盘上。要一次性取得全部数据，需要在 root 下重跑 `collect.sh`。

| 指标 | 值 |
|---|---|
| `inst/候选`（`smsp__inst_executed.sum × 32 ÷ 候选数`） | **23.445** |
| kernel 时长（ncu，区域 `X[-131200,131199] Z[-32776,32775] 17×17 thr=60`） | **43.26 ms** |
| SM cycles（同一区域） | 111,141,481 |
| 寄存器 / occupancy | **38 / 99.18%** |
| 管线构成 | ALU 44.3% / FMA 37.1% / uniform 9.3% / LSU 7.0% / XU 2.4% |

正确性结论：位精确矩阵 54/54 PASS（`04-verify-full.txt`）、与独立 oracle 的全量比对
19/19 PASS（`06-quickcheck.txt`）。另有两条检查在该次采集里没有执行：真 JVM 对拍没有找到
JDK，`05-` 未产出；速度表的 `ratio` 列全空，因为未给 `--ref`，而 `quickbench.sh` 在单
二进制模式下不跑交错 A/B、`det` 恒为 `ok`、不做任何一致性比对。

稳态主循环 **91 条/线程/行**（判定 54 + 记账 37），模型换算 23.34 `inst/候选`，
与上表实测 23.445 差 −0.4%。

---

## 8. 已排除的方向

| 方向 | 为什么不行 |
|---|---|
| **X 向增量哈希** | 数学上不成立：乘法在 XOR **之后**，要求「与常数异或」等于「加法」。穷举 1.8e8 组，bits 99.9964% 不一致 |
| **计数环字节打包** | 前提错误：`slot = r mod sizeZ` 只依赖行号 ⇒ 同行 4 个候选**命中同一槽**。修正版省 6 条 LSU 却付 7 条 ALU（ALU 是瓶颈） |
| **行展开（U=2/U=4）** | 「寄存器/占用率 vs ILP」曲线是平的：把 U=4 压回 40 regs / 99.1% 占用率后，ptxas 转头多花 5.2% 指令，cycles 只差 0.07% |
| **`U` 依赖链合并** | ptxas 已融合：源码层改写后 `smsp__inst_executed` 逐位相同 |
| **内联拒绝检查**（去行级均匀门控） | 让 ballot 变发散，指令 **+18.5%** |
| **4 行批次重构** | 指令 4.40×、时长 5.54×：持有 4 行 ballot 要 16 个寄存器，批次状态叠上去后 38 → 56 regs |
| **消除 shared halo 往返** | warp 间无 `__shfl`，只能每 warp 多算 1 格判定（14.25 条）换掉 10 条 ⇒ 净 +4.25 |
| **`__launch_bounds__` 强压寄存器** | 见「行展开」：曲线是平的，且它让 ptxas 多花指令 |
| **提高 occupancy** | 本核 occupancy 弹性仅约 0.19（相对 +20% occupancy 只值 3.5% 时间）⇒ 收益低于同等工作量的砍指令，不作为方向 |

**净结论**：当前结构（每行一次 publish + 一次 barrier）已是局部最优；跨行摊销类改动都会因
寄存器与调度代价而净亏。

---

## 9. 测量方法

* GPU 起始状态统一：**35 °C**，SM 2340 MHz（频由宿主锁定）。
* 计时只用**交错 A/B**；与频率无关的稳定指标是 `inst/候选`，绝对时间跨批次不可比。
* 位精确性是硬门槛：任何改动都要与冻结参考逐字节比对，不通过就没有速度结论（见 `docs/zh-cn/correctness.md`）。
* 编译产物与冻结参考都不入库：CUDA 工具链在 ELF 层面不可复现，提交二进制没有意义。

---

## 10. 目录结构

```
src/slime_main.cu      主程序：三个融合核 + 结果池 + 主机流程 + 运行期自适应
src/slime_circle.cu    二级筛选：圆盘分数（一 warp 一行协作前缀和 + 每候选一线程打分）
src/slime_cmp.c        用户侧距离/计数过滤器
src/slime_hash.cuh     史莱姆判定的唯一实现（供上面三个共用）
tools/oracle.c         从 Java 规范直译的独立 oracle
tools/oracle_pycheck.py 第三个实现，用于校验 oracle 自己
tools/mccrosscheck.sh  把 wiki 原始 Java 交给真 JVM 跑，与程序对拍（需 JDK）
tools/verify.sh        位精确矩阵（需 --ref 指定冻结参考）
tools/fullcheck.sh     穷举 soak（合法域全跑 + 6 种编译变体）
tools/collect.sh       采集性能与正确性数据（画像/耗时/ncu/位精确/对拍），打包成归档
tools/releasecheck.sh  发布前总检查
tools/gpuinfo.sh       GPU 状态快照 / 等温 / 判标准状态
tools/quickbench.sh    速度表（`--ref` 给交错 A/B 比值）
tools/quickcheck.sh    与独立 oracle 的全量输出快速比对
docs/zh-cn/            中文文档（本目录）
docs/en-us/index.md    英文说明，mccrosscheck 从这里逐字抽取 wiki 的 Java 代码块
docs/files/            样例产物
  result.ncu-rep       一份 ncu 原始报告（可用 ncu --import / ncu-ui 打开）
  collect-headless/    无头环境上 collect.sh 的完整一次采集
docs/img/              截图
```
