# 融合核「不重不漏」的证明

证明的是**目标 A：工具输出不重不漏**。
判定位运算本身与 Java 的逐位等价性由 `src/slime_hash.cuh` 的注释给出（五处变形，
其中 `mod 10` 的常量已对全部 2^31 个取值穷举验证），外部权威由 `tools/mccrosscheck.sh`
（真 JVM 跑 wiki 原始 Java）对拍 —— 两者合起来构成「输出 = 恰好所有满足条件的矩形各一次」的完整论证。

记号里的行号对应 `src/slime_main.cu` 的现行版本（v3.0.0）。

---

## 0. 结论

**定理（不重不漏）** 设请求区域为 `[startX,endX] × [startZ,endZ]`、矩形 `sizeX × sizeZ`、
阈值 `threshold ≥ 0`。程序输出的行集合恰为

```
{ (x,z) : x = 16·c, z = 16·t,  c ∈ [startX', endX'−sizeX+1],  t ∈ Z_scan,
          Σ_{i=c}^{c+sizeX−1} Σ_{j=t}^{t+sizeZ−1} v(i,j) ≥ threshold }
```

其中 `v(i,j)` 是 `docs/en-us/index.md` 那段 Java 的判定（由 `src/slime_hash.cuh` 的变形说明证明与实现一致），
`[startX',endX']` 是文档规定的 256 对齐后区间，`Z_scan` 是该区间内的候选行 —— 每行**恰好一次**。

**依赖的假设只有一个**：编译器/驱动保持 C++ 语义。按约定，这属于编译器厂商的义务，
代码只需保证不含 UB —— 见 `docs/zh-cn/correctness.md` §4 的静态逐条核对。

---

## 1. 记号

| 符号 | 含义 |
|---|---|
| `v(i,j)` | 区块 `(i,j)` 是否为史莱姆区块（`{0,1}`） |
| `X_j(c)` | 第 `j` 行上、以列 `c` 为左端点的 X 窗口和 `Σ_{i=c}^{c+sizeX−1} v(i,j)` |
| `W(c,t)` | 候选 `(c,t)` 的矩形和 `Σ_{j=t}^{t+sizeZ−1} X_j(c)` |
| `K`, `WPB=8`, `TILE_W = WPB·32·K` | 每线程跨步列数、每块 warp 数、每块覆盖的输入列数 |
| `outW = TILE_W − sizeX + 1` | 每个 tile 产出的候选列数 |
| `wordBase = warp·K`，`cl(k,L) = (wordBase+k)·32 + L` | 线程在块内的输入列 |
| `RING = 64` | 共享环行数（`static_assert(RING > MAX_SIZE_Z)`，`MAX_SIZE_Z = 32`） |

---

## 2. 引理

### L1（ballot 位 ↔ 格子）

`ballots[k] = __ballot_sync(0xFFFFFFFF, isSlimeChunk(bx[k], bz))`（第 455 行）的第 `b` 位
等于 `v(colBase + cl(k,b), r)`，其中 `bz` 是第 `r` 行的 Z 基值。

**证**：`__ballot_sync` 把 lane `b` 的谓词放到第 `b` 位；lane `b` 的 `bx[k]` 正是
`d_baseX[gcol]`，`gcol = colOffset + colBase + cl(k,b)`，而 `colOffset = 0`（融合路径）。
`isSlimeChunk(bx[k],bz)` 与 `v` 一致由 `src/slime_hash.cuh` 的变形说明给出。∎

### L2（X 窗口提取）

对 lane `L`、字 `k`，`__popc(__funnelshift_r(w_k, hi, L) & xmask) = X_r(cl(k,L))`，
只要 `cl(k,L) + sizeX − 1 < TILE_W`（即窗口不越过本 block 的第 `TILE_W−1` 列）。

**证**：`__funnelshift_r(lo,hi,s)` 取 64 位拼接 `(hi:lo)` 的第 `[s, s+31]` 位
（CUDA 文档语义）。取 `lo=w_k`、`hi=w_{k+1}`（`k<K−1` 时 `hi` 就是本 warp 的 `ballots[k+1]`，
第 460 行；`k=K−1` 时是 shared 里的 `row[wordBase+K]`，由屏障保证可见）。于是结果是
第 `[L, L+31]` 位，与 `xmask = (1<<sizeX)−1` 相与后剩 `[L, L+sizeX−1]`，
由 L1 就是 `Σ_{i=cl}^{cl+sizeX−1} v(i,r) = X_r(cl)`。∎

> `hi` 的正确性对 `k=K−1` 依赖「下一个 warp 已把本行的 `word 0` 写进 shared」——
> 由第 458 行的 `__syncthreads_or` 保证；最后一个 warp 读到哨兵 0（第 250 行预置），
> 而会读它的 lane 一律被 `outA` 排除（见 L5）。

### L2b（填充表不会污染有效窗口）

`d_baseX` 在 `width` 之后有 `BASE_X_PAD` 个 0（第 795 行 `cudaMemset`），越界列因此可以
无条件 `__ldg`。这类列产生的 ballot 位**永远不会进入有效候选的窗口**：有效候选满足
`clLocal < validRangeX = width − sizeX + 1`（L5），故其窗口 `[c, c+sizeX−1] ⊆ [0, width−1]`。∎

### L3（Z 滑动窗口不变式，归纳）

定义 `A_r = Σ_{j=max(0, r−sizeZ+1)}^{r} X_j(cl)`。则在处理完第 `r` 行之后

```
cnt = A_r        （对每个有效 lane 的每个 k）
```

**证**（对 `r` 归纳）：

* 前置段 `r ∈ [0, peel)`（第 387 行）只做 `cnt += X_r`，故 `cnt = Σ_{j=0}^{r} X_j = A_r`
  （此时 `r ≤ sizeZ−2`，`max(0, r−sizeZ+1) = 0`）。∎
* 主段 `r ∈ [peel, subInputRows)`（第 422 行）做
  `cnt += X_r`（第 460 行）再 `cnt −= X_{r−sizeZ}`（第 466 行）：

  ```
  cnt_r = A_{r−1} + X_r − X_{r−sizeZ}
  ```
  当 `r ≥ sizeZ` 时 `A_{r−1} = Σ_{j=r−sizeZ}^{r−1} X_j`，相减相加即 `Σ_{j=r−sizeZ+1}^{r} X_j = A_r`；
  当 `r = sizeZ−1` 时 `A_{r−1} = Σ_{j=0}^{sizeZ−2} X_j`、`X_{−1} = 0`（见 L4），同样得 `A_r`。∎

发射时 `z = subBaseZ + r − (sizeZ−1)`（第 469 行）正是该窗口的顶行，
于是 `cnt = W(cl, z)`。∎

### L4（环不冲突、旧行读到的确实是 `sizeZ` 行前那一行）

槽位 `= (r mod 64)`。第 `r` 行写槽 `r mod 64`，第 `r` 行读的旧槽是 `(r−sizeZ) mod 64`。

* **不撞车**：`r ≡ r−sizeZ (mod 64) ⟺ sizeZ ≡ 0 (mod 64)`，而 `sizeZ ≤ 32`，故不撞。∎
* **未被覆盖**：两次访问之间写入的行是 `r−sizeZ+1 … r`，占 `sizeZ ≤ 32 < 64` 个**互不相同**的槽，
  都不等于 `(r−sizeZ) mod 64`。∎
* **`r < sizeZ` 时读到 0**：第 381 行把整环清零，此时 `(r−sizeZ) mod 64 ∈ [64−sizeZ, 63]`，
  而这些槽要到第 `r+64−sizeZ ≥ 32 > r` 行才会被写，故此刻读到的是 0。∎

### L5（候选列恰好覆盖一次）

设 `numTiles = ⌈rangeX / outW⌉`（第 979 行）。tile `b` 覆盖输入列
`[b·outW, b·outW + TILE_W)`，产出候选列 `[b·outW, b·outW + outW)`。

* 各 tile 的候选列区间**相邻且不重叠**，并集为 `[0, numTiles·outW)`；
  再与 `outA` 的 `clLocal < validRangeX` 相交，并集恰为 `[0, rangeX)`。∎
* tile 内每个候选列由唯一的 `(k, L)` 产生（`cl(k,L) = (wordBase+k)·32 + L` 在
  `[0, TILE_W)` 上一一对应），且 `outA` 恰好筛掉 `cl ≥ outW` 的 lane。∎
* 有效候选的窗口不越过本 block 的列范围：`cl ≤ outW−1 = TILE_W−sizeX`，
  故 `cl+sizeX−1 ≤ TILE_W−1`，L2 的适用条件成立；需要跨 warp 读 halo 的那些 lane
  读到的是下一个 warp 已写好的字（屏障保证）。∎

### L6（候选行恰好覆盖一次）

`subStart = blockIdx.y · zSubRows`、`subRows = min(zSubRows, chunkCandRows − subStart)`
（第 343–344 行），各 `blockIdx.y` 的 `[subStart, subStart+subRows)` **相邻不重叠**，
并集为 `[0, chunkCandRows)`；`subRows ≤ 0` 的块不发射。∎

对 `subRows ≥ 1`：`subInputRows = subRows + sizeZ − 1 ≥ sizeZ`，故 `peel = sizeZ−1`，
主段 `r ∈ [sizeZ−1, subInputRows)` 共 `subRows` 次发射，
`z = subBaseZ + r − (sizeZ−1)` 取遍 `[subBaseZ, subBaseZ+subRows−1]`。∎

### L7（chunk 推进与溢出重试不重不漏）

* **推进**：`z` 每次增加 `scanned = rows`（第 1047 行），故 `[candZ0, candZ1]` 被连续覆盖、无重叠。∎
* **重试**：若某次 launch 返回 `n > cap`，程序 `continue`（第 1033 行）——
  既不 `drain` 也不计入 `totalHits`，而 `d_pos` 在每次 launch 前都被 `cudaMemset` 归零
  （第 1023 行），于是被丢弃的那一轮结果**不会进入输出**；同一段 `z` 用更小的 `rows` 重跑，
  成功后才推进。故重试既不产生重复也不产生遗漏。∎

### L8（发射门与毒值）

* 有效候选列：`outA[k]` 真 ⟹ `xmaskLane[k] = xmask`（第 367 行），由 L2/L3 得
  `cnt[k] = W(cl,z) ≥ 0`，登记条件 `cnt[k] ≥ threshold` 即真值判定。∎
* 越界列：`xmaskLane[k] = 0`，加减项恒为 0，`cnt[k]` 保持初值 `POISON_CNT = −2³⁰`（第 368 行）；
  登记块内仍逐 k 写 `outA[k] && cnt[k] ≥ threshold`（第 494 行），故越界列**永不登记**。∎
* 快路径闸门 `mx = max_k cnt[k] ≥ threshold`（第 489 行）只是过滤：任一有效列命中 ⟹ `mx` 命中，
  故不漏；进入后逐 k 复核，故不多。`threshold < 0` 已在 `main` 归一化为 0（第 375 行），
  毒值不可能误触发。∎

### L9（登记的唯一性与上界）

`pos = atomicAdd(d_pos,1)` 是严格递增的全局序号，每个命中领到唯一 `pos`；
`pos ≥ cap` 时置 `stop` 并 `break`，**不写** `d_results`（第 495–498 行），
故 `d_results[0..cap)` 无越界写、无覆盖写。缓冲区满的那一轮由 L7 整体作废重试。∎

---

## 3. 主定理的证明

1. 由 L5、L6、L7，每个候选 `(c,t)` 恰好被发射路径考虑一次；
2. 由 L1、L2、L2b、L3、L4，被考虑时 `cnt = W(c,t)`；
3. 由 L8，登记当且仅当 `W(c,t) ≥ threshold`；
4. 由 L9，每次登记写入一个唯一槽位，且缓冲区满时整轮作废重试（L7），故最终输出
   恰好是这些候选各一次。
5. `x = (startX + colBase + cl)·16`（第 499 行）与 `z·16`，与接口约定一致（`docs/zh-cn/correctness.md` §3）。∎

---

## 4. 回退核（`sizeX > 32`）的证明

回退路径是两个核：`computeRowPrefixSum`（整行前缀和）+ `slidingWindowOutputKernel`（滑窗取差）。
结构与融合核不同，但同样可以逐条证。

### L10（整行前缀和 = 真实累加和模 256）

`computeRowPrefixSum` 里，第 `row` 行的 `psRow[c] = (Σ_{i=0}^{c} v(i,row)) mod 256`。

**证**（对 warp 内的分段归纳）：循环按 `col = lane + 32k` 推进，第 `k` 段处理列
`[32k, 32k+31]`。

* 段内：`val` 初值为 `v(lane+32k, row)`；`offset = 1,2,4,8,16` 的 `__shfl_up_sync`
  是标准的 Hillis–Steele 包含式扫描，五次之后 lane `L` 持有 `Σ_{i=32k}^{32k+L} v(i,row)`。∎
* 跨段：`prefix` 在段末取 `__shfl_sync(mask, val, 31) = Σ_{i=32k}^{32k+31} v(i,row)`，
  加上它即得整行累加和；归纳假设给出 `prefix = Σ_{i<32k}`。∎
* 截断：`(uint8_t)` 按 2⁸ 取模，标准保证。∎

> 全掩码 `__shfl_*` 要求 32 个 lane 全部活跃 —— 这正是 `width` 必须按
> `WIDTH_ALIGN = 256`（32 的倍数）向上对齐的原因（`main` 中强制），
> 否则最后一段会有 lane 不进入循环体。

### L11（X 窗口 = 两个前缀之差）

`slidingWindowOutputKernel` 中 `val = (uint8_t)(pR[0] - (pL[0] & lmask))`，
其中 `pR` 指向 `ps[row][globalJ + sizeX − 1]`、`pL` 指向 `ps[row][globalJ − 1]`。

**证**：`globalJ > 0` 时 `lmask = 0xFF`，由 L10

```
val ≡ ps[globalJ+sizeX−1] − ps[globalJ−1] ≡ Σ_{i=globalJ}^{globalJ+sizeX−1} v(i,row)  (mod 256)
```

而该和落在 `[0, sizeX] ⊆ [0, 255]`（`sizeX ≤ MAX_SIZE_X = 255`），唯一确定其模 256 值，
故 `val` 就是真实窗口和。`globalJ = 0` 时 `lmask = 0`，左端点按 0 处理，
`val = ps[sizeX−1] = Σ_{i=0}^{sizeX−1}`，同样正确。∎

### L12（Z 滑动窗口不变式）

初值 `sum = Σ_{row=0}^{sizeZ−1} val_row = W(globalJ, baseZ)`（第 529–534 行）；
主循环做 `sum -= winVals[ringIdx]; sum += newVal`（第 553–557 行），
`winVals` 是以 `sizeZ` 为周期的环、`ringIdx` 依次取 `0..sizeZ−1`。
与 L3 完全同构的归纳给出：处理第 `i` 步后 `sum = Σ_{j=baseZ+i}^{baseZ+i+sizeZ−1} X_j(globalJ)`；
发射时 `z = baseZ + i`（第 538 行）正是窗口顶行，故 `sum = W(globalJ, z)`。∎

### L13（列批量与行批量不重不漏）

* **列**：主机循环 `for (c = 0; c < rangeX; )` 每批宽度 `cols`、区间 `[c, c+cols)`，
  `c += cols`（第 1063 行），各批相邻不重叠、并集 `[0, rangeX)`；
  核内 `j < validRangeX = cols`（第 527 行）与 `globalJ = colOffset + j` 一一对应。∎
* **行**：`i ∈ [0, blockHeight − sizeZ]` 与 `z = baseZ + i` 一一对应（第 537 行）；
  `z ∈ [batchValidStartZ, batchValidEndZ]` 过滤掉跨大块 halo 的部分，
  而 `validStartZ / validEndZ` 在各 Z 大块之间相邻不重叠（`step` 的构造保证）。∎
* **登记与重试**：与 L9/L7 同一套论证（`atomicAdd` 唯一槽位、`pos < cap` 先判界、
  溢出整批作废重试）。∎

---

## 5. 这个证明覆盖什么、不覆盖什么

**覆盖（两个计算核都在内）**：
* 融合核（`sizeX ≤ 32`）：列/行/块/chunk 四级切分不重不漏（L5–L7）；
* 融合核滑动窗口不变式：含前置段、环预清零、环不冲突（L3、L4）；
* 回退核（`sizeX > 32`）：整行前缀和（L10）、X 窗口取差（L11）、
  Z 滑动窗口（L12）、列/行批量切分（L13）；
* 发射门、毒值、越界列排除（L8）；结果登记原子性与 cap 上界（L9）；
* 溢出重试与自适应 chunk 推进（L7、L13）。

**不覆盖**（各自有独立证据）：
* `v` 本身与 Java 的等价性 —— `src/slime_hash.cuh` 的变形说明 + `tools/mccrosscheck.sh` 的真 JVM 对拍；
* 主机侧区间补齐、`computeBases`、`H_max`/`step` 的算术 —— 逐条断言 +
  与真 JVM 对拍（`tools/mccrosscheck.sh`）+ P2 的几何边界扫描；
* C++ → SASS 的保义性 —— 按约定属编译器义务，前提是 0 UB
  （`docs/zh-cn/correctness.md` §4 的静态逐条核对）；
* `--sort` 输出路径（结果池、k 路归并）—— 只保证多重集，不保证顺序（明确排除）。

**一句话**：**两个计算核的「不重不漏」都是证明的**，不是"测了很多没出错"；
剩下的未证明部分是主机侧的算术与 I/O，由断言、外部对拍与矩阵测试覆盖。
