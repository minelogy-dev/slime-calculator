# Slime Calculator / 史莱姆计算器

基于 CUDA 的 Minecraft 史莱姆区块查找器。仅支持 Java 版。

## 快速开始

```bash
git clone <仓库地址>
cd slime-calculator
./build.sh
./build/slime_main 12345 -10 -10 10 10 5 5 3 output.csv
```

扫描种子 `12345`、区块坐标 `(-10,-10)` 到 `(10,10)` 区域内所有 5×5 区块大小、至少包含 3 个史莱姆区块的矩形区域。

## 编译

```bash
./build.sh
```

三个可执行文件输出到 `build/` 目录。

## 工具

### slime_main

主扫描器，使用 CUDA 在矩形区域内搜索史莱姆区块。

```bash
./build/slime_main <seed> <startX> <startZ> <endX> <endZ> <sizeX> <sizeZ> <threshold> <output.csv>
```

在指定矩形区域 `(startX, startZ)`–`(endX, endZ)` 内搜索所有尺寸为 `sizeX` × `sizeZ`、包含超过 `threshold` 个史莱姆区块的矩形区域。坐标为区块坐标（整数）。沿 Z 方向动态分块以适应 GPU 显存。

选区的四个顶点严格保证在搜索范围内。但为了优化 GPU 显存对齐，内部实际使用的区域会向 256 取整，可能会略微扩大扫描范围。

### 性能

在 RTX 4060 Ti 16GB 上，以下命令约需 10 分钟：

```bash
./build/slime_main 114514 -1875000 -1875000 1875000 1875000 17 17 55 output.csv
```

输出（世界/方块坐标）：

```
x,z,slime_count
```

### slime_circle

基于圆形区域的二级筛选。

```bash
./build/slime_circle <input_csv> <radius> <sizeX> <sizeZ> <seed> <output_csv> <threshold>
```

读取 `slime_main` 的输出，查找其半径为 `radius` 的圆形区域内包含超过 `threshold` 个史莱姆区块的子矩形。

### slime_cmp

基于距离的后置筛选。

```bash
./build/slime_cmp <input_csv> <distance> <threshold> <output_csv>
```

筛选离 `(0,0)` 的区块距离 ≤ `distance` 且史莱姆数量 ≥ `threshold` 的记录。

输出：

```
x,z,slime_count,distance
```

### test

一个论证工具，用于证明 Java 的 `nextInt` 拒绝采样最多只需要一次迭代。枚举所有坏种子（8 个可能的 `u` 值 × 2¹⁷ 个低比特组合），然后对种子图做 DFS 计算最长连续坏种子链。结果为 1，说明没有坏种子会生成另一个坏种子，因此 CUDA kernel 中的 `do-while` 可以安全地替换为 `if`。

```bash
./build/test
```

## 完整流程

```
slime_main  <seed> <area> <rect> <threshold>  → candidates.csv
slime_circle candidates.csv <radius> <rect> <seed> <threshold>  → circles.csv
slime_cmp   circles.csv <distance> <threshold>  → result.csv
```
