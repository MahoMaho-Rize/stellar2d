# Stellar2D 全体手册

这是一份面向“第一次接手这个仓库”的总览手册。它不替代设计文档，而是告诉你:

- 仓库里现在有哪些 solver
- 各自适用什么问题
- 运行一次 simulation 的最短路径是什么
- MESA / EOS / opacity 相关功能现在接到了哪里

更细的数学与历史决策，继续看各专题文档。

## 1. 仓库当前结构

最重要的目录:

- `src/`: C++ / CUDA 主代码
- `tests/`: 单测与 GPU 验证
- `scripts/`: 预处理、后处理、MESA 桥接脚本
- `docs/`: 设计文档、问题记录、专题手册
- `third_party/helmholtz/`: Helmholtz EOS 表
- `third_party/mesa_kap/`: MESA opacity 二进制缓存

入口程序:

- 可执行文件: `build/stellar2d`
- 包装脚本: `stellar2d.py`

## 2. solver 总览

这仓库不是单一求解器，而是一组并存的 solver 资产。

需要特别记住:

- **不要覆盖旧 solver 资产**
- 不同 solver 的物理/数值域不同
- “新的实验功能”优先加在新文件或新路径，而不是把旧 baseline 改坏

当前最常见的使用建议:

- `radial1d`: 1D 径向恒星结构 / 脉动 / MESA 对比
- `cart_ale2`: 2D Cartesian 可压缩对流、KH、周期边界实验
- `lowmach`: 低马赫隐式极坐标路线
- `fas` / `simple` / `projection`: 现有极坐标低马赫/隐式家族 baseline
- `pseudo_spectral`: 2D 不可压 Navier-Stokes baseline

更完整的 solver 状态见 [README.md](/Users/kubo/Documents/GitHub/stellar2d/README.md:1) 和 `AGENTS.md`。

## 3. 最常见的三条工作流

### 3.1 普通编译运行

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON
make -j$(nproc)
```

运行示例:

```bash
./stellar2d --solver radial1d --test lane_emden --nr 256 --tend 1
```

### 3.2 MESA profile -> radial1d

流程是:

1. 用 MESA 产生 `profile*.data`
2. 用 `scripts/convert_mesa_ic.py` 转成仓库能读的扁平 IC
3. 用 `radial1d --ic-mesa ...` 启动

典型命令:

```bash
python3 scripts/convert_mesa_ic.py LOGS/profile10.data out.ic

./build/stellar2d \
  --solver radial1d \
  --eos helmholtz \
  --ic-mesa out.ic \
  --ic-mesa-seed-T
```

### 3.3 本机 MESA opacity -> Stellar2D

流程是:

1. 从本机 MESA 的 `data/kap_data` 生成 `.kapbin`
2. 运行时加 `--kap`

详细步骤见:

- [mesa_opacity_workflow.md](/Users/kubo/Documents/GitHub/stellar2d/docs/mesa_opacity_workflow.md:1)

## 4. EOS / opacity / microphysics 现状

现在的状态不是“只会 ideal gas”了。

### EOS

见 [src/eos.h](/Users/kubo/Documents/GitHub/stellar2d/src/eos.h:1)。

当前支持:

- `ideal`
- `ideal_rad`
- `pre_ms`
- `helmholtz`

其中:

- `helmholtz` 需要 `third_party/helmholtz/helm_table.bin`
- 目前最完整地接到 `radial1d`

### Opacity

当前不是运行时直连 MESA，而是:

MESA ASCII `->` `KAPv1` binary `->` GPU lookup

主要文件:

- [src/physics/opacity_table.cuh](/Users/kubo/Documents/GitHub/stellar2d/src/physics/opacity_table.cuh:1)
- [src/physics/opacity_table.cu](/Users/kubo/Documents/GitHub/stellar2d/src/physics/opacity_table.cu:1)

### Nuclear burning

当前已有简化 pp-chain:

- [src/physics/nuclear_pp.h](/Users/kubo/Documents/GitHub/stellar2d/src/physics/nuclear_pp.h:1)

### Radiation / MLT

当前 `radial1d` 已接:

- radiation diffusion
- implicit BE radiation solve
- MLT diagnostics / conductivity

主要代码在:

- [src/gpu/radial1d_solver.cu](/Users/kubo/Documents/GitHub/stellar2d/src/gpu/radial1d_solver.cu:676)
- [src/gpu/radial1d_implicit.cu](/Users/kubo/Documents/GitHub/stellar2d/src/gpu/radial1d_implicit.cu:1)

## 5. 当前最稳的 MESA 相关组合

如果你想走“最少坑”的路线，优先用这一组:

- solver: `radial1d`
- eos: `helmholtz`
- opacity: `--kap`
- radiation: `--radiation`
- MESA IC: `--ic-mesa`

也就是:

```bash
./build/stellar2d \
  --solver radial1d \
  --eos helmholtz \
  --kap --kap-Z 0.02 \
  --radiation \
  --ic-mesa out.ic \
  --ic-mesa-seed-T
```

这是现在 MESA / EOS / opacity / radiation 接线最完整的一条路径。

## 6. 文档导航

如果你只想快速找到对应文档，按这个看:

- 总体运行方式: [README.md](/Users/kubo/Documents/GitHub/stellar2d/README.md:1)
- MESA opacity 接入: [mesa_opacity_workflow.md](/Users/kubo/Documents/GitHub/stellar2d/docs/mesa_opacity_workflow.md:1)
- `radial1d` 与 MESA 对比: [radial1d_mesa_tier2_pk_2026-05-03.md](/Users/kubo/Documents/GitHub/stellar2d/docs/radial1d_mesa_tier2_pk_2026-05-03.md:1)
- `radial1d` 演进记录: [radial1d_ignition_journey_2026-05-02.md](/Users/kubo/Documents/GitHub/stellar2d/docs/radial1d_ignition_journey_2026-05-02.md:1)
- 数学与离散式: [equations.md](/Users/kubo/Documents/GitHub/stellar2d/docs/equations.md:1)
- 已知坑点: [pitfalls.md](/Users/kubo/Documents/GitHub/stellar2d/docs/pitfalls.md:1)
- `cart_ale2` 设计: [cart_ale2_design.md](/Users/kubo/Documents/GitHub/stellar2d/docs/cart_ale2_design.md:1)
- 伪谱法: [pseudo_spectral_design_2026-05-01.md](/Users/kubo/Documents/GitHub/stellar2d/docs/pseudo_spectral_design_2026-05-01.md:1)

## 7. 当前边界

现在最好不要误解成“全仓库都已经 MESA 化”。

真实状态是:

- `radial1d` 已经有比较完整的 MESA / Helm / opacity / radiation 路线
- `cart_ale2` 仍以流体动力学 baseline 为主
- 极坐标 2D solver 还没有统一迁到同一套 microphysics
- MESA opacity 的表驱动路线已经落地，但主要消费方还是 `radial1d`

## 8. 建议的阅读顺序

第一次接手时，推荐顺序:

1. 先读 [README.md](/Users/kubo/Documents/GitHub/stellar2d/README.md:1)
2. 再读这份 `manual.md`
3. 如果要碰 MESA opacity，读 [mesa_opacity_workflow.md](/Users/kubo/Documents/GitHub/stellar2d/docs/mesa_opacity_workflow.md:1)
4. 如果要碰 `radial1d`，读 `radial1d_*` 系列文档
5. 如果要碰 2D ALE，对应读 `cart_ale*` 文档
