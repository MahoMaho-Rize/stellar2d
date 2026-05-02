# MESA OPAL / Ferguson Opacity 工作流

这份文档说明如何从**本机安装的 MESA** 提取 `kap_data`，转换成 `stellar2d` 可直接读取的 `KAPv1` 二进制，并在运行时通过 `--kap` 接到求解器。

当前结论先写在前面:

- `stellar2d` **不会**在运行时直接链接 MESA `kap_lib`。
- 现在采用的是更稳妥的路线:
  MESA ASCII 表 `->` 预处理成 `KAPv1` `->` Stellar2D/GPU 直接读表。
- 这一套目前**最完整接到的是 `radial1d`**。
- 2D 其他 solver 还没有全面切过去,所以不要默认 `--kap` 会影响所有求解器。

## 1. 前置条件

你需要满足下面几项:

- 本机有 MESA 安装目录
- 目录里存在 `data/kap_data/`
- 仓库已经编译为 `USE_GPU=ON`
- 有一套能跑 `scripts/convert_mesa_kap.py` 的 Python,并且带 `numpy`

本仓库的脚本现在会按下面顺序自动找 MESA:

1. 环境变量 `MESA_DIR`
2. `~/MESA/mesa-*`
3. `~/MESA`
4. `~/mesa`
5. `~/mesa-ref`

自动检测逻辑在 [scripts/mesa_local.py](/Users/kubo/Documents/GitHub/stellar2d/scripts/mesa_local.py:1)。

## 2. 把 MESA ASCII 表转换成 KAPv1

### 2.1 默认用法

如果 `MESA_DIR` 已经设置好,或者你的 MESA 在 `~/MESA/mesa-*` 下面:

```bash
python3 scripts/convert_mesa_kap.py
```

它会自动寻找本机的 `kap_data`，并把结果写到:

```text
third_party/mesa_kap/
```

输出文件名形如:

```text
gs98_z0.02.kapbin
lowT_fa05_gs98_z0.02.kapbin
```

### 2.2 只转换常用 solar 一组

最常用的是高温 `gs98` 加低温 `lowT_fa05_gs98`:

```bash
python3 scripts/convert_mesa_kap.py --families gs98 lowT_fa05_gs98
```

这通常已经够 `radial1d` 的 solar / pre-MS 工作流使用。

### 2.3 显式指定 MESA 路径

如果自动检测不到:

```bash
python3 scripts/convert_mesa_kap.py \
  --src "$MESA_DIR/data/kap_data" \
  --dst third_party/mesa_kap \
  --families gs98 lowT_fa05_gs98
```

## 3. 如果系统 Python 没有 `numpy`

这台机器上系统 `python3` 可能没有 `numpy`。这种情况下直接换用 bundled runtime:

```bash
/Users/kubo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/convert_mesa_kap.py --families gs98 lowT_fa05_gs98
```

同理,检查与 round-trip 验证也可以这样跑:

```bash
/Users/kubo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/inspect_mesa_kap.py

/Users/kubo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/verify_mesa_kapbin.py
```

## 4. 检查转换结果

### 4.1 快速查看某一组 ASCII 表

```bash
python3 scripts/inspect_mesa_kap.py
python3 scripts/inspect_mesa_kap.py "$MESA_DIR/data/kap_data" gs98_z0.02
```

这个脚本会打印:

- family / `Z`
- 各个 `X` slice
- `logT` / `logR` 网格范围
- 几个典型 `rho, T` 点上的近似 `kappa`

### 4.2 验证 `.kapbin` 与原始 ASCII 完全一致

```bash
python3 scripts/verify_mesa_kapbin.py
```

它会重新解析 MESA ASCII 和 `.kapbin`，逐格比较 `log10(kappa)`。

## 5. 运行时如何启用 opacity table

运行时入口在 [src/main.cpp](/Users/kubo/Documents/GitHub/stellar2d/src/main.cpp:612)。

当前运行时逻辑是:

- `--kap` 开启表 opacity
- 从 `third_party/mesa_kap/` 读两张表
- 高温表默认 family = `gs98`
- 低温表默认 family = `lowT_fa05_gs98`
- 默认 `Z = 0.02`
- 在 `logT` 的重叠区间做 stitched 切换

常用命令:

```bash
./build/stellar2d \
  --solver radial1d \
  --test lane_emden \
  --eos helmholtz \
  --kap --kap-Z 0.02 \
  --radiation
```

如果你还要从 MESA profile 初始化:

```bash
./build/stellar2d \
  --solver radial1d \
  --eos helmholtz \
  --kap --kap-Z 0.02 \
  --ic-mesa path/to/profile.ic \
  --ic-mesa-seed-T \
  --radiation --mlt --rich-profile
```

## 6. 相关运行时参数

最常用的是这些:

- `--kap`: 开启 tabulated opacity
- `--kap-Z <float>`: 选择 `Z`
- `--kap-highT <family>`: 高温 family,默认 `gs98`
- `--kap-lowT <family>`: 低温 family,默认 `lowT_fa05_gs98`
- `--kap-dir <path>`: `.kapbin` 目录,默认 `third_party/mesa_kap`
- `--kap-logT-lo-end <float>`: seam 低端,默认 `3.9`
- `--kap-logT-hi-start <float>`: seam 高端,默认 `4.1`

这些参数的解析在 [src/main.cpp](/Users/kubo/Documents/GitHub/stellar2d/src/main.cpp:260) 附近。

## 7. 现在到底“实现了什么”

目前已经落地的是:

- MESA Type-1 `kap_data/*.data` 解析
- 按 `(family, Z)` 把不同 `X` slice 堆成 3D 表
- 写成 `KAPv1` 二进制
- GPU 侧 trilinear interpolation
- `radial1d` 里接到 radiation / MLT / rich diagnostics

对应实现:

- 解析/转换: [scripts/mesa_kap.py](/Users/kubo/Documents/GitHub/stellar2d/scripts/mesa_kap.py:1), [scripts/convert_mesa_kap.py](/Users/kubo/Documents/GitHub/stellar2d/scripts/convert_mesa_kap.py:1)
- 运行时表加载: [src/physics/opacity_table.cuh](/Users/kubo/Documents/GitHub/stellar2d/src/physics/opacity_table.cuh:1), [src/physics/opacity_table.cu](/Users/kubo/Documents/GitHub/stellar2d/src/physics/opacity_table.cu:1)
- `radial1d` 接线: [src/gpu/radial1d_solver.cuh](/Users/kubo/Documents/GitHub/stellar2d/src/gpu/radial1d_solver.cuh:112), [src/gpu/radial1d_solver.cu](/Users/kubo/Documents/GitHub/stellar2d/src/gpu/radial1d_solver.cu:103)

## 8. 当前限制

- 只支持 **MESA Type-1** opacity ASCII
- `*_co` 这类 C/O enhanced form=2 还没接
- `kR_*` conductivity 表没有走这条路径
- 当前最完整的消费方是 `radial1d`
- 2D solver 还没有统一迁到这套 EOS + opacity 表工作流

## 9. 推荐最小工作流

如果目标只是“把本机 MESA 的 OPAL/Ferguson 接到 Stellar2D 里跑起来”，推荐顺序是:

1. 生成 `.kapbin`
2. 跑 `inspect_mesa_kap.py`
3. 跑 `verify_mesa_kapbin.py`
4. 用 `radial1d + --kap + --radiation`
5. 再逐步叠加 `--eos helmholtz`、`--ic-mesa`、`--mlt`

最小命令集:

```bash
/Users/kubo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/convert_mesa_kap.py --families gs98 lowT_fa05_gs98

./build/stellar2d \
  --solver radial1d \
  --test lane_emden \
  --eos helmholtz \
  --kap --kap-Z 0.02 \
  --radiation
```
