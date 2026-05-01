# Project-specific guidance for Claude Code

## 不可覆盖的求解器资产

本仓库保留了多个阶段性求解器,**即使它们在某个测试上不稳/不适合当前目标,也不要"原地修复"覆盖它们**。
保留理由:每个求解器背后的物理/数值方法都是正确的,只是适用域不同,未来做对比/回归/教学都需要它们作为基线。

已有资产清单(只读,不要覆盖):

| 求解器 | 位置 | 状态 | 保留理由 |
|---|---|---|---|
| `strang` | `src/gpu/strang_solver.{cu,cuh}` | ✅ stable | 全 2D 显式 Strang split HLLC+MUSCL,standard 基线 |
| `fas` / `simple` / `projection` | `src/gpu/{fas,simple,projection}_solver.*` | ✅ stable | 低马赫隐式方法族,各自有适用区间 |
| `lowmach` | `src/gpu/lowmach_solver.*` | ✅ stable | JFNK 低马赫 |
| `radial1d` | `src/gpu/radial1d_solver.*` | ✅ stable | MESA-style 径向 1D,用于 ALE 回归对比 |
| `wb2d` | `src/gpu/wb2d_solver.*` | ⚠️ 扰动 t≈2 死 | 物理无错,是 well-balanced 欧拉流族的合理实现;未来需要时改进稳定器,不要拆 |
| `ale2d` | `src/gpu/ale2d_solver.*` | ⚠️ 轴对称 hoop bug | 轴对称 Caramana,文档里 `docs/ale_hoop_stress_fix.md` 已说明该怎么补丁。未来继续做球对称时直接上此文件,不要在新 ALE 里重做 |
| `cart_lag` | `src/gpu/cart_lag_solver.*` | ⚠️ HSE dt 退化 | 纯 Lagrangian,hourglass-mode 导致长时间 HSE 退化;文档 `docs/ale_rezone_design.md` 分析了原因。是"真 ALE"的 Lagrangian 相参考实现,不要覆盖 |
| `cart_ale` | `src/gpu/cart_ale_solver.*` | ✅ stable | Cartesian 2D ALE(Caramana Lagrangian + Eulerian rezone + swept remap)。`--remap-order 1|2`:1=donor-cell,2=MUSCL-minmod(默认,Kucharik-Shashkov) |
| `cart_ale2` | `src/gpu/cart_ale2_solver.*` | ✅ stable | cart_ale 的下一代:周期 BC + PPM-in-remap(Colella-Woodward)。`--bc-x/--bc-y periodic`,`--ppm` 启用。KH shear test 标配 |
| `pseudo_spectral` | `src/gpu/pseudo_spectral_solver.*` + `pseudo_spectral_kernels.cu` | ✅ stable | 2D 不可压缩 Navier-Stokes 伪谱法(cuFFT + IFRK3 + skew-symmetric + 圆形 2/3 dealias)。双周期域,涡度-流函数形式。目前只支持 `--test kh_shear`。详见 `docs/pseudo_spectral_design_2026-05-01.md` |

**"做真正的 ALE"的正确路径**:开**新文件 + 新求解器 struct**(例如 `cart_ale_solver.{cu,cuh}` 配 `cart_ale_kernels.cu`),可以**参考**`cart_lag_*` 里的 Lagrangian kernel,但不要修改它们。CMakeLists.txt 里新增编译条目即可。

**做伪谱法扩展**(Boussinesq / MHD / 3D)也遵循相同原则:开新文件,不要在 `pseudo_spectral_*` 内塞新物理(会把基准 KH 搞坏)。`pseudo_spectral` 保留为"最小 2D 不可压 NS 演示"基线。

## 其他约定

- 用户偏好简体中文注释和文档;英文也可以,但不要强行翻译。
- commit 消息格式照搬仓库已有风格(`DOC:`/`ADD:`/`FIX:` 前缀 + 一句话 summary)。
- 新求解器用 `--solver <name>` 在 `src/main.cpp` 里加 dispatch;测试白名单(`hse`, `hse_perturbed`, `sod`, `lane_emden*`)按需扩展。
- 不要写多余的 README 或文档,用户要才写;真要写就放 `docs/`。
