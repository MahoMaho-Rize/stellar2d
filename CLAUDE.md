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
| `radial1d` | `src/gpu/radial1d_solver.*` | ✅ stable | **Lagrangian 隐式 hydro 1D**,2D ALE Newton-Krylov 机制 testbed。**不是** stellar evolution code — 见下 scope |
| `wb2d` | `src/gpu/wb2d_solver.*` | ⚠️ 扰动 t≈2 死 | 物理无错,是 well-balanced 欧拉流族的合理实现;未来需要时改进稳定器,不要拆 |
| `ale2d` | `src/gpu/ale2d_solver.*` | ⚠️ 轴对称 hoop bug | 轴对称 Caramana,文档里 `docs/design/ale_hoop_stress_fix.md` 已说明该怎么补丁。未来继续做球对称时直接上此文件,不要在新 ALE 里重做 |
| `cart_lag` | `src/gpu/cart_lag_solver.*` | ⚠️ HSE dt 退化 | 纯 Lagrangian,hourglass-mode 导致长时间 HSE 退化;文档 `docs/design/ale_rezone_design.md` 分析了原因。是"真 ALE"的 Lagrangian 相参考实现,不要覆盖 |
| `cart_ale` | `src/gpu/cart_ale_solver.*` | ✅ stable | Cartesian 2D ALE 基線(Caramana Lagrangian + Eulerian rezone + swept remap)。`--remap-order 1|2`:1=donor-cell,2=MUSCL(minmod/vanleer/mc)。reflective wall only,用來對比 cart_ale2 的周期 BC 版本 |
| `cart_ale2` | `src/gpu/cart_ale2_solver.*` + `cart_ale2_kernels.cu` | ✅ stable | cart_ale 的**定居版本**:完整 periodic BC(x/y 各自可切)、PPM-in-remap(CW 或 Colella-Sekora limiter)、primitive-space 重建 + 特徵變量投影(Athena Stone+08 Appendix A)、VRAM frame buffer 高頻 I/O。**恆星對流 / compressible 脈動的首選**。適用域:長時間 HSE 演化、低-高 Mach 可壓縮流。**不適用**:2D 全發展湍流 benchmark(有效 Reynolds 數受 Caramana subcell force 限制,KH 譜 k^{-10} 而非 k^{-3}) |
| `pseudo_spectral` | `src/gpu/pseudo_spectral_solver.*` + `pseudo_spectral_kernels.cu` | ✅ stable | 2D 不可压缩 Navier-Stokes 伪谱法(cuFFT + IFRK3 + skew-symmetric + 圆形 2/3 dealias)。双周期域,涡度-流函数形式。目前支持 `--test {kh_shear, forced_turb}`。详见 `docs/design/pseudo_spectral_design_2026-05-01.md` |

**"做真正的 ALE"的正确路径**:开**新文件 + 新求解器 struct**(例如 `cart_ale2_solver.{cu,cuh}` 配 `cart_ale2_kernels.cu`),可以**参考**`cart_ale_*` 或 `cart_lag_*` 里的 Lagrangian kernel,但不要修改它们。CMakeLists.txt 里新增编译条目即可。

**做伪谱法扩展**(Boussinesq / MHD / 3D)也遵循相同原则:开新文件,不要在 `pseudo_spectral_*` 内塞新物理(会把基准 KH 搞坏)。`pseudo_spectral` 保留为"最小 2D 不可压 NS 演示"基线。

## radial1d scope — 能做什么、不能做什么

radial1d 是 **1D Lagrangian 隐式 compressible hydro**:
U = (v, r, e),JFNK Newton with Dual<N> AD matvec,Viallet row scaling,
well-balanced HSE via `F = (U−Uⁿ)/dt − (R(U) − R_hse)`,rad-in-F 耦合
Newton,MLT conductivity Picard-lagged,Helm EOS,pp-chain nuclear 源。

**能做**:
- MESA IC 载入(seed T 或 P)→ 验证能稳定维持 HSE 数万 dt
- ZAMS IC 的 pp 点火瞬间响应(T_c=1.34e7, L_nuc=0.6 L☉ 已验证工作)
- 脉动 / 声波 / 短时标动力学响应
- 作为 2D cart_ale2 / cart_impl 的 Newton-Krylov 机制 testbed

**不能做**(2026-05-04 确认,多次尝试失败):
- **pre-MS KH contraction**(T_c 从 7.4e6 爬到 1.5e7 要 30 Myr)
- **任何长时标 stellar evolution**(KH τ、nuclear τ 级别)
- **正确的 L_surf = f(T_eff) 光球输出**

**为什么 pre-MS KH 跨不过**(根本架构限制,不是 bug):
1. HSE well-balancing 下 F_v ≡ 0 → Newton 不 move v → 核心无 PdV → T_c 不升
2. MESA-style stellar code(Henyey method)不解 (v, ∂v/∂t),只解结构 BVP
   ∂r/∂m, ∂P/∂m, ∂L/∂m, ∂T/∂m 四方程 + 时间只出现在 `T·Δs/Δt` 源项
3. Equal-mass Lagrangian zoning 无 adaptive remesh → 光球塞进单 zone
4. Grey Stefan / Eddington BC 无法代替 Kurucz / PHOENIX T(τ; T_eff, log g) 查表
5. Helm EOS Newton inverse 在 ρ~10⁻⁷, T~4500K 表外 edge 有 inversion bug

**做严肃 pre-MS KH 的正确路径**:用 MESA / KEPLER / GENEC 跑演化,
把它们的 profile 当 IC / snapshot 喂给 radial1d 做 **动力学验证**(脉动、
acoustic 响应、nuclear flash 瞬态),不要让 radial1d **自己演化**过 τ_KH。

**已尝试的 dead-ends**(2026-05-03 / 05-04 session,commit 历史见
`docs/sessions/session_journal_2026-05-03_ignition.md`):
- Eddington 1-zone BC (14f2e38) — 稳定 dt 但 L_surf 仍错
- nuc-aware Viallet Le (8912326) — Newton 看见 ε_pp 但 Δe 负
- Hybrid outer zoning (76599d7) — IC T=4570K 对,但 atm Δr 小使 dt 崩
- Helm T_guess 改 ideal (76599d7) — 在 low-ρ 减少反演错 branch
- Newton rel_tol 1e-3→1e-9 (20b6add) — GMRES drop ≠ Newton converged
- Operator-split atmosphere (0cd7f88) — 原则对,但 hybrid 界面 Jacobian
  条件数爆,新 session 工作量 2-3 天才能做稳

以上 commit **都正确**(各自解一个独立问题),但**组合起来不足以**让
hydro code 跑 KH。不要以为"再修一个小 bug 就过"了,**物理-数值阻抗失配
是系统性的**,下次别再花时间往这条路修。

## cart_ale2 使用注意

- 周期 BC 的 remap/force sync 有微妙陷阱(見 pitfalls P30, P31);**凡是新增 remap kernel,都必須加 `bc_mode` 並處理 wrap edge** — 寫了 `(nx-1)*ny` 而非 `nx*ny` 的 `n_edges`,周期流就會累積漂移。
- Periodic `sync_node` 有 `mode` 參數:**force 用 sum(mode=1),velocity/dX 用 copy(mode=0)**。cell-parallel atomicAdd 寫到邊界 node 只給一半貢獻,sync 時得求和補上;而 state 變量在兩個副本上應該完全相同,平均即可。
- 診斷 KE/PE 時要跳過 periodic duplicate nodes(`in==nnode_x-1` 和 `jn==nnode_y-1`),否則會雙重計算。

## 其他约定

- 用户偏好简体中文注释和文档;英文也可以,但不要强行翻译。
- commit 消息格式照搬仓库已有风格(`DOC:`/`ADD:`/`FIX:` 前缀 + 一句话 summary)。
- 新求解器用 `--solver <name>` 在 `src/main.cpp` 里加 dispatch;测试白名单(`hse`, `hse_perturbed`, `sod`, `lane_emden*`)按需扩展。
- 不要写多余的 README 或文档,用户要才写;真要写就放 `docs/`。
