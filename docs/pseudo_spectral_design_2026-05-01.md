# pseudo_spectral 开发进度 — 2026-05-01

**Branch**: `pseudo-spectral`
**状态**: 第一版 working + 四项数值方法升级 (IFRK3, skew-symmetric, 圆形 dealias, ν_eff 诊断),已通过 1024² Re=10⁶ t=40 及 Re=2×10⁵ k=7 长时演化验证。
**下一步**: 2D Boussinesq 对流扩展(新增温度场 + 浮力项),或 2D MHD。

---

## 0. 一句话总结

2D 不可压缩 Navier-Stokes 偽谱求解器,雙週期域,渦度-流函數形式。cuFFT R2C/C2R + IFRK3 积分因子黏性 + Orszag skew-symmetric 对流 + 圆形 2/3 dealias。单消费 GPU (4070) 上可跑 1024² Re=10⁶ t=40,约 15 分钟。

文件组成 (~1300 行新增):

| 文件 | LOC | 内容 |
|---|---|---|
| `src/gpu/pseudo_spectral_solver.cuh` | 135 | Solver struct + API + Diagnostics |
| `src/gpu/pseudo_spectral_kernels.cu` | 285 | 10 个 CUDA kernel |
| `src/gpu/pseudo_spectral_solver.cu` | 590 | Orchestration + IFRK3 + IC + VTK + frame pool |
| `src/main.cpp` dispatch | ~90 | `--solver pseudo_spectral --test kh_shear` + 7 CLI flag |
| `scripts/render_pseudo_spectral.py` | 295 | Fork 自 render_cart_ale.py,双 panel (ω, |v|) |
| `scripts/spectrum_pseudo_spectral.py` | 155 | 能量谱 E(k) 环积分 + k^{-5/3}/k^{-3} 参考斜率 |

---

## 1. 方法学

### 1.1 方程

渦度-流函數形式的 2D 不可压 NS:

```
∂ω/∂t + u·∇ω = ν ∇²ω
∇²ψ = -ω,  u = ∂ψ/∂y,  v = -∂ψ/∂x
```

譜空間:

```
∂ω̂/∂t = -N̂(ω̂) - ν|k|²·ω̂
```

其中 N̂ 是对流项的谱表示。

### 1.2 对流项 (skew-symmetric form, 预设 ON)

两种等价形式:
- **Advective**: `N_A = u·∇ω`
- **Conservative**: `N_C = ∇·(uω)` (连续下 ∇·u=0 → N_A = N_C)

离散 FFT + 截断后两者**不相等**,差值表现为数值粘性。Orszag (1971) 提出:

```
N_S = ½(N_A + N_C)
```

在譜空間离散下,N_S 对应反对称算子 → **严格保持 enstrophy + KE 守恒**。代价:每 rhs evaluation 多一次 R2C FFT (计算 uω, vω 的譜)。

CLI `--ps-adv-only` 回退到 advective-only 作为对照组。

### 1.3 时间积分 (IFRK3 积分因子法, 预设 ON)

令 `ω̂ = exp(-ν|k|²·t)·ŵ`,粘性项消失:

```
∂ŵ/∂t = exp(+ν|k|²·t) · (-N̂)
```

然后用 SSP-RK3 (Shu-Osher) 推 ŵ。每级把 `exp(-ν|k|²·Δt_sub)` 乘回去。
黏性变成**逐模解析精确解**,不受 dt_visc 限制。

三级公式 (验证过 N=0 极限化简为精确扩散):

```
stage 1:  ω₁ = E·ωₙ + Δt·E·(-N̂(ωₙ))              (E = exp(-νk²Δt))
stage 2:  ω₂ = ¾·E^½·ωₙ + ¼·E^{-½}·ω₁ + ¼Δt·E^{-½}·(-N̂(ω₁))
stage 3:  ωₙ₊₁ = ⅓·E·ωₙ + ⅔·E^½·ω₂ + ⅔Δt·E^½·(-N̂(ω₂))
```

实测 ν=3e-3 512² 下比全显式 SSP-RK3 快 **48×**,物理结果到 4 位小数一致。

CLI `--ps-explicit` 强制回到全显式(受 dt_visc 约束)用于对比验证。

### 1.4 Dealiasing (圆形 2/3 rule)

硬截断 `|k|² ≤ (N/3·2π/L)²`。之前版本用的是方形 `max(|kx|,|ky|) ≤ N/3`,角落方向有偏向。圆形版对各向同性湍流更自然。

### 1.5 初始条件 (KH shear, Gibbs-clean)

双 tanh 剪切层 + 正弦 vy 扰动:

```
vx(y)   = vshear · (2·band(y) - 1),   band = ½[tanh((y-y_L)/δ) - tanh((y-y_H)/δ)]
vy(x,y) = amp · sin(k·2π·x/Lx) · (G(y,y_L) + G(y,y_H))
```

其中 `δ = max(4·dy, 0.01·Ly)`(保 Gibbs ringing 谱可解析),`σ = 0.05 Ly`。
解析求 ω₀ = ∂vy/∂x - ∂vx/∂y 设置 IC,FFT 后**立即套 dealias mask** 清除残留高 k 噪声。

---

## 2. 数值诊断 (Diagnostics struct)

```c++
struct Diagnostics {
    double total_KE;        // ∫½(u²+v²) dA
    double total_enstrophy; // ∫½ω² dA
    double max_v, max_omega;
    double eps_KE;          // 动能耗散率 = 2ν·Ω
    double eps_enstrophy;   // enstrophy 耗散率 = ν·∫|∇ω|² dA
};
```

2D 不可压 NS 下:
- `d(KE)/dt = -2ν·Ω` → `eps_KE` 是理论预期
- `d(Ω)/dt = -ν·∫|∇ω|² dA` → `eps_enstrophy` (譜空间 Σk²|ω̂|²)

**ν_eff 量化**: 比较实测 `-dKE/dt` 与 `eps_KE`:
- 若比值 ≈ 1 → 纯物理粘性,数值粘性小
- 若 `-dKE/dt > eps_KE` → 有显著数值粘性 → 降 dt 或加 dealias

实测 256² ν=1e-4 比值 0.97,1024² Re=2×10⁵ 比值 0.96 → 数值粘性 < 5%。

CSV 每次 diagnostic 都写入 eps_KE, eps_enstrophy 两欄。

---

## 3. 验证与跑过的算例

### 3.1 IFRK3 等价性验证

256² ν=1e-4 tend=2, IFRK3 vs 显式对比:KE 到 4 位小数一致,步数相同(都受对流 CFL)。
低 ν 下两者等价,高 ν 下 IFRK3 快几十倍。

### 3.2 Skew-symmetric vs Advective

512² ν=1e-6 tend=4:
- KE 守恒:两者差 1.6e-6 (千万分之一,都很好)
- Enstrophy:skew 保留多 1.4% → Orszag 预期方向

### 3.3 主要算例

| 算例 | 网格 | ν (Re_box) | t_end | 步数 | 状态 |
|---|---|---|---|---|---|
| KH k=4 long evolution | 1024² | 1e-5 (10⁵) | 100 | 197833 | ✓ (旧版显式) |
| KH k=4 high-Re | 1024² | 1e-6 (10⁶) | 40 | 121747 | ✓ IFRK3 |
| KH k=7 asymmetric | 1024² | 5e-6 (2×10⁵) | 40 | 99207 | ✓ IFRK3 + skew |

视频: `videos/ps_kh_1024_*.mp4`;能量谱: `videos/ps_kh_1024_*_spectrum.png`。

### 3.4 解析尺度核对 (k=7 Re=2×10⁵ run)

| 量 | 值 |
|---|---|
| Kolmogorov 尺度 η | 9.9×10⁻⁴ |
| dx/η | **0.99** (well-resolved DNS) |
| Taylor micro λ_T | 0.067 |
| Re_λ | ~1.3×10⁴ |

k=7 run 是**真正 well-resolved DNS**。k=4 Re=10⁶ run 的 dx/η ≈ 3,是 borderline / quasi-DNS。

---

## 4. CLI 参数清单

```
--solver pseudo_spectral        # 选择伪谱法求解器
--test kh_shear                 # (目前唯一支持的 test case)

--nr, --ntheta <N>              # 网格 nx, ny (复用既有 flag 名称)
--tend <T>                      # 终止时间
--cfl <C>                       # 对流 CFL 数 (预设 0.4)

# pseudo_spectral 专用 ------------------------------
--ps-nu <ν>           运动黏度 (预设 1e-4)
--ps-Lx <Lx>          域尺寸 x (预设 1.0)
--ps-Ly <Ly>          域尺寸 y (预设 1.0)
--ps-vshear <v>       基流速度 (预设 0.5)
--ps-k <k>            KH 扰动模数 (预设 4;质数如 7 破对称)
--ps-explicit         强制用全显式 SSP-RK3 (禁 IFRK3)
--ps-adv-only         强制 advective 对流形式 (禁 skew-symmetric)
--perturb <amp>       扰动振幅 (复用 flag,预设 1e-2;0.05~0.1 强非线性)

# 输出控制 (复用 cart_ale2 的系统) ----------------
--vtk-dt <Δt>         VTK 输出物理间隔 (预设 0;0=用 step 间隔)
--vtk-interval <N>    VTK 步间隔 (预设 follow --output-interval)
--diag-interval <N>   诊断 CSV 步间隔
--output-interval <N> 通用输出步间隔
--frame-buffer        启用 VRAM 帧池 (高频输出不阻塞 GPU)
--frame-headroom-mb <MB>  VRAM 预留容量 (预设 1024;1024² 建议 2048+)
```

---

## 5. 工程架构 (复用 cart_ale2 pattern)

- **VRAM frame pool**: 帧缓冲在 GPU 端,满了一次性 D2H + 写 binary VTK。单 flush O(100ms),否则每帧 IO 会阻塞计算
- **Binary VTK**: big-endian,和 cart_ale2 相容的 field 命名 (ω 写入 `e_int`, |v| 写入 `mach`),这样 `render_cart_ale.py` 可跑但不建议(label 对不上);专用 `render_pseudo_spectral.py` 双 panel 渲染
- **diagnostics.csv / frames.csv**: 完全对齐 cart_ale2 的 CSV schema,加两栏 eps_KE, eps_enstrophy
- **命令行风格**: 复用既有 `--vtk-dt`, `--frame-buffer`, `--output-interval` 等 flag,新增 `--ps-*` 专用

---

## 6. 已知限制与不适用场景

1. **只支援 Kelvin-Helmholtz**:`init_kh_shear` 是目前唯一 IC。加其他 IC 要开新 `init_*` 函数
2. **无重力/热力学/磁场**:纯涡度-流函数,不能做 RT, Bénard convection, MRI, MHD turbulence 等
3. **无 3D**:2D Fourier layout。扩 3D 要重写 cuFFT plan + d_w_* 改为三分量 + ψ 向量势
4. **无多 GPU / MPI**:单 GPU,1024² 舒服,2048² 勉强(cuFFT plan 吃 VRAM 大),4096² 撞 VRAM 墙
5. **无边界层**:只能双周期,无法做 wall-bounded (需要 Chebyshev)
6. **无自适应 dt 控制器**:目前 dt 由 CFL + dt_max 简单决定,没有 PI controller;对长时跑不是问题

---

## 7. 下一步扩展路径

按复杂度与物理价值排:

### 7.1 2D Boussinesq 对流 (推荐首选, ~1 天)

加温度场 θ + 浮力项:

```
∂ω/∂t + u·∇ω = ν∇²ω - g·∂θ/∂x        (浮力项)
∂θ/∂t + u·∇θ = κ∇²θ                    (温度平流-扩散)
```

新增:
- `d_theta`, `d_theta_hat` (+ 1 double, +1 complex buffer)
- 温度 rhs 用同样的 IFRK3 + skew 流程
- 新 IC: `init_rayleigh_benard(Ra, Pr, Atwood)` 或 `init_rt(Atwood)`
- 新 CLI: `--ps-g`, `--ps-kappa`, 新 `--test rayleigh_benard` / `--test rt_boussinesq`

对应天体物理:**恒星对流区**、红巨星包层、**Rayleigh-Taylor 超新星抛壳**(小 Atwood)。

**不要覆盖** 现有 `pseudo_spectral_*`,开新文件 `pseudo_boussinesq_*`。

### 7.2 2D MHD (中等, ~3–5 天)

加磁场 B = (Bx, By),可用磁势 A_z 简化(B = ∇ × (A_z·ẑ), 类比 ψ)。
方程族:

```
∂ω/∂t + u·∇ω = ν∇²ω + (B·∇)j_z            (Lorentz 力)
∂A_z/∂t + u·∇A_z = η∇²A_z                 (磁感应方程)
j_z = -∇²A_z                                (电流)
```

对应天体物理:**MRI (吸积盘)**、ISM 2D MHD 湍流、太阳表面磁对流。

### 7.3 3D Fourier (大工程, ~2 周)

重写整个 solver 为 3D:
- cuFFT 3D plan
- ω 变三分量 (ω_x, ω_y, ω_z),每个 3D 复数数组
- 或用速度 (u, v, w) 直接,projection 到 divergence-free
- VRAM 吃紧:256³ 约 134M cells × 8B × 多 buffer ≈ 10GB+ 就紧了

---

## 8. 文件结构速查

```
src/gpu/
├── pseudo_spectral_solver.cuh    (public API, Diagnostics, flag 注释)
├── pseudo_spectral_solver.cu     (init/destroy/IC/step/diagnostics/VTK/frame pool)
└── pseudo_spectral_kernels.cu    (10 个 CUDA kernel)

scripts/
├── render_pseudo_spectral.py     (fork cart_ale 渲染器, 双 panel)
└── spectrum_pseudo_spectral.py   (能量谱环积分 + Kraichnan 参考)

docs/
└── pseudo_spectral_design_2026-05-01.md  (本文档)
```

---

## 9. Commit 历史摘要

- `ADD: pseudo_spectral — 2D 不可压 NS 偽譜法求解器 (cuFFT + RK3)`
- `ADD: pseudo_spectral — IFRK3 积分因子法 + 谱分析 + 渲染品质`
- `OPT: pseudo_spectral 四项数值方法升级 (skew-form + 圆形 dealias + ν_eff 诊断 + IC 消 Gibbs)`

本文档记录到 commit `229afda` 为止。
