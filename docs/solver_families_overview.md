# Solver Families Overview — 执行文档

**Created**: 2026-05-07
**Purpose**: stellar2d 代码库已经膨胀到 6 族 / 12+ 求解器,新加 solver
或做验证时迷路成本很高。这份文档是**唯一入口**,按"我要做 X → 去哪里 →
跑什么 test"组织。

CLAUDE.md 的"不可覆盖求解器资产"策略仍然有效;本文档只是**导航**,
不重复那里的保留理由。

---

## 0. 快速决策树

```
我要做什么?
├── 可压缩 2D 恒星对流(Andrassy / Stein-Nordlund 类)
│   ├── 显式基线 ────────── athena_vl2    (照抄 Athena++ vl2,最干净)
│   ├── ALE 基线 ────────── cart_ale2     (Caramana + PPM,主力生产)
│   └── 隐式 CFL 突破 ───── cart_impl     (Cartesian BE+JFNK,最新)
│
├── 低马赫恒星脉动 / 声波 / p-g mode
│   ├── JFNK 球极 ────────── lowmach       (内能方程,WB)
│   ├── FAS 非线性 MG ──── fas / fas2     (总能,球极)
│   └── 压力修正 ────────── simple / projection
│
├── 1D 径向恒星力学 testbed
│   └── ──────────────────── radial1d      (Lagrangian + Helm + pp-chain + rad)
│
├── 2D 不可压湍流 / KH / 强制湍流
│   └── ──────────────────── pseudo_spectral  (ω-ψ,IFRK3 + skew-symm)
│
├── 球面 barotropic 流(Rossby 波,行星大气 toy)
│   └── ──────────────────── sph2d_spectral   (球谐 + Fourier)
│
├── anelastic 对流(声波已滤)
│   └── ──────────────────── anelastic_sl     (Chebyshev × Fourier,⚠️未充分验证)
│
└── 球极传统 2D 欧拉(已被上面替代,保留对比)
    ├── explicit        (RK2 HLLC MUSCL)
    ├── wb2d            (well-balanced ⚠️ t≈2 死)
    ├── ale2d           (轴对称 ALE,⚠️ hoop bug)
    └── compressible    (legacy AmgX,⚠️ 不再建议用)
```

---

## 1. 六族架构图

```
stellar2d/src/gpu/
├── explicit/           Cartesian + polar 显式 Godunov
│   ├── strang          2D Cartesian Strang-split (test-only)
│   ├── athena_vl2      2D Cartesian vl2 unsplit (Andrassy baseline)
│   └── wb2d            2D polar well-balanced (reference, ⚠️ 不稳定)
│
├── implicit_lowmach/   球极 (r,θ) 隐式低马赫族
│   ├── lowmach         JFNK BE,内能方程,1D 径向引力
│   ├── fas / fas2      FAS nonlinear multigrid,总能方程
│   ├── simple          SIMPLE pressure correction
│   └── projection      Fractional-step pressure projection
│
├── ale/                2D Cartesian / axisymmetric ALE
│   ├── cart_ale2       ★ 主力可压缩对流(periodic BC + PPM + char)
│   ├── cart_ale        cart_ale2 的前身(reflective-wall only)
│   ├── cart_lag        纯 Lagrangian(reference,hourglass bug)
│   ├── cart_impl       Cartesian BE+JFNK(新,替代球极 lowmach)
│   └── ale2d_*         球极轴对称 ALE(⚠️ hoop bug)
│
├── radial1d/           1D 径向 Lagrangian
│   └── radial1d        MESA-style,Helm + pp-chain + rad diff + species
│
├── spectral/           伪谱 / 球谐
│   ├── pseudo_spectral ★ 2D 不可压 NS,ω-ψ
│   ├── sph2d_spectral  薄球壳 barotropic(Rossby / β-plane)
│   └── anelastic_sl    ⚠️ Chebyshev-Fourier anelastic(未充分验证)
│
└── common/             gpu_hllc, gpu_linalg, gpu_common, gmg_gpu
```

★ = 当前主力生产 solver。未标 ★ 的是 baseline / reference。

---

## 2. 物理正确性总表

基于 2026-05-07 审查。详细分析见各 solver 的 docstring 或
`docs/design/*.md`。

| 族 | 主 solver | 正确性 | 已知问题 | 建议信任度 |
|---|---|---|---|---|
| Explicit | `athena_vl2` | ✅ 照抄 Athena++ | 需 bit-compare 锁忠实度 | 高(待验证) |
| Explicit | `strang` | ✅ 扰动存储 WB 干净 | test-only,不 dispatch | 高 |
| Explicit | `wb2d` | ✅ flux-level WB 正确 | t≈2 扰动死(已知)| 低 |
| Implicit lowmach | `lowmach` | ✅ 2026-05-07 修复 S_E | 内能方程语义特殊 | 高 |
| Implicit lowmach | `fas/fas2` | ✅ 总能方程正统 | | 高 |
| ALE | `cart_ale2` | ✅ 非常干净 | KH 谱 k⁻¹⁰ (Caramana AV 限制) | 高 |
| ALE | `cart_lag` | ✅ 纯 Lagrangian | hourglass mode (HSE drift) | 中 |
| ALE | `cart_impl` | ✅ 总能,无 S_E 坑 | 新,测试尚少 | 中(待验证) |
| Radial1D | `radial1d` | ✅ 核心力学完全正确 | 架构限制(见 CLAUDE.md) | 高(限定域内) |
| Spectral | `pseudo_spectral` | ✅ 非常干净 | | 高 |
| Spectral | `sph2d_spectral` | ✅ 球谐正统 | | 高 |
| Spectral | `anelastic_sl` | ⚠️ **未充分验证** | 需 BV 脉动测试 | **未知** |

---

## 3. Standard Test 矩阵

每族对应的"必备 smoke test"。⭐ = 推荐先做。

### 3.1 Explicit 族

| Solver | Test | 验证 | 现有? | 文件/入口 |
|---|---|---|---|---|
| `athena_vl2` | ⭐ Athena++ bit-compare | 复刻忠实度 | 需加 | 跑同 IC,对比 VTK |
| `athena_vl2` | Sod 2D | HLLC + PLM | 需加 | `--test sod` |
| `athena_vl2` | Linear wave convergence | 2nd-order | 需加 | entropy/acoustic/shear |
| `athena_vl2` | Gresho 涡 | AV 稳定 | 需加 | drift < 1% at t=3 |
| `athena_vl2` | Implosion (Liska-Wendroff) | 对称性 | 需加 | x=y 对称 bitwise |
| `athena_vl2` | Andrassy 2022 | 生产 benchmark | ✅ | `init_andrassy2022` |
| `strang` | 熵波 convergence | 2nd-order | ✅ | `test_strang_convergence.cu` |
| `strang` | HSE long-time | WB 扰动存储 | ✅ | `test_strang_step.cu` |
| `wb2d` | Lane-Emden 静态 | WB | 需加 | 10⁴ dt δρ/ρ₀ ≤ round-off |

### 3.2 Implicit lowmach 族

| Solver | Test | 验证 | 现有? | 文件/入口 |
|---|---|---|---|---|
| `lowmach` | ⭐ IE drift zero-g | 2026-05-07 S_E 修复 regression | 需加 | 零重力 + 扰动,∫ρe dV 守恒 |
| `lowmach` | Lane-Emden perturbed | p/g-mode 频率 | ✅ | `--test lane_emden_perturbed` |
| `lowmach` | Newton 二次收敛 | JFNK 收敛率 | 部分 | `test_newton_tuning.cu` |
| `lowmach` | Acoustic CFL 突破 | 隐式核心价值 | 需加 | dt=10·dt_CFL_acoustic |
| `fas/fas2` | Lane-Emden 静态 | WB + FAS 收敛 | ✅ | `test_fas_verify.cu` |
| `fas/fas2` | Bubble 浮力 | 对流 | ✅ | `--test bubble` |

### 3.3 ALE 族

| Solver | Test | 验证 | 现有? | 文件/入口 |
|---|---|---|---|---|
| `cart_ale2` | ⭐ Uniform advection periodic | P30/P31 回归 | 需加 | mass drift = 0 |
| `cart_ale2` | Sod 2D | Riemann | ✅ | `--test sod` |
| `cart_ale2` | Sedov 2D cylindrical | 自相似激波 | ✅ | `init_sedov` |
| `cart_ale2` | Noh 球形 | 强激波 | ✅ | `init_noh` |
| `cart_ale2` | Gresho 涡 | AV 稳定 | ✅ | `init_gresho` |
| `cart_ale2` | Yee isentropic 涡 | 周期扩散 | ✅ | `init_yee_vortex` |
| `cart_ale2` | KH Lecoanet | 剪切不稳 | ✅ | `--test kh_lecoanet` |
| `cart_ale2` | Andrassy 2022 | 长时间对流 | ✅ | `init_andrassy2022` |
| `cart_ale2` | Compatible energy bit test | 每步 ΔKE+ΔIE=0 | 需加 | `1e-14` |
| `cart_impl` | ⭐ Andrassy 三码对比 | 与 cart_ale2 + athena_vl2 | 需加 | v_rms, mixing depth |
| `cart_impl` | Newton 二次收敛 | CGS2 + Viallet | 需加 | `‖F_{k+1}‖/‖F_k‖²` |

### 3.4 Radial1D 族

| Test | 验证 | 现有? |
|---|---|---|
| ⭐ Sedov-Taylor 点源 | 激波 + Lagrangian AV | 需加 |
| ⭐ Sod tube 球形 | Riemann + AV | 需加 |
| Polytropic star HSE | 长时稳定 | ✅ |
| P-mode 径向脉动(n=0,1,2) | 声学本征值 | 需加 |
| Evrard 冷球坍缩 | self-gravity | 需加 |
| Helm EOS round-trip | inverse 一致 | 需加(锁已知 bug) |
| pp-chain τ_burn 短时 | nuclear Picard lag | ✅ 隐式(已验证 ignition) |
| Radiation diffusion MMS | parabolic 子循环 | 需加 |
| MESA ZAMS 稳定 | 外部 profile | ✅ |

### 3.5 Spectral 族

| Solver | Test | 验证 | 现有? |
|---|---|---|---|
| `pseudo_spectral` | ⭐ Taylor-Green | 解析线性衰减 | 需加 |
| `pseudo_spectral` | KH single-mode growth | Miles 1961 σ | ✅(隐式在 KH test) |
| `pseudo_spectral` | 2/3 dealias enstrophy | ν=0 守恒 | 需加 |
| `pseudo_spectral` | Forced turb k⁻⁵/³ | Kolmogorov | ✅ `--test forced_turb` |
| `sph2d_spectral` | ⭐ Rossby-Haurwitz 4 波 | 球面解析 | 需加 |
| `sph2d_spectral` | l=1 纯旋转不演化 | DC mode | 需加 |
| `anelastic_sl` | ⭐ BV 脉动频率 | 声波已滤 | 需加 **(critical)** |
| `anelastic_sl` | MMS Poisson + advection | 谱收敛率 | 需加 |
| `anelastic_sl` | Rayleigh-Bénard onset | Ra_c ≈ 1708 | 需加 |

---

## 4. "加新 solver"流程(2026-05-05 refactor 后)

见 CLAUDE.md 第 "main.cpp 拆分后的 solver 添加流程" 节。要点:
- 求解器本体入 `src/gpu/<family>/`
- 写 `src/drivers/<name>.cpp`(可用 `sim/run_loop.h` 基础设施)
- CLI 选项入 `src/cli/options.{h,cpp}`
- `src/main.cpp` 加 dispatch 行
- CMakeLists.txt 加源文件条目

**不要往 `main.cpp` 塞逻辑**。

---

## 5. 数据流一致性陷阱

各 solver 的 state 语义**不统一**,切忌跨 solver 复制 kernel。

| Solver | `rhoE` 存的是 | 压力计算 |
|---|---|---|
| `lowmach` | **ρ·e_int**(内能) | `P = (γ-1)·rhoE` |
| `fas/fas2` | **ρE**(总能) | `e = (rhoE - KE)/ρ; P = (γ-1)·ρ·e` |
| `cart_impl` | **ρE**(总能) | 同 fas |
| `cart_ale*` | 不存 rhoE,存 `e_int` per cell | `P = (γ-1)·ρ·e_int` |
| `athena_vl2` | **ρE**(总能) | 同 fas |
| `radial1d` | 存 `e_int`,`rhoE` 只做诊断 | `P = (γ-1)·ρ·e_int` 或 Helm |
| `strang` | `E − p̄/(γ-1)` 扰动存储 | 从扰动重建后同 fas |

**lowmach 是唯一的内能存储族**。这是已知 pitfall(P32, 2026-05-07):
曾误把 FAS 的 `S_E = ρv·g` 源项抄到 lowmach 残差里,造成 O(Ma) 虚假
IE↔PE 转换,在 HSE (v=0) 下不可见,只在对流阶段露。删除即正确;
不要回加。

---

## 6. 推荐验证投入(按性价比)

如果有一周时间做系统性 test 补齐:

| 优先级 | Test | 预计工时 | 收益 |
|---|---|---|---|
| P0 | `anelastic_sl` BV 脉动 | 6h | 确认 anelastic 物理 |
| P0 | `lowmach` IE drift zero-g | 1h | 锁 2026-05-07 修复 |
| P0 | `athena_vl2` Athena++ bit-compare | 2h | 复刻忠实度永久锁死 |
| P1 | `pseudo_spectral` Taylor-Green | 2h | 谱 NS 黄金 smoke test |
| P1 | `cart_ale2` uniform advection drift | 1h | P30/P31 CI 锁 |
| P1 | `radial1d` Sedov + Sod | 4h | 1D Lagrangian 标准 |
| P2 | `sph2d_spectral` Rossby-Haurwitz | 2h | 球谐 NS 解析对比 |
| P2 | `cart_impl` Andrassy 三码对比 | 4h | 新 solver 充分验证 |
| P2 | `radial1d` p-mode 脉动谱 | 3h | 声学本征值 |

**P0 合计 9h**,覆盖本代码库三大风险点(anelastic 正确性、lowmach
修复锁、athena_vl2 忠实度)。

---

## 7. 相关文档

- [CLAUDE.md](../CLAUDE.md) — 资产保留策略 + 已踩坑列表(P01–P32)
- [docs/pitfalls.md](pitfalls.md) — 完整 pitfall 日志
- [docs/projects/spectral_liouville/equations.md](projects/spectral_liouville/equations.md) — 权威方程参考(§1–§17)
- [docs/design/](design/) — 各 solver 设计文档
- [docs/sessions/](sessions/) — session journal,追 commit 动机

---

## 8. 本文档维护

每次新加一个 solver 或 test:
- 更新 §1 决策树
- 更新 §2 正确性总表(即使是 "待验证")
- 更新 §3 test 矩阵相应列
- 如果动了 `rhoE` 语义,更新 §5 陷阱表

如果 solver 有物理 bug 被修复,把修复日期 + 一句话描述加到 §5(类似
P32 2026-05-07 S_E 的例子)。不要把修复细节塞进 README,那是 pitfall
责任;这里只记**跨 solver 复制代码时会踩的坑**。
