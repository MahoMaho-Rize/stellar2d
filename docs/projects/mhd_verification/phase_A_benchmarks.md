# Phase A — athena_mhd 基准测试实测记录

**Date**: 2026-05-08
**Branch**: `athena-mhd-solver`
**Commit**: `235eafd` (Phase A gap closure)
**GPU**: RTX 40-series(见 `nvidia-smi`),CUDA 12.x
**Solver**: `src/gpu/explicit/athena_mhd_*`(VL2 + HLLD + CT)
**Build**: `cmake --build build --target <test>`

本文档是**代码快照 × 硬件 × 日期**三元组下的实测数据。
派生书 `docs/mhd_derivations/` 负责推导(硬件无关、纯数学);本文档
负责"在上面这些具体条件下跑得怎么样"。两者通过 pass criterion 的
§ 编号相互链接:测试文件 / 结果表里每一个阈值都能追溯到派生书
§A/§B/§C/§F 里的一条 sympy-verified identity。

---

## 总览

| 测试套件 | 入口 | 断言数 | 状态 |
|---|---|---|---|
| A1 oblique linwave 2D | `tests/test_athena_mhd_linwave_oblique.cu` | 16/16 | ✅ |
| A3 field loop long-time | `tests/test_athena_mhd_field_loop_long.cu` | 5/5 | ✅ |
| A4 CPAW longtime + η_eff | `tests/test_athena_mhd_cpaw_longtime.cu` | 13/13 | ✅ |
| A2 OT 2D spectrum pipeline | `scripts/mhd_verification/*` | 3 res · slope+k_diss | ✅ |
| A5 standard benchmark gate(本次加入) | `tests/test_athena_mhd_benchmarks.cu` | 30/30 | ✅ |
| (附)Brio-Wu dedicated | `tests/test_athena_mhd_brio_wu.cu` | — | ✅ |
| (附)linwave convergence 1D | `tests/test_athena_mhd_linwave_convergence.cu` | — | ✅ |
| (附)field_loop short | `tests/test_athena_mhd_field_loop.cu` | — | ✅ |

合计:**4 × smooth/linear 测试(A1-A4)+ 6 × shock/nonlinear 测试(A5)= 10 类 benchmark,64 断言全通过**。

---

## A1 — Oblique linwave 2D 收敛(4 模态 × 3 分辨率)

**派生**: §F1 rotated eigenvector + §F1b joint rotation covariance (strong form)
**方法**: Stone+08 §6.2 setup, $L_x = 2, L_y = 1$,$\mathbf{k} = 2\pi(1,2)/L$,$\theta \approx 63.4°$。
**指标**:`(1 − amp(t_end)/amp(0))` 对分辨率的收敛斜率,fast/alfven/slow 用 RMS(δB_y),entropy 用 RMS(δρ)(§F1 eigenvector: entropy 模态 $r_{B_y} = 0$, $r_\rho = 1$)。

| mode | decay(32) | decay(64) | decay(128) | slope 32→64 | slope 64→128 | max\|∇·B\| |
|---|---|---|---|---|---|---|
| fast    | 4.38e-1 | 9.46e-2 | 1.36e-2 | 2.213 | 2.794 | 2.5e-13 |
| Alfvén  | 3.23e-1 | 6.11e-2 | 7.57e-3 | 2.400 | 3.014 | 3.1e-13 |
| slow    | 1.89e-1 | 2.36e-2 | -7.47e-5† | 3.002 | (noise floor) | 2.3e-13 |
| entropy | 1.43e-3 | 1.82e-4 | 2.29e-5 | 2.972 | 2.996 | 7.8e-14 |

† slow 在 N=128 进入 round-off 噪声地板,负号是 ULP 级随机;最低-pair slope 3.002 足够通过 ≥ 1.8 阈值。

**通过阈值**:每模态最低-pair slope ≥ 1.8(§F5 解释为什么 2nd-order scheme 在 amp-retention 测量下给 p = 3)。

---

## A2 — Orszag-Tang 2D 谱 + ν_eff(3 分辨率 × t = 0.5)

**派生**: §F2 MHD turbulence spectrum
**Pipeline**:
1. `scripts/mhd_verification/run_ot_spectrum_scan.sh` — 驱动 stellar2d 跑 N ∈ {128, 256, 512}, t = 0.5
2. `scripts/mhd_verification/analyze_ot_spectrum.py` — VTK → 2D FFT → shell-averaged E(k) → slope fit + k_diss 检测

| N | 惯性区 slope | k_diss | max\|∇·B\| | wall-time |
|---|---|---|---|---|
| 128 | -2.02 | 53 | ≤ 1e-13 | ~1 s |
| 256 | -2.10 | 104 | ≤ 3e-13 | ~8 s |
| 512 | -2.17 | 212 | ≤ 1e-12 | ~60 s |

**分辨率标度**:
- $k_\mathrm{diss}(256)/k_\mathrm{diss}(128) = 1.96$(p ≈ 0.97)
- $k_\mathrm{diss}(512)/k_\mathrm{diss}(256) = 2.04$(p ≈ 1.03)

**解读**:
- slope -2.1 不是 K41/IK 的 -5/3。2D 可压缩 MHD 有激波贡献 k⁻²(Biskamp 2003 §7, Stone+08 §6.4)。派生书 §F2 的 "-5/3" 是不可压缩极限,不适合本测试。
- k_diss ∝ N(p ≈ 1)说明 VL2+PLM+HLLD 的耗散尺度由 grid cutoff 主导,不是 2nd-order viscous cascade。这是 solver-family characteristic。

**输出**:`scripts/mhd_verification/spectra/ot_combined_spectrum.{pdf,png}`

---

## A3 — Long-time field loop(10 次穿越 CT 保持)

**派生**: §F3 CT round-off bound + B_cc aliasing
**Setup**: GS05 field-loop IC, N = 128², 10 diagonal crossings, limiter = minmod

| 量 | 测量值 | F3 界 / 预期 |
|---|---|---|
| max\|∇·B\| over full run | **1.85e-15** | worst-case 1.14e-9, random-walk 1.14e-11 |
| max ME_cc(t) / ME_cc(0) | 1.56× | F3-aliasing 允许 ~1.6× |
| final ME_cc / ME_cc(0) | > 0.5 | loop 结构保持 |

**关键结论**:CT 在真实硬件上是 **round-off 精确**(实测比 random-walk 界还紧 4 个数量级),符合 Gardiner-Stone 2005 §3.4.1。

**Limiter 注记**:van Leer(默认)在此 IC 上 t > 5 起数值不稳(C⁰ kink 触发的可压缩失稳),不是 CT bug——同时 CT 仍然把 ∇·B 保持到 10⁻¹⁴。minmod 长时稳。

---

## A4 — CPAW longtime + η_eff 提取(3 分辨率 × 5 周期)

**派生**: §F4 linear Alfvén dispersion + §F5 modified-equation O(h⁴)
**Setup**:§A11 ALFVEN linear-wave IC, $A = 10^{-6}$(深线性), slab $N \times 4$, 5 wave periods

| $N$ | amp(5T)/amp(0) | $\gamma_\mathrm{num}$ | $\eta_\mathrm{eff} = 2\gamma/k^2$ |
|-----|----------------|-----------------------|----------------------------------|
| 32  | 0.9155 | 1.8e-2 | 8.93e-4 |
| 64  | 0.9896 | 2.1e-3 | 1.06e-4 |
| 128 | 0.9986 | 2.9e-4 | **1.45e-5** |

**Scheme-order inversion**(§F5-p3 正确公式):
- $p(32 \to 64) = 3.08$
- $p(64 \to 128) = 2.87$

§F5 证明 p = 3 是 amp-retention 测量下 2nd-order scheme 的**正确**签名(不是 super-convergence)。F4 原本预测的 p = 2 是 derivation bug——它忘了 N_step ∝ 1/h 这一项。实测 p = 2.87–3.08 与 F5-p3 自洽。

**Numerical magnetic Reynolds**(domain-scale):
- $\mathrm{Re}_m^\mathrm{num}(128) = v_A L / \eta_\mathrm{eff} \approx 6.9 \times 10^4$
- 外插(h² 标度):$\mathrm{Re}_m^\mathrm{num}(512) \approx 1 \times 10^6$

---

## A5 — Standard benchmark gate(30/30)

**入口**: `tests/test_athena_mhd_benchmarks.cu`
**覆盖**:1D 激波管 + 非线性 Alfvén + 涡旋 + 强激波 + 旋转磁流体

所有测试在单 GPU 上连续跑完,总时间 < 2 分钟。

### #1 Brio-Wu(γ = 2 shock tube)

256 × 4, t = 0.1, 366 steps

| 断言 | 测量 | 阈值 |
|---|---|---|
| mass conservation | 7.9e-16 | < 1e-12 |
| energy conservation | 4.7e-15 | < 1e-10 |
| max\|∇·B\| | **0.000** | < 1e-10 |
| finite | ✓ | — |

### #2a Ryu-Jones RJ2a(7-wave tube)

512 × 4, t = 0.2, 987 steps, outflow BC

| 断言 | 测量 | 阈值 |
|---|---|---|
| max\|∇·B\| | **0.000** | < 1e-10 |
| finite | ✓ | — |
| max_v | 1.30 | < 5.0 |
| 剩余 mass > 0 | drain ratio 1.25 | — |

### #2b Ryu-Jones RJ4d(switch-on slow rarefaction)

512 × 4, t = 0.16, 843 steps

| 断言 | 测量 | 阈值 |
|---|---|---|
| mass conservation | 1.9e-14 | < 1e-12 |
| energy conservation | 2.6e-14 | < 1e-10 |
| max\|∇·B\| | **0.000** | < 1e-10 |
| finite | ✓ | — |

### #3a CPAW 1D L¹ 收敛

N ∈ {32, 64, 128}, t = 1.0(一个周期)

| N | L¹(δB_y) |
|---|---|
| 32  | 1.94e-3 |
| 64  | 4.86e-4 |
| 128 | 1.17e-4 |

- slope 32→64 = **1.996**
- slope 64→128 = **2.055**

**标准 2nd-order L¹ signature**,与 A4 的 p = 3 amp-retention 签名并存——两者都对应同一个 2nd-order scheme,§F5 解释为什么两种 diagnostic 给出不同的 effective p。

### #3b CPAW 2D amplitude preservation

32 × 16, t = 1.0, 58 steps

| 断言 | 测量 | 阈值 |
|---|---|---|
| mass | 4.66e-4 | < 1e-3(low-res limit) |
| energy | 4.79e-3 | < 1e-2(low-res limit) |
| max\|∇·B\| | 1.9e-14 | < 1e-10 |
| ME ratio final/IC | **1.0011** | ∈ (0.5, 2) |

### #4 Orszag-Tang @ t = 0.5

128², 293 steps

| 断言 | 测量 | 阈值 |
|---|---|---|
| mass | 2.4e-13 | < 1e-12 |
| energy | 4.8e-15 | < 1e-10 |
| max\|∇·B\| | 1.2e-13 | < 1e-10 |
| finite | ✓ | — |

### #5 MHD Blast(强激波 + 磁场拓扑)

**Setup**: 200 × 300(长宽比 2:3),t = 0.2,中心 P_in = 10 压进 P_out = 0.1 背景,均匀 B₀ @ 45°(β_out = 0.2 强磁化)

506 steps to t = 0.2

| 断言 | 测量 | 阈值 |
|---|---|---|
| mass | 3.8e-13 | < 1e-12 |
| energy | 2.0e-13 | < 1e-10 |
| max\|∇·B\| | 1.6e-12 | < 1e-10 |
| finite | ✓ | — |
| **diagonal asymmetry (y = -x)** | **3.3e-15** | < 5e-2 |

**Mirror symmetry 亮点**:Stone+08 §8.4 指出维度分裂 Godunov 在此几何上损失 percent 级对称性。VL2 + HLLD + CT 在我们的 solver 上做到 **3.3 × 10⁻¹⁵(机器精度)**——说明 dimension-split + corner EMF 做对了。

### #6 MHD Rotor(Tóth 2000 Test 1)

**Setup**:128²,t = 0.15,r < 0.1 区域 ρ = 10 + ω = 200 刚体旋转(peak v = 20),外部 ρ = 1 静止,P = 1 uniform,$B_x = 5/\sqrt{4\pi}$

624 steps

| 断言 | 测量 | 阈值 |
|---|---|---|
| mass > 20% IC | ✓(outflow BC + Mach-10) | — |
| max\|∇·B\| | 2.0e-12 | < 1e-10 |
| finite | ✓ | — |
| max_v ∈ (2, 30) | 6.12(从 20 衰减) | — |

**Torsional Alfvén wave transport 正确**:peak 速度从 20 衰减到 6.12(磁场 torque 将 angular momentum 传出),非 catastrophic blow-up。

---

## 附表 — Brio-Wu 专测(已覆盖于 A5-#1,保留文件作 regression)

`tests/test_athena_mhd_brio_wu.cu` 与 A5-#1 逻辑重复,单独存在作为 build/CI 检查入口。

## 附表 — linwave convergence 1D

`tests/test_athena_mhd_linwave_convergence.cu`:对 fast/Alfvén/slow/entropy 四模态在 1D(slab Ny = 4)上做 L¹ 收敛测试,均给出 slope ≈ 2.0;与 A1 oblique 的 2D 版对比,1D 是退化特例。

## 附表 — field_loop short

`tests/test_athena_mhd_field_loop.cu`:1–2 次穿越的短时 smoke;A3 是长时版本。

---

## 运行命令汇总

一次跑全部 A-phase 测试:

```bash
cmake --build build --target \
    test_athena_mhd_linwave_oblique \
    test_athena_mhd_field_loop_long \
    test_athena_mhd_cpaw_longtime \
    test_athena_mhd_benchmarks

build/test_athena_mhd_linwave_oblique     # A1  16/16
build/test_athena_mhd_field_loop_long     # A3   5/5
build/test_athena_mhd_cpaw_longtime       # A4  13/13
build/test_athena_mhd_benchmarks          # A5  30/30

# A2 OT 谱(~2 分钟 GPU)
bash scripts/mhd_verification/run_ot_spectrum_scan.sh
python scripts/mhd_verification/analyze_ot_spectrum.py
```

或者通过 ctest:

```bash
cd build && ctest -R athena_mhd --output-on-failure
```

---

## Phase A → Phase B/C gate

本文档实测数据确认:

- **Smooth/linear 区间**(A1/A3/A4): 2D 耦合、CT 长时保持、η_eff ~ 10⁻⁵ 定量
- **Turbulent 区间**(A2): VL2+HLLD 的 2D 可压缩 MHD 谱行为(-2.1 slope, k_diss ∝ N)
- **Shock/nonlinear 区间**(A5): 强激波 positivity、维度分裂 mirror symmetry 机器精度、torsional Alfvén transport 正确、所有场景 ∇·B ≤ 10⁻¹²

Suzuki 风物理栈需要的 robustness 基础全部到位。下一步(Phase B/C):
- MHD KH 剪切(Alfvén stabilization 定量)
- 源项接入(C6 Spitzer, C7 冷却, C8 色球, E1 驱动)—— 派生已完成,kernel 未实现
- 重力 + 分层 Alfvén(well-balanced MHSE, §B4)

---

## Artifacts

| 派生 | 路径 |
|---|---|
| §F1 (rotated eigenvector) | `docs/mhd_derivations/sections/f1_oblique_linwave.md` |
| §F1b (joint rotation covariance, strong form) | `docs/mhd_derivations/scripts/f1b_joint_rotation_covariance.py` |
| §F2 (MHD spectrum) | `docs/mhd_derivations/sections/f2_mhd_turbulence_spectrum.md` |
| §F3 (CT round-off + B_cc aliasing) | `docs/mhd_derivations/sections/f3_ct_roundoff_and_bcc_aliasing.md` |
| §F4 (CPAW decay) | `docs/mhd_derivations/sections/f4_cpaw_decay_eta_eff.md` |
| §F5 (VL2+PLM O(h⁴) → p = 3) | `docs/mhd_derivations/sections/f5_vl2_plm_amplitude_decay.md` |

| CSV 输出 | 内容 |
|---|---|
| `build/field_loop_long.csv` | A3 40 个 (t, divB, ME_cc) 快照 |
| `build/cpaw_longtime.csv` | A4 3 行 η_eff 表 |
| `scripts/mhd_verification/spectra/ot_combined_spectrum.pdf` | A2 三分辨率 E(k) 叠加图 |
