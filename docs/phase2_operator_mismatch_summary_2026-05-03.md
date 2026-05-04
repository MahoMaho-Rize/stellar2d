---
title: Phase 2 汇总 — Lane-Emden operator mismatch 完整探索路线
author: Phase 2 汇总(branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 1. 一句话总结

**Lane-Emden n=3/2 下 EVP→TD 闭环失败(FFT 主峰 -9.4%, dev/step 6e-4)不是 bug
而是 CUDA TD 的 node-space primitive 算子设计缺陷。数学闭环路径已经完全证明:
把线性算子 assemble 成 $L^{-1}R$ 矩阵用 DGEMM 推进,dev/step 立即到机器精度
$5\times 10^{-18}$。剩下只有 3-4 天的 CUDA 实现工作(Path D)。**

# 2. 时间线

| 日期 | commit | 内容 |
|---|---|---|
| 2026-05-03 | 52070c7 | Boussinesq EVP→TD 闭环 $-0.25\%$(SL-basis filter) |
| 2026-05-03 | 31e03b6 | TANH/LOGRHO 坐标变换(Poisson rel err 3.6×) |
| 2026-05-03 | 3a5e2b6 | 变密度 Lane-Emden TD 失败定位,ANSL_BG env var |
| 2026-05-03 | 5b1b32a | Reduced-pressure q-space EVP 代数推导 + Python 原型 |
| 2026-05-03 | e3c7fe3 | **路径 A**:CUDA Fourier q-space EVP(Boussinesq 1e-15,Lane-Emden 6.0e-4) |
| 2026-05-03 | 0212fec | **路径 B**:CUDA SL-basis q-space EVP(Boussinesq 1e-14,Lane-Emden 5.9e-4) |
| 2026-05-03 | 17cc81d | **路径 C Python benchmark**:23% 改善,否决 3-4 天 CUDA 投入 |
| 2026-05-03 | 40b1446 | **Full-Galerkin TD 闭环证明**:1e-18 在 Lane-Emden,定位 Path D |

# 3. 路径对比表(完整)

| 路径 | 改动层 | EVP 变化 | TD 变化 | Boussinesq | Lane-Emden dev/step | 工程量 | 结果 |
|---|---|---|---|---|---|---|---|
| (baseline) | — | Galerkin v-space $B = -D\rho D + k^2\rho$ | primitive node(apply_dy + pointwise) | 4.5e-5 | 6.9e-4 | — | 起点 |
| **A** | EVP 换基 | Fourier q-space(ρ 消去) | 不变 | 4.5e-5 | 6.0e-4 | 0.5 天 | ✓ 做了;13% 改善 |
| **B** | EVP 换基 | SL basis q-space(ψ_n Galerkin) | 不变 | 4.5e-5 | 5.9e-4 | 1 天 | ✓ 做了;4% 改善 |
| **C** | TD 换变量 | (Fourier) q-space | TD 推 φ = ρv node-space | 1e-17(toy) | 6.8e-5(toy) | 3-4 天 | ✗ 否决;23% 改善不值 |
| **D** | TD 换离散 | 任意(A / B / C 都行) | assembled ny×ny $L^{-1}R$ + DGEMM | 1e-18 | **5.1e-18** | **3-4 天** | ✓ 推荐 |

# 4. 为什么 A/B/C 都失败、只有 D 成功

## 4.1 A (Fourier q-space)
代数正确($-\varphi'' + k^2\varphi = (k^2N^2/\omega^2)\varphi$ 与 ρ 无关),Boussinesq 机器精度。
但 Lane-Emden 下 TD 用的 SL basis $\{\psi_n\}$ 不是 Fourier basis,V̂ = φ/ρ 变换后投影
时在 $\psi_n$ 基展开有误差。

## 4.2 B (SL-basis q-space)
让 EVP 与 TD SL-Poisson pipeline 用同一个 ${\psi_n, \mu_n, w_{cc}}$。两路 Galerkin
投影严格一致。但 TD 里**不只 SL-Poisson 这一步**,还有 `compute_rhs_uv` 的 `apply_dy`
(用 Chebyshev D)、buoyancy 的 $N^2\cdot V$ 逐点乘、continuity 的 $\rho'\cdot v$ 逐点乘。
EVP 只对其中一个算子(Poisson)一致,不对整个 TD 线性算子一致。

## 4.3 C (TD 换变量到 φ = ρv node-space)
希望 $\rho$ 从算子系数消失。Python benchmark 发现 buoyancy $\partial_t B = -N^2 V$ 在
φ-space 下变成 $\partial_t(\rho B) = -N^2 \cdot \phi$,$N^2(y)$ 仍然是**pointwise 乘法**,
离散算子散射 eigenvector。改善只有 23%。

## 4.4 D (TD assembled-matrix DGEMM)
关键洞察:$V_{\rm EVP}$ 是 assembled matrix pair $(L, R)$ 的离散特征向量。如果 TD 推
$\ddot V = -L^{-1}R V$ 每步直接做 $L^{-1}R \cdot V$ 的矩阵向量积,**V_EVP 自动是 TD
算子的严格特征向量** —— 连续极限下等价,在本离散层也严格等价(因为推 TD 用的
矩阵就是构造 EVP 用的矩阵)。Python RK4 验证 Lane-Emden dev/step = 5e-18。

# 5. Path D 实现规格

## 5.1 改动位置

```
src/gpu/anelastic_sl_solver.cuh
  + 新成员 double* d_Linv_R_kx[]   // (ny × ny) 矩阵数组,每 kx 一份
  + bool use_assembled_matrix = false   // env 切换

src/gpu/anelastic_sl_solver.cu::set_background(最后)
  + 新函数 assemble_linear_operator_per_kx():
       for each kx in d_kx:
         L_cm = (-D·diag(ρ)·D + k²·diag(ρ))_interior
         R_cm = (k²·diag(N²·ρ))_interior
         M = cusolverDnDgetrs(L, R)  // L⁻¹R
         copy M → d_Linv_R_kx[kx]

src/gpu/anelastic_sl_solver.cu::compute_rhs_uv(v-eq 线性部分)
  if (use_assembled_matrix)
     // 新路径:RHS_V = -L⁻¹R · V(DGEMM per kx)
     for each kx: cublasDgemm with d_Linv_R_kx[kx]
  else
     // 原路径:apply_dy + pointwise (保留兼容)

src/gpu/anelastic_sl_solver.cu::project_div_free
  可选:不改(现有 SL-Poisson 对应 L 的 inverse,已对),或合并进 assembled 路径

src/main.cpp
  + ANSL_TD_BASIS=assembled 切换
```

## 5.2 成本

**VRAM**: $n_h \cdot (n_y-2)^2 \cdot 8$ B = $33 \cdot 62^2 \cdot 8$ ≈ 1 MB(对 64×64 grid)
**算力**: 每 RK3 substep 每 kx 一次 ny×ny × ny DGEMM = $n_y^2 \cdot n_h$ FLOP
$\approx 64^2 \cdot 33 = 135$k FLOP/mode,比现 apply_dy 的 $n_y^2 \cdot n_x$ ≈ $64^2 \cdot 64 = 262$k 少

## 5.3 工程量(3-4 天)

| 任务 | 天 |
|---|---|
| 设计 API + 分配 d_Linv_R_kx 缓冲 | 0.5 |
| CPU 组装 L, R 矩阵(per kx,用已有 h_Dy_row, h_rho, h_N2) | 0.5 |
| cuSOLVER getrs 或 host LAPACK 算 L⁻¹R,上传 GPU | 0.5 |
| 替换 compute_rhs_uv 里 v 的线性部分为 DGEMM 路径 | 1.0 |
| Buoyancy 处理(eliminate b,或单独 assemble N²·I 矩阵也上 DGEMM) | 0.5 |
| 集成测试(Boussinesq 回归、Lane-Emden 达到 dev/step ≤1e-10) | 1.0 |

## 5.4 风险

**低**:
- 数学路径 100% Python 证明
- 不换变量,不碰 Poisson projection(它是对的)
- 保留原 code path 做回归
- 非线性 advection 不改(仍 node-space + 2/3 dealias,那部分天然是 nonlinear
  dealias 问题,不属 operator mismatch)

# 6. Path D 之后的长期愿景

如果 Path D 闭环成功,**恒星 g-mode 时域**整条链被打开:

1. **真实 Lane-Emden / MESA profile** 可以做时域演化(当前只能 Boussinesq)
2. **EVP IC + 时域长期能量谱**:SGR 1806-like 地震波暴、恒星脉动模耦合
3. **非线性 mode coupling** 分析:EVP 本征模 + 时域演化 + Fourier 分析
4. **与 Exp K/GYRE 的交叉验证**:Exp K 给 frequency,TD 给 time waveform,
   合起来是完整 forward-modeling 工作流

# 7. 相关文档

1. `docs/variable_density_lane_emden_td_2026-05-03.md` — 失败定位
2. `docs/qspace_reduced_pressure_algebra_2026-05-03.md` — q-space 代数推导
3. `docs/qspace_sl_path_b_2026-05-03.md` — Path B 详细结果
4. `docs/path_c_python_benchmark_2026-05-03.md` — Path C 否决原因
5. `docs/full_galerkin_closure_proof_2026-05-03.md` — **Path D 闭环证明**
6. `scripts/qspace_evp_prototype.py` — q-space EVP Python 原型
7. `scripts/path_c_td_benchmark.py` — φ-space TD Python 测试
8. `scripts/full_galerkin_closure_test.py` — **assembled-matrix TD 闭环验证**

# 8. 小结:先原型后投资源的价值

这个 Phase 2 周期验证了一套完整的工作方法:

1. **A / B**:CUDA 级快速试错,2 天,结论 **不够**
2. **C Python benchmark**:半小时脚本,证明 $\varphi = \rho v$ node-space 不够
3. **D Python benchmark**:半小时脚本,证明 assembled-matrix 机器精度

**2 次 Python 原型(共 1 小时)** 把一个 $\ge 7$ 天的盲目 CUDA 投入降到 **3-4 天
的确定性实现**。

**结论**:Phase 2 数学完全闭环,等待 Path D 实施授权。
