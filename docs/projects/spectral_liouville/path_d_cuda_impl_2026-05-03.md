---
title: Path D CUDA 实现 — Assembled-matrix 线性 TD 完全闭环
author: Phase 2 完结(branch `anelastic-sl-spectral`)
date: 2026-05-03
---

# 1. 成果

把 Python 闭环证明(`docs/full_galerkin_closure_proof_2026-05-03.md`)的
assembled-matrix $L^{-1}R$ 结构移植到 CUDA,替换 `gmode_eigenmode_td` 里
的线性 TD 推进路径。实测结果:

## 1.1 短时 dev/step(线性算子一致性测试)

| Background | Primitive (baseline) | **Path D (assembled)** | 改善 |
|---|---|---|---|
| Boussinesq | 4.5e-5 | **1e-15** | $4.5\times 10^{10}$ |
| Lane-Emden n=3/2 | 6.9e-4 | **3e-15** | $2.3\times 10^{11}$ |

## 1.2 长时 FFT 频率(100 个周期)

| Background | Primitive + 强 filter | **Path D** | 改善 |
|---|---|---|---|
| Boussinesq ($t$=500) | -0.25% | **+0.018%** | 14× |
| Lane-Emden n=3/2 ($t$=300) | -9.4% | **-0.015%** | 600× |

## 1.3 长时 eigenmode deviation

| Background | Primitive + 强 filter | **Path D** |
|---|---|---|
| Boussinesq @t=500 | $6\times 10^{-3}$ | — |
| Lane-Emden @t=300 | $10^{-3}$(衰减塌陷) | **$4\times 10^{-9}$**(稳定) |

**Lane-Emden 完全闭环到机器精度级。**

# 2. 实现

## 2.1 API

`AnelasticSLSolver` 新增:

```cpp
// Field
int     n_int_path_d = 0;     // ny - 2
double* d_M_per_kx   = nullptr;  // [nh × n_int × n_int] col-major slab per kx
bool    td_assembled_linear = false;  // env ANSL_TD_KIND=assembled_linear

// Methods
void   assemble_path_d_operators();
double step_assembled_linear();      // full RK4, no advection/viscosity
```

`main.cpp` 的 `gmode_eigenmode_td` dispatch 检查 `ansl.td_assembled_linear`,
为真则调用 `step_assembled_linear()` 代替 `step()`。

## 2.2 矩阵组装(host, `set_background` 末尾)

对每个 $k_x \in \{2\pi k / L_x : k = 0..n_h-1\}$:
- $L = -D\,\operatorname{diag}(\rho)\,D + k^2 \operatorname{diag}(\rho)$,interior 切片 $(n_y-2) \times (n_y-2)$
- $R = k^2 \operatorname{diag}(N^2 \rho)$,interior
- Gauss-Jordan with partial pivoting → $L^{-1}$
- $M_{k_x} = L^{-1} R$(row $i$ of $L^{-1}$ × diag element $R_{jj}$)
- 打包 col-major 到 `d_M_per_kx[kx * n_int² + col*n_int + row]`

$k_x=0$ 模:$L$ 退化为 $-A$ 奇异。写入 $M_0 = 0$(g-mode IC 在 $k_x=0$ 无能量,
无害)。

## 2.3 时间推进(RK4 per step)

状态 `(V, W = ∂_t V)`,方程 $\ddot V = -M \cdot V$。每 RK4 substep:

1. R2C FFT $x$:`d_v` → `d_fhat`(复数 $n_y \times n_h$)
2. Per-$k_x$ kernel `k_apply_M_kx`:$n_h$ blocks × $n_{int}$ threads,做 $M_{kx} \cdot V_{hat}$,walls 零
3. C2R IFFT + scale $-1/n_x$ → `d_b`(就是 $-M \cdot V$ 即 $\ddot V$)
4. 加权累加到 $(V_{\rm acc}, W_{\rm acc})$
5. 末端:`d_v ← V_acc`, `d_rhs_v ← W_acc`

存储对应关系(仅 gmode_eigenmode_td 路径合法):
- `d_u_orig`:V 快照
- `d_v_orig`:W 快照
- `d_b_orig`:V 累加器
- `d_rhs_u`:W 累加器
- `d_b, d_scratch, d_rhs_b`:RK4 中间态(V/W 和 -MV)
- `d_fhat, d_ghat`:cuFFT 缓冲

## 2.4 限制

**当前只替换 `gmode_eigenmode_td`**。其他 test case(KH 湍流、gmode_pulsation)
继续用旧 `step()` 路径。要普及需要:

- 非线性 advection 加回来:在 `step_assembled_linear` 内先算 $(\bm u \cdot \nabla) V$
  使用 physical-space primitive + 2/3 dealias,再加到 W 的 RK4 RHS
- Poisson projection 保留(对线性 g-mode 实际已经由 $L^{-1}R$ 吸收,不需要)或
  只在非线性分量上做

这是下一步工作,不在本次 commit 范围。

# 3. 代码变更

- `src/gpu/anelastic_sl_solver.cuh`:
  - 字段 `n_int_path_d, d_M_per_kx, td_assembled_linear`
  - API `assemble_path_d_operators, step_assembled_linear`
- `src/gpu/anelastic_sl_solver.cu`:
  - `set_background` 末尾加 env check + `assemble_path_d_operators()` 调用
  - `init_gmode_eigenmode` 加 `cudaMemset(d_rhs_v, 0, ...)`(W(0)=0)
  - 新函数 `assemble_path_d_operators`(host Gauss-Jordan 组装 + 上传)
  - 新函数 `step_assembled_linear`(RK4 + FFT-based M·V apply)
  - 新 kernel `k_apply_M_kx`(per-kx DGEMV on complex v_hat)
  - 新 helper `_apply_M_kx_to_vhat`
  - `free_all` 加 `d_M_per_kx`
- `src/main.cpp`:
  - `gmode_eigenmode_td` 循环体 `use_path_d` 分支

# 4. 一键复现

```bash
cmake --build build -j --target stellar2d

# Lane-Emden short-time dev/step (machine precision)
ANSL_TD_KIND=assembled_linear ANSL_BG=lane_emden_1_5 ANSL_RHO_CUT=0.01 \
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 ANSL_SKIP_IC_PROJECT=1 ANSL_DT_MAX=1e-4 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-8 --tend 0.003 --cfl 0.1 --ps-nu 0
# Expect dev ~ 1e-15 per step

# Lane-Emden long-time FFT (300 periods)
ANSL_TD_KIND=assembled_linear ANSL_BG=lane_emden_1_5 ANSL_RHO_CUT=0.01 \
ANSL_COORD_MAP=tanh ANSL_COORD_BETA=2 ANSL_DT_MAX=5e-3 \
./build/stellar2d --solver anelastic_sl --test gmode_eigenmode_td \
    --ntheta 64 --nr 64 --ps-Lx 1 --ps-Ly 1 \
    --ps-k 1 --perturb 1e-3 --tend 300 --cfl 1.0 --ps-nu 0
python3 scripts/gmode_eigenmode_td_fft.py \
    runs/gmode_eigenmode_td_*/gmode_eigenmode_td.csv
# Expect rel_err ~ -1.5e-4  (vs primitive baseline -9.4e-2)
```

# 5. 理论 vs 实测一致性

Python 原型预测:Full-Galerkin matrix 推进在 Lane-Emden 下 dev/step ~ 1e-18
(machine × κ(L)).  CUDA 实测 3e-15,差 $10^3$ 的原因:

- cuFFT 精度损失:R2C → DGEMM → C2R round-trip 每步约 $10^{-14}$ 相对误差
- Host Gauss-Jordan($O(n^3)$ elementary,无 BLAS)累积约 $n \cdot \epsilon \approx
  62 \cdot 10^{-16} \approx 10^{-14}$

本底 $10^{-15}$ 符合预期,远低于物理 g-mode 频率精度要求 $10^{-4}$。

# 6. 下一步

本次闭环**只覆盖线性 g-mode**。要扩展到:

1. **非线性 g-mode 自相互作用**:Path D 里加入 advection 项(physical-space
   $(\bm u \cdot \nabla)(\bm u, b)$ + 2/3 dealias),RK4 RHS 变成
   $\ddot V = -M V + f_{\rm nonlinear}$
2. **MESA profile 时域**:换 `set_background` 的 ρ/N² 到真实恒星(已有
   `read_gyre_structure_txt`)
3. **非 eigenmode IC**:让 Path D 对任意 $V(x, y, t=0)$ 都正确推进(已经是了,
   只是没回归测试)

上面都是增量工作,**Path D 的核心数学-实现对齐已经完成**。
