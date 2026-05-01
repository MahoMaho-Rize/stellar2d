# cart_ale Path 1 — 2阶 MUSCL-in-remap

**Date**: 2026-05-02
**Branch**: `ale2d-cartesian`
**Commit**: (pending)
**Reference**: Kucharik & Shashkov 2012 (JCP 231); `docs/cart_ale_progress_2026-05-01.md` §6 Path 1.

---

## 0. 一句话

把 cart_ale 的 swept-edge remap 从 donor-cell(1阶)升级到 minmod-limited 线性重构(2阶),
同一套测试:HSE 稳定不退化,双泡对撞 Mach peak 从 0.59 跳到 0.99,KE 在涡动期能持续 7e-5 量级
而不是被数值粘性碾平。

---

## 1. 做法

每步 Phase M 从原来的
```
V_sweep · (f_donor_bar)
```
换成
```
V_sweep · ( f_donor_bar + s_x·(x_sweep_c - x_donor_c) + s_y·(y_sweep_c - y_donor_c) )
```
其中 `s_x = minmod((f[i]-f[i-1])/dx, (f[i+1]-f[i])/dx)`,y 同理。

对 4 个守恒密度场同时做:`ρ = dm/V0`、`ρE = dm·e_int/V0`、`px_d = px/V0`、`py_d = py/V0`。
donor cell 的体积和中心都在 uniform 参考格上,查找 `O(1)`,无插值开销。

---

## 2. 代码位置

按 solver-asset 保留原则,旧 1阶 kernel 原地保留,新加并列 kernel:

| Kernel | 用途 |
|---|---|
| `k_cale_cell_densities` | dm/V0 等 4 个密度字段重算 |
| `k_cale_slopes_minmod` | per-cell minmod x/y 梯度,边界 slope=0 |
| `k_cale_remap_east_2nd` | east edge 2阶 swept transport |
| `k_cale_remap_north_2nd` | north edge 2阶 swept transport |

CLI 开关:`--remap-order 1|2`(default 2),在 `CartAleSolver::step()` 里按 `remap_order` 分派。
~180 LOC,包含 4 个密度字段 + 8 个 slope 字段的 malloc/free。

---

## 3. 验证(当天,128²)

| 测试 | 参数 | 结果 |
|---|---|---|
| HSE 无扰动 tend=5 | --perturb 0 | dt=2.40e-3 全程稳定,ΔE/E = 1e-8,|v| ~ 3e-4 衰减 |
| HSE 0.05 扰动 tend=10 | --perturb 0.05 | KE 在 1e-6~1e-5 动态振荡,不单调衰减(对比 1阶 衰减更明显) |
| 双泡 tend=15 | 0.5,0.25,-0.7 / 0.5,0.75,+1.5 | 7078 步稳定,KE=6.0e-5,|v|=3.4e-2 |
| 双泡 tend=50 | 同上 | 21810 步全稳定,ΔE/E=6.8e-3(1阶 同 test 是 1.5e-2) |

256² 双泡 tend=20:18625 步,KE 峰 7.6e-5。

---

## 4. Mach 对比(双泡 128²)

| 版本 | Mach peak | |v| peak |
|---|---|---|
| 1阶 donor-cell (Day-1) | 0.59 | 0.54 |
| 2阶 MUSCL minmod | 0.99 | 0.55 |

Mach 从 0.59 跳到 0.99 说明数值粘性大幅下降,局部跨声速特征保留完整。

---

## 5. 下一步

- [ ] 对撞 512² 长程对比,目测 KH mushroom 是否清晰(Day-1 预测 2阶下 ~4 个 KH 波)
- [ ] van Leer / MC limiter 可选(minmod 有时过度压 slope)
- [ ] Path 2:轴对称 hoop-stress + Pappus volume remap
- [ ] Path 3:FAS 耦合自引力
