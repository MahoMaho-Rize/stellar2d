# cart_ale2 标准验证测试

5 个 canonical benchmark,用来验证 ALE solver 在 **shock capturing /
各向同性 / 涡 / 平流守恒** 四个轴上的基本正确性,在上头跑 Andrassy /
恒星对流之前先把这 5 项跑过。

每个测试:
1. 用 cart_ale2 跑一次(见 "Command")
2. 调对应 `scripts/tests_ale2/*_compare.py`(见 "Check")
3. 检查输出 `PASS` / `FAIL`,数值在 "Expected" 范围

**前提**:已经 build `stellar2d` binary(GPU 路径);`runs/` 目录
存在。所有命令默认 256² 分辨率,可以按需改 `--nr / --ntheta`。

| | 核心考点 | 推荐 res | 时长 |
|---|---|---|---|
| Sod | 1D shock / contact / rarefaction capture | 256² | < 1 min |
| Sedov | 2D cylindrical 自相似 blast | 256² | ~2 min |
| Noh | 强 AV / stagnation pressure | 256² | ~3 min |
| Gresho | stationary rotation,AV 是否误触发 | 128² | ~1 min |
| Yee vortex | 平流 + remap diffusion | 256² | ~5 min |

---

## 1. Sod shock tube

**物理**:1D Riemann problem(ρL=1, pL=1 | ρR=0.125, pR=0.1),
γ=1.4,x₀=0.5,t=0.2。2D 里做为 y-均匀初值。

**Command**:
```bash
./stellar2d --solver cart_ale2 --test sod \
    --nr 256 --ntheta 256 --tend 0.2 \
    --cfl 0.3 --remap-order 2 --bc-x reflect --bc-y reflect \
    --vtk-dt 0.2 --run-dir runs/sod_256
```

**Check** (C++ compute_error + pytest tst/, 2026-05-07 onwards):
```bash
# Pass --compute-error to emit sod-errors.dat; pytest reads it.
cd tst && STELLAR2D_BIN=../build/stellar2d pytest test_ale2/test_ale2_sod_gpu.py -v
```
`scripts/tests_ale2/sod_compare.py` (legacy 120-line python Toro
solver) was removed — the analytic is now in `src/gpu/common/sod_exact.h`,
shared with `athena_vl2`.

**Expected**:`L1(ρ)/|Δρ|` < 6%(256² MUSCL-vanleer 2026-05-07
实测 5.0%,128² 实测 7.3%)。ALE swept-remap 在 Sod 上的典型
L1 比 Godunov (Athena vl2) 高约 2×,因为 remap 上的 donor-cell
重建比 HLLC 多一层数值耗散。L∞ ~ 0.23–0.29 的 contact smearing
正常。

---

## 2. Sedov 2D cylindrical blast

**物理**:点爆 E₀=1 沉积在中心 ~4 cell,ρ₀=1,p_amb=1e-5,γ=1.4。
自相似 shock 半径 `r_sh(t) = ξ₀·(E/ρ)^{1/4}·t^{1/2}`,2D 柱对称
ξ₀≈1.0 (Kamm 2007 LA-UR-07-2849 Table 6)。

**Command**:
```bash
./stellar2d --solver cart_ale2 --test sedov2d \
    --nr 256 --ntheta 256 --tend 0.05 \
    --cfl 0.3 --remap-order 2 --bc-x reflect --bc-y reflect \
    --vtk-dt 0.05 --run-dir runs/sedov_256
```

**Check**:
```bash
python3 scripts/tests_ale2/sedov_compare.py \
    --run-dir runs/sedov_256 --t 0.05
```

**Expected**:`|Δr_sh|/r_sh_theor` < 10%(2026-05-07 实测 7.7% 在 256²)。
t=0.05 时 r_sh_theor ≈ 0.224。ρ_peak 理论 = 6;实测 2.25 远低于
理论,**与 cart_ale2 的 Caramana-subcell AV 在点爆这种极强压
缩瞬间耗散大有关**,是方法固有。如果对 shock-peak 有要求,
推荐用 `athena_vl2` 代替(HLLC 给出 ρ_peak ≈ 4.5–5)。
cart_ale2 用作 **stellar convection / HSE long-time**,shock
峰值精度不是它的主打指标。

---

## 3. Noh 2D implosion

**物理**:ρ=1, p=10⁻⁶, v = −r̂ 全域,γ=5/3。
exact 后果:`ρ_post = 16`, `p_post = 16/3`, shock 速度 1/3。
在 t=2 时 r_sh = 2/3。**ALE / AV 测试最残酷的一题**,
经典 Noh 1987 解 / LANL RAGE 测试。

**Command**:
```bash
./stellar2d --solver cart_ale2 --test noh \
    --nr 256 --ntheta 256 --tend 2.0 \
    --cfl 0.2 --remap-order 2 --bc-x reflect --bc-y reflect \
    --vtk-dt 2.0 --run-dir runs/noh_256
```

**Check**:
```bash
python3 scripts/tests_ale2/noh_compare.py \
    --run-dir runs/noh_256 --t 2.0
```

**Expected**:⚠️ **此测试在 cart_ale2 当前 BC 下无法通过** —
Noh 的规范做法需要 outer boundary 持续注入 v=−r̂ 的流入 BC
(或者足够大的 domain 让 shock 不到边);cart_ale2 只支持
periodic / reflect,reflect 会把 IC 里 boundary 处的 v=−r̂ 直接
clamp 为 0,**掐断 IC 能量来源**,导致 ρ_post 根本无法达到 16。

2026-05-07 实测 ρ_post ≈ 0.23(t=2.0),对应 **IC 被 BC 破坏
后的残余演化**,而不是 Noh 物理。

**保留此 IC 的意义**:一旦以后加了 "driven inflow" BC 或
"outflow" BC,直接 `--test noh` 即可;现在不算基础正确性
回归项。

---

## 4. Gresho stationary vortex

**物理**:稳态 2D Euler 解。v(r) 三段:
- r < 0.2 : vφ = 5r,      P = 5 + 12.5 r²
- 0.2 ≤ r < 0.4 : vφ = 2−5r, P = 9 + 12.5 r² − 20r + 4 ln(5r)
- r ≥ 0.4 : 0,             P = 3 + 4 ln 2

应当 **精确保持不动**。AV 在纯旋转下误触发会导致速度衰减。
ρ=1,γ=1.4,域 [0,1]²。

**Command**:
```bash
./stellar2d --solver cart_ale2 --test gresho \
    --nr 128 --ntheta 128 --tend 3.0 \
    --cfl 0.3 --remap-order 2 --bc-x reflect --bc-y reflect \
    --vtk-dt 3.0 --run-dir runs/gresho_128
```

**Check** (C++ compute_error + pytest tst/):
```bash
cd tst && STELLAR2D_BIN=../build/stellar2d pytest test_ale2/test_ale2_gresho_gpu.py -v
```
`scripts/tests_ale2/gresho_compare.py` removed — error scored in C++
via `CartAle2Solver::compute_gresho_error` (inside r<0.5 disk).

**Expected**:`L1 drift` < 0.02 (v_max=1 → 2% 相对衰减)。如果
`--shear-aware-av 0` 关 shear weight 则 drift 会明显增大,正好
展示 shear-aware AV 的效果。

---

## 5. Yee-Vinokur-Djomehri isentropic vortex

**物理**:等熵 vortex 叠加在 (u∞,v∞)=(1,1) 平流场上,全周期 BC 域
[0,10]² 居中 (5,5)。t=10 正好平流一个域,应当完全回到 IC。
测的是 **remap 数值耗散** + 平流 phase error。γ=1.4, β=5。

**Command**:
```bash
./stellar2d --solver cart_ale2 --test yee_vortex \
    --nr 256 --ntheta 256 --tend 10.0 \
    --cfl 0.3 --remap-order 2 --bc-x periodic --bc-y periodic \
    --vtk-dt 10.0 --run-dir runs/yee_256
```

**Check** (C++ compute_error + pytest tst/):
```bash
cd tst && STELLAR2D_BIN=../build/stellar2d pytest test_ale2/test_ale2_yee_gpu.py -v
```
`scripts/tests_ale2/yee_compare.py` removed — error scored in C++
via `CartAle2Solver::compute_yee_error`. NOTE: pytest test uses
short-t (t=1.0) for the fast bucket; full t=10 round-trip convergence
is tracked in this README, not the fast CI.

**Expected**:`L1(ρ − ρ_IC)` < 0.02(256²,2nd-order MUSCL)。
PPM-prim 应当更好,L1 < 0.01。如果 L1 > 0.05 基本是 remap
1st-order(`--remap-order 1` 会这样)。

---

## 跑一把所有 5 项

**快速验证**(Sod / Gresho / Yee 走 pytest,2026-05-07 起):

```bash
cd tst
STELLAR2D_BIN=../build/stellar2d pytest test_ale2 -m fast -v
# 或者 ctest -L fast 在 build/ 跑(自动设环境变量)
```

**完整 convergence**(Sedov / Noh 仍用 python,无 C++ 解析解):

```bash
for t in sedov2d noh; do
    ./stellar2d --solver cart_ale2 --test $t --nr 512 --ntheta 512 ...
    python3 scripts/tests_ale2/${t}_compare.py --run-dir runs/${t}_512
done
```

推荐:第一次做 solver 改动后 `ctest -L fast` 扫一遍;
正式 PR/paper 前用 512² 跑 Sedov / Noh 看 convergence。

## 调试建议

- **任何 FAIL 先看** `diagnostics.csv` 看 KE+IE 守恒漂移 —
  E 漂移 > 1% 说明 compatible update 有问题,不是 IC 或 limiter
- **Sod / Sedov fail** 先换 `--remap-order 1` 看是不是 MUSCL/PPM
  reconstruction bug
- **Noh fail** 试 `--cq-lin 0.5 --cq-quad 2.0`(调 AV 系数)
- **Gresho fail(drift 大)** 先验 `--shear-aware-av 1`(应当默认开)
- **Yee fail** 看 L∞:如果集中在域边界附近,是 periodic sync 问题
  (见 P30/P31 in CLAUDE.md)

## 文件

- IC 实现:`src/gpu/ale/cart_ale2_solver.cu` (`init_sedov`, `init_noh`,
  `init_gresho`, `init_yee_vortex`),头文件 `cart_ale2_solver.cuh`
- Driver dispatch:`src/drivers/cart_ale2.cpp`
- test_case 白名单:`src/sim/setup.cpp`(grid-less branch)
- pytest tests(Sod / Gresho / Yee,2026-05-07 起):
  `tst/test_ale2/test_ale2_{sod,gresho,yee}_gpu.py`,读 C++ 端写的
  `<test>-errors.dat`(`CartAle2Solver::compute_{sod,gresho,yee}_error`)。
- 分析脚本(仅 Sedov / Noh 仍用 python):
  `scripts/tests_ale2/{sedov,noh}_compare.py` + `_common.py`。
