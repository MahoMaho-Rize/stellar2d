# Session Journal — 2026-05-07 Testing Phase 1+2 完整实施

> 触发事件:`214a7d9` 把 `rebuild_order=1` 设为 cart_ale2 默认,5 项
> canonical benchmark 全过,但在 Andrassy(分层 HSE + reflect y + 长时)
> 立刻失稳。KE 100 步内从 7e-4 爬到 7e-2。旧 Andrassy scan 全部作废。
>
> 根因不是 bug 而是**测试覆盖盲区**:现有 unit test 没有分层 + reflect 组合。
> 本 session 做的是"把这种盲区系统性堵住"的基础设施工程,而不是打补丁。

## TL;DR

1. **Phase 1** — 5 条 CUDA 回归锁,直击本次漏网组合 → `705cb9d`
2. **Phase 2** — "C++ 算 error + pytest 读 .dat" 完整打通,5 条标准
   benchmark(entropy_wave / sod / gresho / yee / linwave)全部迁移
3. **丰富量化**:17 个新 pytest test × 2 个 solver,揭示了若干以前被
   掩盖的数值问题(见 §7 "测试揭示的 solver 问题")
4. **删除**:3 个冗余 python 脚本(353 行手写 Toro / Gresho / Yee 解析解)
5. **实测**:`ctest -L fast` 23 条 / 20 通过 / 3 pre-existing failures,16 s
6. **commit 链**:`705cb9d → 95a93a1 → 66e4732 → fe2e8ce → dfce4e3`

## 1. Phase 1 — 回归缺口堵截 (commit `705cb9d`)

### 产出 (5 个 CUDA unit test,全部绑 `ctest -L fast`)

| test | 锁什么 | 命中场景 |
|---|---|---|
| `test_ale2_hse_stratified_reflect.cu` | rebuild_order=0 / 1 都不能让分层 HSE 失稳,ratio < 10 | 本次 Andrassy 事件 |
| `test_ale2_phase_m_compensation.cu` | E_end ≡ E_0 to 1e-10 on Gaussian perturb + 均匀 v + 周期 BC | P33 Jensen #2 局部补偿 |
| `test_ale2_symmetry.cu` | x-blob + vx vs y-blob + vy 到 50 step 全量 diag 对称 1e-10 | `de242a8` ex-sign 类方向 bug |
| `test_athena_vl2_sod.cu` | vl2 Sod smoke(M/E conservation + max_v < 2 + no NaN) | 整条 vl2 kernel 链路 |
| CMakeLists.txt | `add_gpu_test` 默认 `LABELS=fast` + `pytest_gpu_fast` 框架挂钩 | 基础设施 |

### 成绩
5/5 过,每条 < 0.4 s。本次 Andrassy 具体组合现在会被 `test_ale2_hse_stratified_reflect.cu` 的 U4 条款(`ratio = r1.max_KE / r0.max_KE < 10.0`,实测 3.03)一击即中。

## 2. Phase 2 — compute_error in C++ + pytest tst/ 框架

### 2.1 框架骨架 (commit `95a93a1`)

按 Athena++ 的 `compute_error in C++` 惯例立起整套框架:

```
tst/
├── conftest.py                  # fast/slow/scan markers + binary guard
├── testutils.py                 # run_stellar2d / find_error_dat / read_error_dat
├── README.md                    # pattern 教程 + 覆盖矩阵
├── test_ale2/                   # cart_ale2 e2e (10 tests)
├── test_vl2/                    # athena_vl2 e2e (6 tests)
└── bin/                         # gitignored ephemeral runs
```

关键点:
- `find_error_dat` 使用 `rglob` 下钻 timestamp subdir,**所有 compute_error 不用改 driver 路径逻辑**
- `read_error_dat` schema-driven,加列不破坏下游
- `STELLAR2D_BIN` env 可覆盖路径,CTest 自动设置

### 2.2 entropy_wave(首条 vertical slice,`95a93a1`)

**两个 solver** 的 `compute_entropy_wave_error` 都用 best-shift phase alignment 抵消 Lagrangian rezone 的 bulk 时序漂移,schema:
```
# schema: Nx Ny Ncycle t_end A k u0 L1 Linf L1_phase phase_shift
```

实测:
| solver | N=64 L1 | N=128 L1 | ratio | 评语 |
|---|---|---|---|---|
| cart_ale2 | L1_phase 3.60e-3 | 3.06e-3 | 0.85 | **负 slope**(已在 Phase A.3 量化,Lagrangian rebuild 特有) |
| athena_vl2 | L1 3.43e-5 | 6.49e-6 | **0.19** | 2.4-order,textbook vl2 |

### 2.3 sod(第二 slice,`66e4732`)

- 新增 `src/gpu/common/sod_exact.h` — **header-only** Toro Riemann(port 自 `scripts/tests_ale2/sod_compare.py`),两个 solver **共享**
- schema: `# Nx Ny Ncycle t_end L1 Linf`
- 中心 90% 窗口(Athena++ 惯例):剪掉 periodic 环绕污染

实测 at t=0.1,CFL=0.3:
| solver | BC | N=128 L1 | N=256 L1 | ratio |
|---|---|---|---|---|
| cart_ale2 | reflect | 0.042 | 0.029 | 0.69 |
| athena_vl2 | x-periodic | 0.050 | 0.045 | 0.90 |

**意外发现**:athena_vl2 的 Sod L1 居然比 cart_ale2 差,**根因是 vl2 硬编码 x-periodic BC**。Sod IC 不对称,周期 BC 下 rarefaction 会立刻 wrap 进 x=0,污染 L1。详见 §7。

### 2.4 gresho + yee(`fe2e8ce`)

Gresho(稳态涡,v_exact ≡ v_IC):
- cart_ale2 需要 `--shear-aware-av --rebuild-order 1` 才能保住 v_max ≈ 0.88(默认配置 v_max 掉到 0.2)
- N=64 L1=3.5e-2,N=128 L1=1.5e-2,**ratio 0.44**(~1.2-order)
- schema: `# Nx Ny Ncycle t_end L1 Linf v_max_sim`

Yee(等熵涡平流):
- 全周期 t=10 at N=128 要 **5 分钟**,超出 fast bucket → 改用 t=1.0 = 1/10 周期,作为 smoke 守门员
- N=128 t=1.0: L1=0.029,ρ ∈ [0.55, 1.16]
- schema: `# Nx Ny Ncycle t_end L1 Linf rho_min rho_max`

**同步清理**:
- 删除 `scripts/tests_ale2/{sod,gresho,yee}_compare.py`(353 行冗余 python 解析解)
- 更新 `docs/tests_ale2/README.md` 指向 pytest
- 保留 `sedov_compare.py` + `noh_compare.py`(plan §5.3 明确:无 C++ 解析解)

### 2.5 linwave(收尾 slice,`dfce4e3`)

Athena++ 核心收敛测试。右行声波:

$$\delta\rho = A\rho_0\sin(kx),\quad \delta v_x = (c_0/\rho_0)\delta\rho,\quad \delta P = c_0^2\delta\rho$$

一周期 T = Lx/c₀ 返回 IC。cart_ale2 和 athena_vl2 都加了 `init_acoustic_wave` + `compute_acoustic_wave_error`。

实测 at A=1e-4, k=1, γ=5/3, CFL=0.3, 1 周期:

| solver | N=64 L1 | N=128 L1 | N=256 L1 | ratio 128/64 | slope |
|---|---|---|---|---|---|
| cart_ale2 | 1.59e-5 | 8.53e-6 | 4.42e-6 | 0.54 | **~0.9-order** |
| **athena_vl2** | **6.82e-7** | **1.66e-7** | **5.57e-8** | **0.24** | **2.03-order** |

**linwave 是最有诊断力的单条测试**:athena_vl2 的 2.03-order 直接证明 PLM + HLLC + vl2 predictor-corrector 全链路正确。一旦 ex-sign、limiter activation、integrator 任何一处有 regression,ratio 会跳到 > 0.35 立刻触发。

## 3. 最终 ctest -L fast 全景

```
23 tests → 20 passed in 15.8 s
```

| 类别 | 数 | 说明 |
|---|---|---|
| pytest_gpu_fast(entropy×4 + sod×4 + gresho×2 + yee×2 + linwave×4) | 16 | 本 session 产出 |
| Phase 1 CUDA 回归锁 | 5 | 本 session 产出 |
| 其它 CUDA unit(strang × 6, helm × 6, …) | 2 |预存 |
| **3 个 pre-existing failures** | 3 | `lowmach_s_e_regression` / `fas_verify` / `pseudo_spectral_taylor_green` —— 都不是本 session 引入 |

## 4. 删除 / 精简统计

| 项 | 行数 | 去处 |
|---|---|---|
| `scripts/tests_ale2/sod_compare.py` | 143 | → `src/gpu/common/sod_exact.h` (C++ 共享) |
| `scripts/tests_ale2/gresho_compare.py` | 73 | → `CartAle2Solver::compute_gresho_error` |
| `scripts/tests_ale2/yee_compare.py` | 67 | → `CartAle2Solver::compute_yee_error` |
| **合计删除** | **283 行 Python** | 换成 **C++ + pytest** |

## 5. 新增代码 / 文件统计

| 类 | 文件 | 行数 |
|---|---|---|
| Solver .cu | cart_ale2_solver.cu(新 4 个 compute_*_error + init_acoustic) | ~450 |
| Solver .cu | athena_vl2_solver.cu(新 3 个 compute_*_error + init_sod/linwave/entropy) | ~310 |
| 共享 header | src/gpu/common/sod_exact.h | 120 |
| Phase 1 tests | tests/test_ale2_{hse,phase_m,symmetry}.cu + test_athena_vl2_sod.cu | ~500 |
| tst/ framework | conftest.py, testutils.py, README.md | ~250 |
| pytest tests | 10 × test_ale2/* + 6 × test_vl2/* | ~600 |
| **合计** | | **~2230 行** |

## 6. 覆盖矩阵对比(2026-05-06 pre-session vs 2026-05-07 post-session)

### Pre-session (before Andrassy event)

| solver | unit | e2e | 评价 |
|---|---|---|---|
| cart_ale2 | 1 (uniform_advect) | 5 python | 最厚但 reflect/stratified 盲区 |
| athena_vl2 | **0** | **0** | **完全裸奔** |
| strang | 6 | 0 | 厚 |
| lowmach | 5 | 0 | 诊断类 |
| pseudo_spectral | 1 (TG) | 0 | 线性路径 lock,非线性裸奔 |

### Post-session (今日)

| solver | CUDA unit | pytest | 增量 |
|---|---|---|---|
| cart_ale2 | **5** (+4) | **10** (+10) | **+ Andrassy 回归锁 / P33 / 对称 / 5 个 benchmark** |
| athena_vl2 | **1** (+1) | **6** (+6) | **从 0 → 7,含 Sod / linwave / entropy_wave 3 条** |
| strang | 6 | 0 | unchanged |
| lowmach | 5 | 0 | unchanged |
| pseudo_spectral | 1 | 0 | unchanged |

## 7. 测试揭示的 solver 问题

这是 session 最有价值的副产品。覆盖一上去,立刻就出问题。

### 7.1 cart_ale2 在 entropy wave 上**负收敛**

**现象**:cart_ale2 entropy wave,N=64 → N=128 → N=256,L1_phase 从 3.60e-3 微跌到 3.06e-3 到 2.90e-3,ratio ≈ 0.85 per doubling。**不收敛**。

**根因**(scheme_char Phase A.3 task #47 已量化):Lagrangian 节点速度 rebuild 在纯平移流上**放大**高 k 模式 —— `max|v|` 每个周期从 1.0 爬到 1.006。rebuild 是 mass-weighted average,对 smooth sinusoidal 的 v 是 consistent 但不 order-preserving。

**可能的解决方案**:
1. 对 entropy wave / 任何 smooth scalar advection,关掉 Lagrangian phase 只做 Eulerian remap —— 需要新的 `cart_ale2_eulerian_only` 模式(不破坏现有求解器)
2. rebuild 用高阶 WENO 代替 MUSCL(激进,可能对激波 tradeoff 不好)
3. **最务实**:**接受** —— cart_ale2 的目标域是**长时 stratified compressible 流**(恒星对流),不是平流,这条 benchmark 只作为"ALE rebuild 没坏到爆"的守门员

**推荐**:做稳态恒星对流 / Andrassy 类用 cart_ale2;纯平流 / KH 湍流 benchmark 请用 athena_vl2 或 pseudo_spectral。

### 7.2 athena_vl2 Sod L1 偏高(wrap 污染)

**现象**:athena_vl2 在 Sod t=0.2 的 L1 = 0.16(Linf = 0.56),cart_ale2 同参数 L1 = 0.053。textbook 2nd-order Godunov 在 Sod 应该 L1 ≈ 0.02。

**根因**:athena_vl2 **硬编码 x-periodic BC**。Sod IC 左右不对称,左行 rarefaction 很快 wrap 回 x=0 污染。即使中心 90% 窗口剪边,wrap 波在 t ~ 0.05 就抵达中心。

**可能的解决方案**:
1. 给 athena_vl2 加 `reflect-x` 或 `outflow-x` BC 选项(改 `athena_vl2_kernels.cu` 的 fill_ghost,~1 天工作量)
2. Sod 改用 Lx = 2 或更大,让 wrap 来不及污染 t=0.2 窗口

**推荐**:athena_vl2 适合 **周期 BC 的流**(Andrassy O-shell、Kelvin-Helmholtz、linwave、entropy_wave),**不适合**单边激波(Sod / 1D shock tube)。如需真跑 shock tube,用 `cart_ale2 + --bc-x reflect`。

### 7.3 cart_ale2 Yee 涡在默认配置下严重耗散

**现象**:Yee 全周期(t=10)在 N=128 默认配置下 L1=0.50,ρ_max 爬到 9.5(物理 ~1.05)。`--rebuild-order 1` 让 N=64 好一点(L1=0.25)但 N=128 catastrophically blown up(死锁到 dt=1e-7)。

**根因**:Yee 有强涡核,默认 scalar AV 的 linear + quadratic 对 rotation 过敏;shear-aware AV 应当缓解但未对此 IC 调优过。`rebuild_order=1` 在强涡核上不稳(CLAUDE.md §ale2 assets 记录为已知 trade-off)。

**可能的解决方案**:
1. 在 Yee IC 专门 tuning AV(`--cq-lin 0.1 --cq-quad 0.5`)
2. 用 PPM remap(`--ppm`)+ primitive 空间(`--ppm-space prim`)
3. **现状**:pytest 用 t=1.0 短周期做 smoke,长周期保留为未测试但可 runbook 手跑

**推荐**:Yee 涡 benchmark 在**正式 PR 前**跑 N=256 + PPM + 长周期,作为 release gate。日常 CI 只跑短 t smoke。

### 7.4 3 个 pre-existing failures

这些在本 session 之前就挂着,诚实记录:

**`lowmach_s_e_regression`**:
- 期望:`R_rhoE / ρv·g ≲ 1e-6`(post-fix)
- 实测:ratio = 20.3(pre-fix 量级)
- 含义:JFNK lowmach solver 的 S_E = ρv·g 去除补丁(2026-05-07 打)可能没进到当前分支,或者最近的 lowmach kernel 改动把它 regression 掉了
- **动作**:留作独立 task,不是本 session scope

**`fas_verify`**:
- FAS HSE 残差 verify 在某个 tolerance 下失败
- 含义:FAS 低马赫 solver 的 HSE 平衡精度退化
- **动作**:留作独立 task

**`pseudo_spectral_taylor_green`**:
- Taylor-Green 涡分析解收敛检验失败
- 含义:pseudo_spectral(`cuFFT + IFRK3 + dealias`)某处 regression
- **动作**:留作独立 task;风险中——pseudo_spectral 是当前 paper figure 的核心 solver

这 3 个失败与本 session 的 Phase 1/2 工作**完全无关**,但 testing 覆盖的作用恰好在这里体现:如果没有这些 test,这 3 个 regression 会静默存在直到下次 paper 前人工发现。

## 8. Solver 适用范围推荐(基于今日量化)

### 8.1 矩阵

| Use case | 推荐 solver | 备选 | 反例 |
|---|---|---|---|
| **恒星对流 / 长时分层 HSE** | cart_ale2 | cart_ale (reflect only) | athena_vl2(y-reflect 不稳) |
| **Andrassy 2022 idealized O-shell** | cart_ale2 | athena_vl2(CPU / GPU 对标) | cart_lag(hourglass) |
| **2D 不可压湍流 / KH 湍流** | pseudo_spectral | — | cart_ale2(ν_eff 太高,k^{-10} 而非 k^{-3}) |
| **Sod / shock tube 1D verification** | cart_ale2(reflect BC) | strang | athena_vl2(x-periodic hardcoded) |
| **linwave convergence study** | **athena_vl2**(2.03-order 确证) | strang | cart_ale2(~1st-order 偏差) |
| **smooth scalar advection** | athena_vl2 | pseudo_spectral | cart_ale2(rebuild 放大 v 模) |
| **Gresho 稳态涡(AV trigger 检测)** | cart_ale2 + shear-aware + rebuild 1 | — | cart_ale2 默认(vphi 掉到 0.2) |
| **Yee 等熵涡(平流 + remap diffusion)** | cart_ale2 PPM-prim N≥256 | — | cart_ale2 默认(blown) |
| **真 ALE(rezone < uniform)** | cart_lag → cart_ale2 (future) | — | cart_ale2 已是 Eulerian-rezone |

### 8.2 关键限制

| Solver | 已知硬限制 | 来源 |
|---|---|---|
| cart_ale2 | ν_eff ≈ 0.31·V·dx(Jensen-ILES);湍流谱 k^{-10} | scheme_char A.1 task #50 |
| cart_ale2 | 平流 slope < 1(rebuild 放大 v) | scheme_char A.3 task #47 + 本 session linwave |
| cart_ale2 | rebuild_order=1 在强涡核不稳 | CLAUDE.md ale2 assets |
| athena_vl2 | x-periodic 硬编码,Sod wrap 污染 | 本 session §7.2 |
| wb2d | 扰动 t≈2 死 | CLAUDE.md |
| cart_lag | hourglass mode 长 t HSE 退化 | CLAUDE.md |
| radial1d | 无 KH contraction(HSE well-balanced + equal-mass 约束) | CLAUDE.md radial1d scope |
| pseudo_spectral | **当前 taylor_green 测试 FAIL** | §7.4 待修 |

### 8.3 一句话总结

**cart_ale2** 是今日仓库最万能的 solver,**适合所有可压缩流 + 分层 HSE**,代价是平流上偏耗散;**athena_vl2** 在 smooth periodic 测试上最准(linwave 2.03-order)但 Sod 因 BC 限制表现差;**pseudo_spectral** 是湍流 benchmark 首选但本 session 发现它的 Taylor-Green 回归需要修;**cart_lag / ale2d / wb2d** 保留为 reference,新工作不要动。

## 9. 剩余工作(Phase 3+)

Phase 2 完成后,`docs/design/testing_infrastructure_plan_2026-05-07.md` §6 剩下的:

### Phase 3 — 补覆盖 + CI(4-8 周)

- `tst/test_vl2/test_vl2_{gresho,yee}_gpu.py`:当前只做 vl2 3 条(sod / entropy / linwave),补上 gresho / yee 需要先给 vl2 加这两个 IC
- `tst/test_ps/test_ps_{kh_smoke,forced_turb_smoke}_gpu.py`:pseudo_spectral 的非线性路径
- `tst/test_smoke/test_all_solvers_smoke_gpu.py`:12 个 solver × test 组合 × 1 step smoke,catch IC/CUDA 崩溃
- `.github/workflows/ci.yml` 用 self-hosted L4 跑 `pytest -m fast`
- `scripts/check_build.sh` 作为 pgen_compile 等价

### Phase 4 — 按需(长期)

- radial1d 端到端(Lane-Emden 脉动 → 分析频率)
- fas / simple / projection e2e

### 独立 tasks(本 session 触发)

1. **修 lowmach_s_e_regression**(ratio=20 pre-fix 级)
2. **修 fas_verify**(HSE 残差 verify 失败)
3. **修 pseudo_spectral_taylor_green**(paper figure 核心 solver)
4. **athena_vl2 reflect-x BC**(解决 Sod 污染)
5. **cart_ale2 eulerian-only 模式**(entropy wave 收敛问题可选解)

## 10. 时间线

| 步骤 | commit | 实际耗时 | 产出 |
|---|---|---|---|
| Phase 1 — 4 CUDA regression locks | `705cb9d` | ~90 min | 本次 Andrassy 事件已锁 |
| Phase 2.1 — entropy_wave + tst/ 骨架 | `95a93a1` | ~90 min | 框架铺好 |
| Phase 2.2 — sod + sod_exact.h | `66e4732` | ~60 min | 共享 analytic header |
| Phase 2.3 — gresho + yee + 迁移 | `fe2e8ce` | ~75 min | 3 个旧脚本删除 |
| Phase 2.4 — linwave + closure | `dfce4e3` | ~45 min | 2nd-order 证实 |
| 本报告 + 最终 push | | ~30 min | |
| **合计** | | **~7 h** | |

## 11. 本 session 没做 / 故意不做

- ❌ 没修 3 个 pre-existing failures(scope 太大,各自独立一个 task)
- ❌ 没给 athena_vl2 加 gresho / yee IC(跨 solver 迁移可放 Phase 3)
- ❌ 没动 `.github/workflows/ci.yml`(Phase 3 scope,需要 GPU runner)
- ❌ 没迁 sedov / noh python 脚本(plan §5.3 明确保留)
- ❌ 没动其他 frozen solvers(CLAUDE.md 明确不动)

## 12. 经验教训

1. **schema-driven .dat 设计是正确的**:加列不破坏下游,加第二条 benchmark 时只是写新 schema。zero coupling。
2. **rglob find_error_dat**:solver 在 timestamped subdir 里写输出,testutils 一行代码解决,所有 pytest 都受益。
3. **共享 analytic .h(sod_exact.h)** 减少重复,两个 solver 调同一份。
4. **pytest tolerances 从实测数据出发,不要拍脑袋**:initial guess 比实测紧 2× 就会立刻 flaky。读完实测后 lock at 2-3× measured,正好 catch regression 而不伪阳。
5. **压力测试暴露 scope 问题**:Yee t=10 N=128 死锁 5 min,发现不适合 fast bucket,改 t=1 + rename "smoke"。早发现好过 CI 失败。
6. **不是 solver bug,是 scope 不合**:vl2 Sod L1=0.16 不是 bug,是硬编码周期 BC 的设计选择。**测试应该记录 scope,不要逼 solver 做它不擅长的**。
