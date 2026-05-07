# stellar2d 测试体系建设方案

> 创建日期:2026-05-07
> 作者:MahoMaho-Rize + Claude(敏捷 2 个月后的第一次 QA 停顿)
> 触发事件:`214a7d9` 把 `rebuild_order=1` 作为 default 上游,5 项标准 benchmark 全过,
> 但在 Andrassy 2022(反射壁 + 分层 + 长时)立刻失稳。KE 在 t<100 从 7e-4 跳到 7e-2。
> 旧 scan 数据因此全部作废,10-run ale2 scan 重跑。
> **这是一个本可由测试抓住的 regression**,只是当前测试框架对"反射壁 + 分层"不覆盖。

## TL;DR

- 当前测试体系分布零散:少量 CUDA unit(strang/lowmach/pseudo_spectral/ale2_uniform_advect)+
  5 个 ale2 end-to-end python compare 脚本;athena_vl2 零测试;ale2d/cart_lag/cart_ale/wb2d/
  anelastic_sl/sph2d_spectral 完全裸奔。
- 拟分 4 个 Phase 建设,吸收 Athena++ / AthenaK 经验但适配我们 GPU + 多 solver 异质
  的现实。Phase 1 堵住 Andrassy regression 类缺口(1 周),Phase 2 做 compute_error
  in C++ + pytest 框架(3 周),Phase 3 补完覆盖 + CI(4 周),Phase 4 按需扩展。
- **不**引入重量级依赖(Kokkos/Catch2/GoogleTest/Lcov)。用 Python pytest + 已有 CUDA
  unit 模板。

## 1 现状盘点(2026-05-07)

### 1.1 已有测试

| solver / 模块 | 单元 / 回归测试 | 端到端脚本 | 覆盖评价 |
|---|---|---|---|
| `cart_ale2` | `tests/test_cart_ale2_uniform_advect.cu`(P30/P31 lock,64²,200 step,bc_mode=3) | `scripts/tests_ale2/{sod,sedov,noh,gresho,yee}_compare.py` | **最厚** — 1 unit + 5 e2e |
| `athena_vl2` | 无 | 无 | **零** |
| `pseudo_spectral` | `tests/test_pseudo_spectral_taylor_green.cu`(FFT+IFRK3+dealias 综合 lock) | 无 | 单解析解 |
| `lowmach` (JFNK) | `test_lowmach.cu`, `test_lowmach_s_e_regression.cu`(5-07 加),`test_precond_quality`, `test_newton_tuning`, `test_eps_sweep`, `test_mini`, `test_solver_diagnosis` | 无 | 厚,但多为诊断(NO_CTEST) |
| `strang` | `test_strang_{init, muscl, hllc, step, unit, convergence}.cu` | 无 | **最成熟** — 6 按模块拆分 + 1 收敛 |
| `fas`/`simple`/`projection` | `test_fas_verify.cu` + `test_fas_diagnose_hse.cu` | 无 | 薄 |
| `radial1d` | 无专属(借用 Helm/Dual tests) | 无 | 间接 |
| `ale2d`, `cart_lag`, `cart_ale`, `wb2d`, `anelastic_sl`, `sph2d_spectral` | **0** | **0** | 裸奔 |

### 1.2 5-07 session 新加的 3 个 untracked test file(写得好)

- `tests/test_cart_ale2_uniform_advect.cu` — 测 bc=3 的 P30/P31 lock。**但只 bc=3,反射壁未覆盖**,
  Andrassy 用的是 bc=1(x 周期+ y 反射),这次 regression 正好在这条隙里漏过
- `tests/test_lowmach_s_e_regression.cu` — 写得最好,3 个 sub-test 逐级 isolate ρv·g bug。
  写 "pre-fix ≈ 1, post-fix ≲ 1e-6" 的预期值,回归 guard 不会失灵。**模板级样板**
- `tests/test_pseudo_spectral_taylor_green.cu` — 单一 IC lock 5 条独立路径(FFT sign、ψ 逆、
  IFRK3、dealias、skew-sym convection)。非线性路径(KH、forced turb)未覆盖

### 1.3 `scripts/tests_ale2/*_compare.py` 的问题

1. **python 里重写解析解**:`sod_compare.py` 120 行 Toro Riemann,跟 `init_sod()` 两边维护,易偏
2. **Noh 事实上不通过**,README 标出但无替代测试
3. **无收敛阶检查** — 只单点 L1,`de242a8`(swept-remap 方向错,MUSCL 实际只有 1 阶)本可以被一条
   L1(128) vs L1(256) ratio < 0.5 断言抓到,早抓到两周
4. **依赖 `runs/` 预存结果**,不是 standalone。没 CTest 集成
5. **无对称性测试** — 我们 ex 号 bug 是典型的 x/y 方向错,一个 90° 旋转对称 test 直接 catch

## 2 Athena 工程经验选择性吸收

### 2.1 Athena++ regression 框架要点

- 每 test 一个 `.py` 实现 3 钩子:`prepare(**) / run(**) / analyze() -> bool`
- `run_tests.py` 用 `importlib + iter_modules` 按目录扫描,组织成 suite
- **error 计算在 C++ pgen 里**,`problem/compute_error=true` 打开后 `UserWorkAfterLoop` 写
  `linearwave-errors.dat`,python 只 `np.loadtxt` 读表 + `if data > tol: return False`
- 同 test 嵌套 flux/integrator/xorder/res 循环(Sod 里 3 flux × 3 方向 × 2 res = 18 runs)
- 写的是 **L1 比值 < 0.6**,严格验 2 阶收敛
- Lcov 做代码覆盖率,Jenkins `.sh` 编排
- 25 个 suite、~80 个测试脚本。最核心的"linear wave convergence"在 11 个 suite 里重复出现

### 2.2 AthenaK(GPU 版)经验

AthenaK 双框架并存:老的 `scripts/` + Athena++ 3 钩子 vs 新的 `test_suite/` + pytest。新的值得借鉴:

1. **`_cpu / _mpicpu / _gpu` 文件名后缀 + `pytest -k _gpu` 筛选**
2. **CPU 和 GPU 不跑同一 test** — Sod shock tube 只 CPU;Linear wave 3D AMR 有 CPU / MPI / GPU 三版;
   GPU 专注 2D/3D AMR 或 GPU-only physics(MRI)
3. **build 在 `run_test_suite.py` 入口 clean-build**,然后 pytest 按 `-k` 筛选
4. **GPU 容差比 CPU 松** — `(1.4e-05, 0.27)` 容差比 CPU Sod 的 `1e-something` 宽,**文档化 atomic
   non-associativity 带来的 run-to-run 噪声**
5. testutils.py 里 `run/make/cmake/mpi_run/cleanup/clean_make` 全套小工具,~300 行可复用

### 2.3 吸收与拒绝对照表

| 源 | 要点 | 采纳? | 备注 |
|---|---|---|---|
| Athena++ | prepare/run/analyze 3 钩子 + importlib | ⚠️ 改 | 采用 pytest 风格(AthenaK 已这样),不用 importlib |
| Athena++ | compute_error 在 C++,python 读 `.dat` | ✅ 核心 | 解决 sod_compare.py 120 行 python Toro 重写的尴尬 |
| Athena++ | linear wave 是核心 test | ✅ | 1 条抓时间积分 + 重构 + flux 的全部收敛性 |
| Athena++ | 同 test 多 flux/limiter/integrator 循环 | ✅ | ale2 已有 `remap-limiter`, `ppm-space`, `rebuild-order` 3 维度 |
| Athena++ | symmetry suite(x1/x2/x3 对称) | ✅ | ex 号 bug 可抓住 |
| Athena++ | pgen_compile smoke | ✅ 改 | 改成"每个 `--solver X --test Y` 跑 1 step" |
| Athena++ | Jenkins `.sh` 编排 | ❌ | 我们无 HPC queue,改 GitHub Actions + 本地 pytest |
| Athena++ | Lcov 代码覆盖率 | ❌ | 小团队 ROI 低 |
| Athena++ | 每 test 独立 athinput + ParameterInput 系统 | ❌ | 我们用 CLI flag,不重做 input 系统 |
| AthenaK | `_cpu/_mpicpu/_gpu` 命名约定 | ⚠️ 改 | 我们 `_gpu` 唯一,换 `_fast/_slow/_scan` 更有用 |
| AthenaK | GPU 容差比 CPU 松(文档化) | ✅ | atomic add 的 floating-point non-associativity |
| AthenaK | CPU 和 GPU 选等价但不同题目 | ❌ | 我们 CPU path 不维护 |
| AthenaK | `clean_make` 在入口做 | ❌ | CUDA 重编 5+ min,开发流里用增量 build |
| AthenaK | Kokkos 抽象 | ❌ | 我们 raw CUDA,不换 |

## 3 我们的现实约束

1. **只 GPU path**(memory.no_cpu_testing)— CPU build 不维护,所有 test 跑 GPU
2. **CI 预算 5-10 min** — 无 Jenkins queue,scan-类(Andrassy, 512²)永远手动
3. **多 solver 异质** — 12 个 solver 族共存,CLI/IC/BC 约定各异。测试按 solver 组织,不按 physics
4. **已有遗留 GPU unit test** — 保留,不重写

## 4 目标结构

```
stellar2d/
├── tests/                          # 已有 CUDA-level unit 测试(保留)
│   ├── test_strang_*.cu            # 保留不动
│   ├── test_cart_ale2_uniform_advect.cu
│   ├── test_lowmach_s_e_regression.cu
│   └── test_pseudo_spectral_taylor_green.cu
│
├── tst/                            # 新:端到端 pytest 驱动(借 AthenaK)
│   ├── conftest.py                 # pytest 全局 fixture(build 检查、run_base、cleanup)
│   ├── testutils.py                # run / read_error_dat / pass_fail(~150 行)
│   ├── athena_read_alike.py        # VTK/CSV 解析(复用 scripts/andrassy2022/diagnose.py)
│   ├── test_ale2/                  # 每 solver 一子目录
│   │   ├── __init__.py
│   │   ├── test_ale2_sod_gpu.py    # 读 sod-errors.dat,测 L1 + 收敛率
│   │   ├── test_ale2_linwave_gpu.py       # 新 — acoustic + entropy wave
│   │   ├── test_ale2_symmetry_gpu.py      # 新 — x/y 90° 旋转对称
│   │   ├── test_ale2_gresho_gpu.py
│   │   ├── test_ale2_yee_gpu.py
│   │   ├── test_ale2_sedov_gpu.py         # 保留 python 检查(无解析解峰值)
│   │   └── test_ale2_hse_stratified_gpu.py  # 新 — bc=1 反射 + 分层 HSE 保持(抓本次 regression)
│   ├── test_vl2/                   # 新 — athena_vl2 零测试补上
│   │   ├── test_vl2_sod_gpu.py
│   │   ├── test_vl2_linwave_gpu.py
│   │   └── test_vl2_symmetry_gpu.py
│   ├── test_ps/                    # pseudo_spectral(TG 已有 CUDA unit)
│   │   ├── test_ps_kh_smoke_gpu.py
│   │   └── test_ps_forced_turb_smoke_gpu.py
│   ├── test_smoke/                 # 每 solver × 每 test_case 跑 1 step
│   │   └── test_all_solvers_smoke_gpu.py
│   └── run_tests.sh                # 一键:pytest tst/ -m fast
│
├── scripts/tests_ale2/             # 旧:docs 还引用,保留到 Phase 2 末
│                                   # Phase 2 迁走后删除
│
└── .github/workflows/
    └── ci.yml                      # GPU runner: pytest tst/ -m fast
```

**命名约定**:

- `_gpu` 后缀:所有 e2e 测试(我们无 CPU 路径,仍保留以便将来清晰)
- `@pytest.mark.fast`:< 30 s 单 test,默认 CI 跑
- `@pytest.mark.slow`:几分钟,`pytest -m "fast or slow"` 才跑
- `@pytest.mark.scan`:小时级,永远手动

## 5 关键模式:**compute_error in C++**

这是最重要的借鉴,也是最大杠杆点。

### 5.1 现状痛点

```python
# scripts/tests_ale2/sod_compare.py(120 行)
def sod_exact(x, t, gamma=1.4, ...):
    # 手写 Toro / Sod Riemann solver
    # 与 init_sod() 在 src/gpu/ale/cart_ale2_solver.cu 里分开维护
    # 任意 IC 参数变化都需要两边同步
```

### 5.2 目标模式

```cpp
// src/gpu/ale/cart_ale2_solver.cu
void CartAle2Solver::compute_sod_error(double t_now, const std::string& run_dir) {
    // 内部跑一份 exact Sod(C++),对比 GPU download 的 ρ,算 L1/L∞
    // 把结果追加到 run_dir + "/sod-errors.dat",格式:
    //   # Nx Ny Ncycle L1_rho L1_vx L1_P Linf_rho
    //   128 128 12345 1.98e-03 2.1e-03 1.5e-03 ...
}
```

```python
# tst/test_ale2/test_ale2_sod_gpu.py
@pytest.mark.fast
def test_sod_convergence():
    for res in (128, 256):
        run([f"--nr={res}", "--test=sod", "--compute-error",
             f"--run-base=tst/bin/sod_{res}"])
    e128 = read_error_dat("tst/bin/sod_128/sod-errors.dat")[3]
    e256 = read_error_dat("tst/bin/sod_256/sod-errors.dat")[3]
    assert e256 < 0.06,            "256² Sod L1 超差"
    assert e256 / e128 < 0.6,      "收敛率 < 0.74 阶"
```

### 5.3 应用到的 test

| test | 解析解 | C++ compute_error | python 保持 |
|---|---|---|---|
| Sod | 有(Toro Riemann) | ✅ | 无 |
| Linear wave | 有(eigenvector · exp(-ikx)) | ✅ | 无 |
| Gresho vortex | 有(稳态) | ✅ | 无 |
| Yee vortex | 有(round-trip) | ✅ | 无 |
| Taylor-Green | 有(纯扩散) | 已在 CUDA unit 里 | 无 |
| Sedov | 自相似(需数值积分 ξ₀ 因子) | ❌ — python 继续 | ✅ |
| Noh | 自相似(需 BC 支持) | ❌ — 阻塞于 inflow BC | ✅ |

## 6 分阶段实施计划

### Phase 1:**堵缺口**(本周内,不重构)

**目的**:把最容易导致 silent regression 的缺口堵住,不搬基础设施。
全部是 `.cu` unit test 扩展,绑 CTest,不改 build system。

| # | 任务 | 预估 | 能抓什么 |
|---|---|---|---|
| 1.1 | `tests/test_ale2_hse_stratified_reflect.cu`<br>(bc_mode=1 分层 HSE,100 step,KE < 1e-6) | 0.5 天 | **本次 Andrassy regression** |
| 1.2 | `tests/test_ale2_phase_m_compensation.cu`<br>(非 uniform IC,10 step,断言 `(E_end − E_0)/E_0 < 1e-10`) | 0.5 天 | Phase-M compensation 回归 |
| 1.3 | `tests/test_ale2_symmetry.cu`<br>(x-IC 跑 50 step vs y-IC 90° 旋转 50 step,`max|KE_x − KE_y| < 1e-12`) | 0.5 天 | 方向相关 bug(e.g. ex 号) |
| 1.4 | `tests/test_athena_vl2_sod.cu`<br>(vl2 零测试 → 1 条最基础) | 0.5 天 | vl2 kernel 级回归 |
| 1.5 | CMakeLists.txt:1.1-1.4 绑 `ctest -L fast` | 0.2 天 | — |

**产出**:5 个新 `.cu` test,全部 CTest 集成。本次 regression 再发生会被 1.1 直接抓住。

### Phase 2:**compute_error in C++ + pytest 框架**(2-4 周)

| # | 任务 | 预估 |
|---|---|---|
| 2.1 | 在 cart_ale2_solver.cu 加 `compute_sod_error`, `compute_linwave_error`, `compute_gresho_error`, `compute_yee_error`;`--compute-error` 统一 CLI trigger | 3 天 |
| 2.2 | 建 `tst/` 目录、`conftest.py`、`testutils.py`(抄 AthenaK 精简到 ~150 行) | 1 天 |
| 2.3 | 迁移 5 个 `scripts/tests_ale2/*_compare.py` → `tst/test_ale2/*_gpu.py`,改成读 `.dat` | 2 天 |
| 2.4 | 加 `tst/test_ale2/test_ale2_linwave_gpu.py`(C++ 算 L1,pytest 循环 {plm, ppm_cs, ppm_cw} × {128, 256} × {hlle, hllc} 做 sweep) | 2 天 |
| 2.5 | 加 `tst/test_ale2/test_ale2_symmetry_gpu.py`(x/y 对称) | 1 天 |
| 2.6 | 所有 test 加 `@pytest.mark.{fast, slow, scan}` | 0.5 天 |
| 2.7 | `scripts/run_tests.sh` + README runbook 更新 | 0.5 天 |
| 2.8 | 删除旧 `scripts/tests_ale2/*` 或仅保 README 链接迁移 | 0.5 天 |

**产出**:tst/ 框架就位;ale2 全 benchmark 迁移完成;linwave 收敛率作为新核心 test。

### Phase 3:**补完覆盖 + CI**(4-8 周)

| # | 任务 | 预估 |
|---|---|---|
| 3.1 | `tst/test_vl2/test_vl2_{sod, linwave, symmetry}_gpu.py`(vl2 → 3 条) | 1 周 |
| 3.2 | `tst/test_ps/test_ps_{kh_smoke, forced_turb_smoke}_gpu.py` | 2 天 |
| 3.3 | `tst/test_smoke/test_all_solvers_smoke_gpu.py`(循环每 `--solver X --test Y` 跑 1 step,catch init) | 3 天 |
| 3.4 | `.github/workflows/ci.yml`:GPU runner(self-hosted 或 on-demand L4)跑 `pytest -m fast`,20 min 预算 | 1 周 |
| 3.5 | `tst/style/cpplint.sh`:照 Athena,针对 `src/gpu/*.cu` | 1 天 |
| 3.6 | `scripts/check_build.sh`:make clean + full rebuild 作为 "pgen_compile" 等价体 | 0.5 天 |

### Phase 4:**长期,按需**

- radial1d 端到端(Lane-Emden pulsation → analytic 频率)
- fas/simple/projection e2e
- 其他 solver(ale2d, cart_lag, wb2d, cart_ale, anelastic_sl, sph2d_spectral)按 "能不能抓回归" 为 ROI 决定;
  CLAUDE.md 标它们为 frozen 资产,不主动补,只在使用前追加

## 7 GPU-specific 设计决策

### 7.1 Tolerance 文档化,不跟 CPU 参照

每个 test docstring 标:

```python
"""Sod convergence test (cart_ale2 GPU).

Tolerances:
  - L1(ρ)/|Δρ| < 0.06 at 256²         (ALE+swept-remap 固有 ~5%)
  - L1(256)/L1(128) < 0.6            (~0.74-阶收敛,MUSCL + limiter shock 退化)
  - GPU atomic-add 带来的 run-to-run 噪声 < 0.1%(见 docs/gpu_atomic_noise.md)
"""
```

### 7.2 Bit-reproducibility 不强求

GPU atomic add 不满足结合律,同 seed 两次跑 L1 可能差 1e-10。不断言 bit-same,只断言:
- 物理守恒量在**同一运行内**达指定精度
- L1 vs 解析解的**阈值**

### 7.3 Smoke test 不跑到 tend

每个 `--solver X --test Y` 跑 **1 step** 或 **0.01 物理秒**,只 catch:
- IC loader crash(init_andrassy2022 读 slab 之类)
- CUDA launch error(kernel signature 不匹配)
- NaN 检测(`std::isnan(E_final)`)

### 7.4 fast/slow/scan 分层

| marker | 典型时长 | CI 行为 |
|---|---|---|
| `fast` | < 30 s | 默认 CI,每 PR 跑 |
| `slow` | 1-5 min | `pytest -m "fast or slow"`,nightly |
| `scan` | > 30 min | 手动,paper 前跑 |

### 7.5 `--run-base tst/bin/<test>` 隔离

每 pytest 函数用自己的 run_base,conftest.py 统一 cleanup。不要所有 test 共用 `runs/`。

### 7.6 GPU 容差相对 CPU 放宽原则

- 绝对量:CPU 严格 `< 1e-12` → GPU `< 1e-10`(atomic round-off)
- 收敛率:CPU 严格 `ratio < 0.3`(2 阶) → GPU `< 0.5`(limiter 变异 + atomic)
- 阶数:2 阶 formal 承认为 1.74-阶 practical(log_2(0.6) ≈ 0.74)

## 8 风险与不做

### 8.1 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| GitHub Actions GPU runner 贵/难获得 | 中 | Phase 3 开始前 POC on-demand L4(Lambda Labs 等) |
| `.dat` 格式漂移(加字段导致旧 test 读错列) | 低 | 每个 `.dat` 第 1 行 `# <schema>`,testutils 里按 header 解析 |
| pytest fixture 泄漏 GPU memory | 低 | `conftest.py` autouse fixture 在 teardown 强制 `cudaDeviceReset` |
| compute_error 在 GPU IC 下耗时太长 | 低 | 解析解 128² 在 CPU python 反而简单,C++ 只在 IC 完成时跑一次 + IC 参数 pass-through |
| 迁移期 ale2 benchmark 两套并存 | 中 | Phase 2.3 单次切换,旧脚本迁走后立刻删;docs/tests_ale2/README.md 同步更新 |

### 8.2 明确不做

- ❌ 不迁移已有 `tests/test_strang_*.cu`(工作正常)
- ❌ 不给 frozen 资产补 e2e(CLAUDE.md 说明不动)
- ❌ 不引入 Kokkos / Catch2 / GoogleTest
- ❌ 不做 Lcov 覆盖率(小团队 ROI 低)
- ❌ 不做 MPI 测试(我们单 GPU,无 MPI path)
- ❌ 不跑 CPU path(memory.no_cpu_testing)

## 9 验收标准

Phase 1 结束:**本次 Andrassy regression 在 CI 中会被 1.1 抓住** — 提交 `214a7d9` 前 `ctest -L fast` 红,不会 merge。

Phase 2 结束:
- Sod + linwave + gresho + yee 全部 C++ 算 error,python 只读 `.dat`
- 所有 ale2 benchmark 通过 `pytest tst/test_ale2 -m fast` 执行,5 min 内完成
- `scripts/tests_ale2/*` 删除

Phase 3 结束:
- athena_vl2 有 sod/linwave/symmetry 3 条 e2e
- `test_all_solvers_smoke_gpu.py` 覆盖 10+ solver × test 组合
- `.github/workflows/ci.yml` 在每 PR 跑 `pytest -m fast` < 20 min

## 10 附录:Phase 1 任务具体规格

### 10.1 `test_ale2_hse_stratified_reflect.cu` 详规

```cpp
// 目的:lock 2nd-order rebuild + reflect wall + 分层 HSE 不会失稳
// 这是本次 2026-05-07 Andrassy regression 的最小复现
//
// IC: ρ(y) = ρ_top + (ρ_bot - ρ_top)·(1 - y/Ly)
//     P(y) 满足 HSE: dP/dy = -ρ·g,g = 1.0 向下
//     v = 0
// BC: x periodic, y reflect
// Run: 100 step, CFL=0.3
//
// 对每个 rebuild_order ∈ {0, 1}:
//   U1. max(KE) over 100 steps < ε₀
//        ε₀(0) = 1e-6 (1st-order rebuild 稳)
//        ε₀(1) = 1e-4 (2nd-order 可能轻微爬升,但不能爆)
//   U2. E_end / E_0 - 1 < 1e-8 (compensation 工作)
//   U3. max|ρ(t=end) - ρ_IC| < 1e-6 (静力学保持)
//
// rebuild_order=1 下 Andrassy-like 失稳表现:
//   KE 100 步内从 0 增长到 ~1e-1(完全失稳,失败)
//   本 test 会立刻报错 KE > ε₀
```

### 10.2 `test_ale2_phase_m_compensation.cu` 详规

```cpp
// 目的:lock Phase-M KE→IE compensation 对非 uniform 流有效
//
// IC: Gaussian ρ perturb over uniform 1,0 base;P=1;v=0;g=0
//     高斯峰值 ρ=1.3,宽度 0.1
// Run: 30 step,10 次对 compensation kernel 的 exercise
//
// U1. |E_end - E_0| / E_0 < 1e-10(机器精度)
// U2. M_end = M_0 到机器精度
// U3. 每一步 compensation 后 IE >= IE_before(不准损失 IE)
```

### 10.3 `test_ale2_symmetry.cu` 详规

```cpp
// 目的:抓 x/y 方向相关 bug(历史:ex 号 bug in swept-remap)
//
// Two runs:
//   Run A: init with a density blob at (xc=0.3, yc=0.5), vx=0, vy=0, no gravity
//          Advect 50 step with (u, v) = (0.5, 0)
//   Run B: 把 A 的 IC 绕 (0.5, 0.5) 旋转 90° (xc=0.5, yc=0.3),(u, v) = (0, 0.5)
//          Advect 50 step
//
// U1. total_KE_A(50) ≈ total_KE_B(50) 到 1e-10
// U2. diag.total_IE, total_E 对称性相同
// U3. 若做 download_xslice(A) 和 download_yslice(B) 后 90° rotate,pointwise 差 < 1e-8
```

### 10.4 `test_athena_vl2_sod.cu` 详规

```cpp
// 目的:vl2 solver 零单元测试的最小补全 — 至少验证启动不崩 + 物理合理
//
// IC: standard Sod(init_sod 里 vl2 的调用)
// BC: reflect x/y(vl2 目前默认)
// Run: tend=0.2,128²
// Checks:
//   U1. M_end == M_0 到 1e-12
//   U2. E_end > 0.8 * E_0(vl2 + outflow 的激波能量守恒)
//   U3. max_v < 2(不爆)
//   U4. 256² L1(ρ) < 0.05(中间 y 切片 vs sod_exact)
```

---

**附:本方案由 2026-05-07 Andrassy scan regression 事件触发。**

真正的回归抓漏:`214a7d9` commit 引入 `rebuild_order=1` 默认,通过 5 项 canonical benchmark
(Sod/Sedov/Noh/Gresho/Yee),但在未被任何 test 覆盖的组合(2nd-order MUSCL remap + 2nd-order
corner rebuild + 反射壁 + 长时分层对流)下触发 checkerboard 不稳定性。Andrassy scan 的 16 个
run 作废,ale2 scan 重跑中。Phase 1.1 的 HSE-stratified-reflect test 是本次教训的直接产物。
