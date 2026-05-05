# cart_ale 开发进度 — 2026-05-01

**Branch**: `ale2d-cartesian`
**状态**: cart_ale 第一版 working prototype 完成,已通过多组验证,代码已推送。
**下一步**: Kucharik-Shashkov 二阶 MUSCL-in-remap(Path 1)。

---

## 0. 一句话总结

今天从零做出一个**可工作的 2D Cartesian GPU ALE 求解器** `cart_ale`,通过了从线性声学到 1000% 非线性扰动到双泡对撞的完整 stress test,代码和演示视频已推送到 GitHub。

三天起步的 stellar2d 仓库现在有 6 个求解器并存:strang、fas/simple/projection/lowmach、radial1d、wb2d、ale2d、cart_lag、**cart_ale**。每一个都是独立资产,都不覆盖。

---

## 1. 起点:昨天的 cart_lag HSE 退化问题

- **现象**:cart_lag(纯 Caramana compatible Lagrangian)在 256² HSE 无扰动下 dt 从 2e-3 退化到 1e-13,t≈2.85 死。
- **诊断**:spurious hourglass mode(staggered 四边形的 zero-energy mode)+ 一阶 HSE defect 驱动 → 节点在零能通道持续漂移 → tangle 或 dt 退化。
- **已记录**:`docs/ale_rezone_design.md`(昨天写的)提出 Eulerian rezone + swept-edge conservative remap 方案。

## 2. 今天开干:cart_ale 新求解器(资产保留原则)

用户在 compact 后明确要求:**"已有代码跑起来存在问题但是物理上并不错误,只是不适合这个工作,这些都可以留着作为未来的资产"**。

→ **不修改 `cart_lag_*`**,开全新文件 `cart_ale_{solver,kernels}.{cuh,cu}`,并把规则写入:
- `CLAUDE.md`:项目级约定,列出所有求解器资产清单
- `memory/feedback_preserve_solvers.md`:全局记忆

### 2.1 架构设计

每步三阶段:
```
Phase L (Lagrangian):  Caramana compatible 推一步,mesh 变形
Phase M (Remap):       在原始和变形 mesh 之间做 swept-edge donor-cell remap
Phase R (Rezone):      mesh 复位到初始 uniform 位置
```

Rezone 完全 Eulerian(`mesh ← mesh_0`),保证 hourglass 没机会发作。

### 2.2 代码组成(~1600 行新增)

| 文件 | LOC | 内容 |
|---|---|---|
| `cart_ale_solver.cuh` | 100 | Solver struct,含 X0/Y0 reference mesh,remap 暂存 buffer |
| `cart_ale_kernels.cu` | 420 | Lagrangian kernel + swept-edge remap kernel(east/north 分开)+ velocity rebuild |
| `cart_ale_solver.cu` | 310 | Orchestration,三个 IC(uniform/sod/HSE polytrope,后加 bubble)+ diagnostics + VTK |
| `cart_ale_solver.cu::write_vtk_2d` | 65 | Cartesian STRUCTURED_GRID VTK(density/pressure/velocity/mach) |
| `main.cpp` dispatch | ~80 | `--solver cart_ale --test {hse,hse_perturbed,hse_bubble,sod}` |
| `scripts/render_cart_ale.py` | 280 | Cartesian MP4 renderer(从 render_video_fast.py 改来,保留全局固定色标) |

### 2.3 Swept-edge remap 核心

对原始 mesh 上每个 interior edge(east 或 north),Lagrangian phase 后四个节点是(A, A', B, B'),扫过的四边形面积:
```
swept_signed_area = shoelace(A → A' → B' → B)
```
正号 → 物质从 donor 转移到 receiver,反号 → 反向。

Donor-cell 一阶:直接用 donor cell 的 `ρ, ρe, p_x, p_y` 除以 donor volume 得"密度",乘以 swept volume 转移。atomic 原子加减。

动量:cell-centered `p_x = 0.25·Σ m_corner·v_corner`,remap 后按 adjacent cell 平均 rebuild node velocity。

---

## 3. 验证系列(按时间顺序)

### 3.1 HSE 无扰动 256² tend=10(冒烟测试)
- dt 稳在 1.197e-3 全程
- M 守恒机器精度
- |v| ~ 1e-5 震荡不发散
- **cart_lag 在 t=2.85 退化到 1e-13 的问题彻底消失**

### 3.2 HSE 无扰动 256² tend=1000(524k 步)
- dt 全程 1.9e-3
- ΔE/|E| = 2.5e-4
- KE 半衰期 96 时间单位
- **用户问:"扰动幅度很小啊?"** → 发现是默认 `perturb_amplitude=1e-3`,实际 0.1% 扰动

### 3.3 HSE 扰动 256² tend=1000 `--perturb 0.05`
- ΔE/|E| = 7.5e-3
- KE 初始 4.4e-5,半衰期 41 时间单位
- Mach 峰 6.9%

### 3.4 HSE 扰动 256² tend=1000 `--perturb 10`(1000% 扰动)
- **|v|peak = 0.40,Mach peak = 0.46** 跨声速
- ΔE/|E| = 3.3e-3
- IE 永久抬升 5%
- 527k 步全稳定

### 3.5 HSE 扰动 256² tend=20 `--perturb 10`(高时间分辨率)
- **发现 KE 峰间距 1.322 ± 0.027 = 声速穿越时间 Ly/cs ≈ 1.25**
- 即:代码自发找到基频径向 p-mode,物理真实的声学振荡周期
- 这是这次所有 run 中**物理意义最深**的发现

### 3.6 HSE 浮力泡 256² tend=50(α=-0.5,y=0.3,rb=0.1)
- 完整生命周期:Release → Rise(τ_B≈1.18)→ Impact + RT → Mixing → Relax
- KE peak 在 t=3.42(气泡冲顶)
- ΔIE=-2.6e-3(内能转化为 KE)

### 3.7 HSE 双泡对撞 256² tend=50
- 下:α=-0.7(低密度高熵) 上升
- 上:α=+1.5(高密度低熵) 下降
- |v|peak = 0.54,Mach peak = **0.59** 跨声速对冲
- ΔE/|E| = 1.51e-2

---

## 4. 关键技术问题和解决

### 4.1 可视化 VTK 坐标系
cart_ale 是 Cartesian 不是轴对称 → 原有 `render_cartesian.py` 用 (r,θ)→(x,z) 不适用。
→ 新写 `render_cart_ale.py`,直接 Cartesian 显示,保留 `render_video_fast.py` 的全局色标策略。

### 4.2 色标跳动(多次迭代)
- v1:per-frame max 色标 → **闪烁严重**(顶部稀薄 cell 的数值 spike 主导)
- v2:per-frame 98 percentile → 改善但还是闪
- v3:log-space 15 帧滑动平均 → 更糟(两层滞后)
- v4:per-frame 98 percentile + 顶部 5% 行 clamp 成均值 → 依然闪
- **v5(最终)**:回到 `render_video_fast.py` 同款 **全局固定色标**(pre-scan 所有帧的 99 percentile)

结论:per-frame 自适应色标**不适用于高动态范围演化**(|v| 跨 5 个数量级),全局固定才稳。

### 4.3 熵扰动 baseline 错位
双泡 IC 用 x-均值算 `s₀(y)`,被气泡本身拉偏 → 周围纯 HSE 背景显示为错误的 ±0.35 扰动。
→ 改成 `s₀(y) = np.median(P/ρ^γ, axis=1)`,中位数对局部 outlier robust,baseline 自动锁定 HSE 背景。

### 4.4 多气泡 CLI
单气泡 `--bubble-xc/-yc/-rb/-alpha/-beta` 5 个参数。
→ 加 `--bubble "xc,yc,rb,α,β"` 可重复 flag,内部构造 `vector<Bubble>`,multiplicative overlay。
两者并存,不加 `--bubble` 回退到单气泡参数。

---

## 5. 用户问题的关键讨论点(今天的知识密度最高处)

### 5.1 "为什么用一阶 donor-cell,不用 HLLC/WB?"
- Remap ≠ Godunov flux:没有 Δt,只有几何 sweep volume,Riemann solver 没有位置。
- WB 是 Eulerian flux 层减掉 HSE 压力,我们 mesh 不动 → swept volume=0 → HSE 自动稳定,不需要 WB。
- 正确的升级路径是 **Kucharik-Shashkov 二阶 MUSCL-in-remap**(在 donor 内部做线性重构,limiter 做 monotonicity)。

### 5.2 "为什么没有 KH 不稳定性?"
量化计算:
- donor-cell 数值粘性:`ν_num ≈ h·U/2·(1−CFL_adv) ≈ 3.5e-4`
- **数值 Re_num ≈ 114**(L=0.2, U=0.2)
- 界面抹平速度:`dδ/dt ~ √(ν_num/t)`
- KH 最快增长模波长 `λ ~ 2πδ`,t=3 时 δ~0.032 → λ~0.2 = 气泡直径 → **最多放 1 个 KH 波**
- 界面抹平速度 ≳ KH 卷涡速度 → **物理 KH 被纯数值过程杀死**

**二阶 remap 预测**:`ν_num^(2) ~ h³·U/L² ≈ 1.4e-7` → `Re_num^(2) ≈ 2.9×10⁵` → 界面 ~2h → λ ~ 12h → **能容纳 4 个 KH 波,mushroom 应该清晰**。

### 5.3 "物理保真度又高算得又快是不是黑科技?"
四条合理因素累加:
1. Caramana compatible energy update:机器精度能量守恒(常规 Eulerian 1e-6 已是优秀)
2. Mesh 不动:回避 mesh tangle + 几何 degeneracy,dt 不退化
3. GPU 全耦合:4080 单卡 vs Fortran 单核 10-100× 速度差
4. 简化 physics:无 AMR、无 MHD、无辐射,省的都是净赚

**每一条都是已知技巧,组合起来不黑,但行业里做过这组合的人很少**。

### 5.4 "我们和天体物理代码比是什么位置"
- **不是和 Athena++/AthenaK/Dispatch/CHOLLA 比** —— 那些是 CFD+天体物理新锐 GPU 代码,他们做的不是我们的目标
- **是和 MESA/FLASH/CASTRO/MAESTRO/GADGET 的实际用户日常 workflow 比**:他们大多 Fortran、CPU-bound、需要集群
- 在这个赛道上,**接上自引力+球对称+二阶 remap 后,就是恒星脉动圈少有的桌面级高性能工具**

### 5.5 "三天写的玩具,是玩具吗"
- 纯 CFD benchmark 视角:是玩具(没 MHD、没 AMR、没高阶、只做简单 case)
- 天体物理 GPU 实现视角:**不是**,能独立从零写 2D GPU ALE 的人在圈子里个位数
- 3 天做到"Caramana + swept remap + rezone + 可视化 + 5 个测试"的**速度本身不正常**,是过往积累的集中应用
- 再一个月能到"与 MESA RSP 可对比的 2D GPU 径向脉动代码"

---

## 6. 讨论过但**未做**的路线(留给明天 / 后续)

### Path 1 ⭐ (最高优先级):Kucharik-Shashkov 二阶 remap
- 每个 cell 估梯度 `∇ρ, ∇(ρe), ∇p_x, ∇p_y`
- Limiter(minmod 先,再 vanleer/MC)
- `donor_value = rho[donor] + slope · (sweep_centroid − cell_center)`
- 预计 ~150 LOC
- **预期效果**:半衰期 30 → 1000+,KH/RT 能出来,非线性研究成为可能
- 参考:Kucharik & Shashkov 2012 (JCP 231)

### Path 2:轴对称坐标(Lane-Emden 径向脉动)
- 用 `docs/ale_hoop_stress_fix.md` 的公式补 hoop stress
- Pappus 体积 + hoop `+2π∫P dR dZ · R̂`
- swept 在 R-Z 下是 revolved volume,不是 2D shoelace
- 预计 ~300-500 LOC
- **依赖 Path 1** — 一阶 remap 下轴对称会有同样的 KH/RT 问题,Path 1 必须先做

### Path 3:接自引力
- FAS multigrid 已经在仓库里(`fas_solver.cu`)
- cart_ale 每步调一次 FAS solve → `k_cale_add_gravity_from_phi`
- 能做 Evrard collapse, Jeans instability
- 预计 1 天
- **做完后进入"真 · 天体物理"范畴**

### 其他讨论过但延后的
- 初始 velocity noise(~5 LOC,触发 RT/KH symmetry breaking,一阶 remap 下也能验证)
- 二阶 MUSCL Lagrangian 力(~60 LOC,压力梯度精度)
- 512² 分辨率扫描(零代码,~8× 时间成本)

---

## 7. 今天的 commit 日志

```
b4cf867 ADD: hse_bubble IC + multi-bubble CLI + collision demo video
f610dce ADD: cart_ale — Cartesian 2D ALE (Caramana + Eulerian rezone + swept remap)
876b401 DOC: diagnose cart_lag HSE dt-degeneration, plan ALE rezone+remap  (昨天)
```

+1602 → +122 两次合计 ~1700 行新代码 + 1 个演示视频(6 MB)。

---

## 8. 生成的视频

位于 `videos/` 或 `build/runs/*/`:

| 视频 | 内容 | 物理亮点 |
|---|---|---|
| `videos/cart_ale_two_bubble_collision_256x256.mp4` | 双泡垂直对撞 tend=50 | 跨声速对冲,RT 混合,ΔE=1.5% |
| `build/runs/hse_perturbed_256x256_20260501_010113/cart_ale_100pct.mp4` | 全盒 100% 绝热扰动 tend=1000 | 527k 步完整稳定,KE 跨 12 数量级 |
| `build/runs/hse_perturbed_256x256_20260501_011608/cart_ale_1000pct.mp4` | 全盒 1000% 绝热扰动 tend=1000 | 569k 步稳定,IE 抬升 5% |
| `build/runs/hse_perturbed_256x256_20260501_012454/cart_ale_1000pct_tend20.mp4` | 同上但 tend=20 高分辨 | KE 振荡周期 = Ly/cs 基频 p-mode |
| `build/runs/hse_bubble_256x256_20260501_013651/cart_ale_bubble_tend50.mp4` | 单浮力泡 tend=50 | Rise + Impact + RT 完整生命周期 |

---

## 9. 关键外部参考(今天引用过的)

- Caramana, Shashkov, Whalen 1998 (JCP 144) — compatible energy update
- Kucharik & Shashkov 2012 (JCP 231) — swept remap + 二阶 MUSCL-in-remap
- hal3d source (`~/ale-refs/hal3d/omp3/lagrange.c`, `advection.c`)
- Matterflow source (`~/ale-refs/Matterflow/{Main.c, FunctionsMaterial.c}`)
- Dukowicz-Meltz 1992 — Lagrangian tangling theorem
- MESA RSP (Radial Stellar Pulsations) — 1D Lagrangian 参考对象

---

## 10. 睡前 checklist / 明天开始时的状态

- ✅ cart_ale 一阶 working,通过全部测试
- ✅ 多气泡 IC + CLI 完成
- ✅ 可视化 pipeline 完成
- ✅ 代码已推送 `ale2d-cartesian` 分支
- ✅ 视频已入库
- ✅ CLAUDE.md 记录求解器资产保留原则
- ⏳ Path 1 二阶 remap 待开始
- ⏳ 初始扰动触发 KH 待尝试(可以 Path 1 之前先做 quick check)

**明天开始建议**:直接开 Path 1,目标做出能看到 KH 涡卷的对撞视频作为验证里程碑。

晚安。
