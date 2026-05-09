# cart_ale2 设计文档

**定位**: cart_ale 的下一代,可压缩 2D ALE 求解器,为**恒星对流 + PdV 脉动**设计。**不是**通用湍流 benchmark 工具。

**对应源码**:
- `src/gpu/cart_ale2_solver.{cu,cuh}` — struct、生命周期、step 调度、IC、frame buffer、VTK
- `src/gpu/cart_ale2_kernels.cu` — 所有 device kernel
- `src/main.cpp` — `--solver cart_ale2` 分派

**对应数学参考**: [equations.md §15 §16 §17](equations.md)

---

## 1. 适用域(诚实清单)

### ✅ 适合
- **恒星对流**(Lane-Emden backbone + 低 Mach 扰动,需要长时间 IE 机器精度守恒)
- **MESA → local-box convection**(`--test local_convection`,radial1d ZAMS 薄片 +
  Newton cooling,详见
  [cart_ale2_local_convection_2026-05-03.md](cart_ale2_local_convection_2026-05-03.md))
- **Compressible pulsation**(Cepheid / RR Lyrae 类,精确 PdV work)
- **恒星质量抛射初期**(HSE + 出流,长时间质量守恒)
- **Core collapse / bounce**(强 PdV,中等激波 capture)
- **HSE benchmark + 小扰动**(机器精度 well-balanced,不需要 reference-state subtraction 特殊处理)

### ⚠️ 可以但不最佳
- 行星撞击、超新星 blast wave(能做,Godunov + high-order 更锐利)

### ❌ 不要用
- **2D/3D 全发展湍流** — Caramana subcell force 的 effective viscosity 远高于 HLLC face flux,有效 Reynolds 数只能做到 ~10³ 级别(偽谱 Re=2×10⁵)。KH benchmark 动能谱 falls as $k^{-10}$ past $k_\text{inj}$,**不是** Kraichnan $k^{-3}$。
- **Incompressible** — 该是偽谱(`.claude/worktrees/pseudo-spectral`)的赛道。
- **辐射 / 化学** — 没接,当前版本做不到 operator split。
- **MHD** — 无 B 场。

---

## 2. 架构(相对 cart_ale 的增量)

| 模块 | cart_ale | cart_ale2 |
|---|---|---|
| BC | reflective wall only | **reflective / x-periodic / y-periodic / 双周期**(`--bc-x`, `--bc-y`) |
| Remap order | 1 = donor-cell, 2 = MUSCL | 同 + **PPM**(CW 或 CS limiter,cons 或 prim,char 投影) |
| I/O | 每帧 host-side write_vtk | **VRAM frame buffer** + `flush_frames_to_disk` + `frames.csv`(真实 t 标签) |
| IC | hse / hse_bubble / sod / kh_shear | 同 + **`kh_lecoanet`**(Athena iprob=4 canonical) |
| AV | 标量 Q | 同 + **shear-aware AV**(`--shear-aware-av`) |

pipeline 没变:**Lagrangian(§15)→ Rezone → Remap(§16)**。rezone 永远把 node snap 回 `(X0, Y0)`,所以这是 Eulerian rezone。

---

## 3. PPM 变体矩阵

`--ppm` 打开后,可以组合 3 个正交参数:

| 参数 | 值 | 含义 |
|---|---|---|
| `--ppm-limiter` | `cs`(默认) | Colella-Sekora extremum-preserving(§17.3–17.5) |
| | `cw` | Colella-Woodward 1984 classical clamp(§17.2) |
| `--ppm-space` | `prim`(默认) | 重建 $\{\rho, P, v_x, v_y\}$(§17.6) |
| | `cons` | 重建 $\{\rho, \rho e, \rho v_x, \rho v_y\}$(Kucharik-Shashkov 版) |
| `--ppm-char` / `--no-ppm-char` | on(默认) | 特征变量投影(§17.7–17.8),需要 prim space |
| | off | 直接在 primitive 空间重建 |

### 已验证的稳定性

| 测试 | CS/prim/char | CS/prim/no-char | CS/cons | CW/cons |
|---|---|---|---|---|
| HSE 64² t=0.5 | ✅ E 守恒 10 位 | ✅ | ✅ | ✅ |
| kh_shear 256² t=3 | ✅ | ✅ | ✅ | ✅(基线,E drift 1.6%) |
| kh_lecoanet 256×512 t=5 | ✅ E drift 1.2% | ⚠️ t≈0.043 NaN | ❌ t≈0.044 NaN | ❌ t≈0.044 NaN |

**关键发现**:在 Lecoanet 的 smooth tanh 剪切层,**特征投影是稳定性必要条件**(不是可选优化)。原因:在剪切界面上 acoustic 和 shear mode 耦合,守恒/原始变量 PPM 过零点会产生压力 overshoot → 负压 → NaN。投影到 $(w_0, w_1, w_2, w_3)$ 之后 4 个 mode 独立限幅,shear gradient 不会污染压力。

### 为什么 CS limiter 的收益在 kh_shear 上看不出来?

`kh_shear` 的 tanh 过渡带只有 2 个格点宽(`delta = 2·dy`),CS 的"smooth extremum 保留"机制几乎没发挥空间 — 过渡带本身被 MUSCL/PPM 都当成 shock 对待。`kh_lecoanet` 的过渡带 0.05·Ly(50+ 格点)才是 CS 的战场,但那里 char 投影也必须 on。

---

## 4. 周期 BC 实作

两个**反直觉的陷阱**(详见 [pitfalls.md P30, P31](pitfalls.md)):

### (a) Remap kernel 必须处理 wrap edge

cell ic=nx-1 的 east face 在 reflective 下是墙,在 x-periodic 下是 wrap face(通向 cell 0)。kernel 的 `n_edges` 必须根据 `bc_mode` 切换:

```cpp
bool x_per = (bc_mode & 1) != 0;
int n_edges = x_per ? nx * ny : (nx - 1) * ny;
// ...
int cR_idx = ic + 1;
if (x_per && cR_idx >= nx) cR_idx = 0;
```

同时 donor-relative 坐标 `sx = (cx - xd) / dx_u` 在 wrap 时需要展开:

```cpp
if (x_per) {
    if (sx >  0.5) sx -= nx;   // unwrap to −0.5
    if (sx < -0.5) sx += nx;
}
```

**未来加新 remap kernel 必须遵守这个约定**。

### (b) Periodic node sync 的 master / mode 语义

```cpp
bool x_dup = x_per && (in == nnx - 1);
bool y_dup = y_per && (jn == nny - 1);
if (x_dup || y_dup) return;  // 只 master 继续
```

master 选择不是 `in==0 AND jn==0`(那样只有 (0,0) 通过,x-only 和 y-only 副本从不 sync),而是"**不是副本**"。双周期下 master 是:角 (0,0) + 左边 `(0, jn∈[1, nny-2])` + 底边 `(in∈[1, nnx-2], 0)`,各自 sync 自己的 partner。

然后根据字段类型选 `mode`:

| 字段 | sync mode | 理由 |
|---|---|---|
| `FX, FY`(force) | **sum**(mode=1) | `node_forces` 是 cell-parallel atomicAdd,边界 node 只收到自己那半 domain 的 cell 贡献;两个 partner 之和 = full force |
| `vX, vY`(velocity) | **copy/avg**(mode=0) | 副本物理上同一点,值必须相等;avg 顺便吸收 FP 噪声 |
| `dX, dY`(位移) | **copy/avg** | 同 velocity |
| `m_node`(质量) | **不 sync** | `node_mass` kernel 是 node-parallel,每个 node 自己 wrap 4 cells,两个副本已经各自算出 full mass |

**诊断汇总**(`Σ m_n v_n²` 等)必须跳过 `in=nnx-1` 和 `jn=nny-1` 的副本,否则双重计算导致 KE 虚高。

---

## 5. KH benchmark 结果(诚实数据)

### Uniform advection(最 trivial 测试)

`FORCE_UNIFORM_VX=1` 环境变量让 `init_kh_lecoanet` 改成 uniform vx = vflow。预期:KE 和 E 应该每步严格不变,|v|_max 不变。

| 分辨率 | 步数 | `\|KE(dt) - KE(0)\|` | 备注 |
|---|---|---|---|
| 64² | 1 | $< 10^{-14}$ | ✅ bug 修完后机器精度守恒 |
| 修 bug 前 | 1 | $\sim 3 \times 10^{-4}$ | P30 remap 或 P31 sync 任一未修都会暴露 |

### Lecoanet canonical(Athena iprob=4 几何, k=1, vflow=1, P0=10, amp=0.01)

256×512, tend=5, PPM/CS/prim/char, CFL=0.3:

| $t$ | KE | IE | $E$ drift | $|v|_\text{max}$ |
|---|---|---|---|---|
| 0 | 0.8000 | 50.000 | 0 | 1.000 |
| 1 | 0.5564 | 50.000 | 0.48% | 0.977 |
| 2 | 0.4097 | 50.000 | 0.77% | 0.885 |
| 3 | 0.3030 | 50.000 | 0.98% | 0.773 |
| 5 | 0.1726 | 50.000 | 1.23% | 0.587 |

IE 守恒到 10 位,总 E drift 1.2%。跑速 6500 steps/s on 256×512 (RTX 4070 Ti-class)。

### High-k spectrum test(k=7, 512×1024, tend=8)

动能谱从 $k_\text{inj} = 7$ 开始下降,slope $\sim -10$(不是 Kraichnan $-3$),高 k 有 grid-scale aliasing pile-up。证实 ALE 不是 2D turbulence 工具。

**对比**(同 IC geometry,偽谱解不可压 NS,Re=2×10⁵):
- 偽谱 1024² KH: slope $-3.45 \pm 0.31$(Kraichnan k⁻³,1.4σ 通过),~2 decade 惯性区
- cart_ale2 512×1024 KH: slope ~$-10$,< 1 decade 惯性区

结论已反馈到 [CLAUDE.md](../CLAUDE.md):**别再花时间用 ALE 追 KH 视觉细节**。

---

## 6. CLI 快查

```bash
# HSE 回归(机器精度守恒)
./stellar2d run --solver cart_ale2 --test hse --nr 64 --ntheta 64 \
            --remap-order 2 --ppm --tend 0.5

# Lecoanet canonical KH(canonical benchmark)
./stellar2d run --solver cart_ale2 --test kh_lecoanet \
            --nr 256 --ntheta 512 --cfl 0.3 \
            --bc-x periodic --bc-y periodic \
            --remap-order 2 --ppm --ppm-space prim --ppm-char \
            --tend 5 --vtk-dt 0.025 --frame-buffer

# 对比 PPM/CS vs PPM/CW(看 limiter 影响)
./stellar2d ... --ppm --ppm-limiter cs   # (默认)
./stellar2d ... --ppm --ppm-limiter cw

# 对比 conservative vs primitive 空间
./stellar2d ... --ppm --ppm-space prim   # (默认,稳)
./stellar2d ... --ppm --ppm-space cons   # 会在 Lecoanet 塌

# 关特征投影(只在测试 char 投影贡献时用)
./stellar2d ... --ppm --ppm-space prim --no-ppm-char

# 高 k KH(非 canonical)
./stellar2d ... --test kh_lecoanet --kh-k 7 --perturb 0.05

# 分析
python scripts/render_kh.py runs/<dir>      # 渦度 + speed 视频
python scripts/spectrum_kh.py runs/<dir>    # 动能谱
```

---

## 7. 未来工作

### 不值得做的(已验证是死胡同)
- 继续提高分辨率追 KH 细节 — bottleneck 是方法论不是分辨率
- 在 Caramana framework 里硬塞 low-dissipation 修正 — 会破坏 compatible energy 的 bit 守恒

### 可能值得做的
- **HLLC face flux 版本**(新求解器,开新文件,**不覆盖** cart_ale2)— 会牺牲 compatible energy,换到 Godunov 体系,有机会追上 Athena-class KH
- **恒星对流 workflow**(cart_ale2 的原生目标):Lane-Emden IC + 熵扰动注入 + mixing layer diagnostics,`hse_bubble` 已经是一半
- **AMR**(long shot)— Caramana + swept remap 的 AMR 理论上可行但工作量大

### 保留基线
按 CLAUDE.md 的资产保留规则,**cart_ale2 是只读资产**。要做新物理开新文件(`cart_ale3` 或别的名字),不要修这里。
