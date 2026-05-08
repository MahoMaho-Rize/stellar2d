# MHD Derivations — sympy-driven symbolic manuscript

## 目的

为 stellar2d 的 MHD solver 开发提供 **逐步 symbolic 推导**。每个结论都由 **sympy 自动验证** 产出,作者不手写代数,只写 **假设 + 平衡方程 + 调用 sympy**。

**刻意不用大模型能力做代数**:之前的 stellar_liouville 手稿里我们吃过"模型手算符号跳步犯错"的苦。这里的协议是 —— **任何代数等式都必须由 sympy 对应脚本 print 出来,然后拷到 markdown;不允许手写代数**。

## 项目结构

```
docs/mhd_derivations/
├── README.md                       本文件
├── build_manuscript.sh             汇编所有 section → manuscript.md → PDF
├── run_all.sh                      跑所有 sympy 脚本,刷新 output/
├── scripts/                        sympy 推导脚本(每节 1 个)
│   ├── _common.py                  公用符号定义 + LaTeX 打印 helper
│   ├── a1_ideal_mhd_equations.py
│   ├── a2_conservative_primitive.py
│   ├── a3_flux_jacobian_eigensystem.py
│   ├── a4_hlld_intermediate_states.py
│   ├── a5_ct_divergence_preservation.py
│   ├── b1_flux_tube_geometry.py
│   ├── b2_wave_action_wkb.py
│   ├── b3_parker_critical_point.py
│   ├── c1_ohmic_dissipation.py
│   ├── c2_ambipolar_dissipation.py
│   ├── c3_resistive_energy.py
│   ├── c4_saha_ionization_closure.py
│   ├── d1_cylindrical_mhd.py
│   ├── d2_shearing_sheet_bc.py
│   └── d3_mri_stress_decomposition.py
├── sections/                       markdown 段落,每节对应一个 script
│   ├── 00_preamble.md
│   ├── a0_part_a_overview.md
│   ├── a1_ideal_mhd_equations.md
│   └── ...
└── output/                         sympy 脚本产生的中间 LaTeX snippets
    └── <section>.latex.tex         每个脚本 dump 的 LaTeX(可 \input 到 md)
```

## 三项强规则(不允许违反)

1. **所有代数恒等式必须由 sympy 脚本生成**。脚本 print `sp.latex(expr)`,再手动或 `run_all.sh` 把输出放进 md。**不允许**在 markdown 里手写"通过代数可以得到...="。
2. **每个脚本必须 self-check**:脚本末尾断言 `assert sp.simplify(LHS - RHS) == 0`,如果物理关系成立但 sympy 没证出来,必须 document 原因(如 "sympy 无法 simplify 非多项式三角恒等式,手工用 Wolfram 验证")。
3. **script 只做一件事**:一个 `.py` 推一个 proposition。不要 script 间互相 import 中间结果 —— 全部从 `_common.py` 的定义符号开始重新推。这样任一节都可以独立重跑。

## Scope —— 四大部分

### Part A: Ideal MHD + CT numerical scheme(5 sections)
A1. 理想 MHD 方程组(守恒形式)
A2. 守恒 ↔ primitive 变量变换
A3. flux Jacobian 的 eigensystem(7 波 + entropy)
A4. HLLD intermediate states(Miyoshi-Kusano 2005)
A5. CT scheme 的离散 ∇·B = 0 保持性

### Part B: Super-radial flux-tube 1D reduction(3 sections)
B1. 从 3D 球坐标到 1D flux-tube 的方程投影
B2. Alfvén 波在变 A(r) 上的 WKB 波动作守恒
B3. Parker 风临界点条件(flux-tube 版本)

### Part C: Non-ideal MHD dissipation(4 sections)
C1. Ohmic 耗散项 η_O∇²B
C2. Ambipolar 耗散张量项(关键 MHD gotcha)
C3. 能量方程里的 Joule + ambipolar heating
C4. Saha 电离闭合 → η_O(x_e, T)、η_A(x_e, ρ) 关系

### Part D: Cylindrical shearing-box + MRI(3 sections)
D1. 柱坐标 ideal MHD 方程
D2. Shearing-periodic BC 下 shearing-sheet 近似推导
D3. α_SS stress 的 exact Maxwell + Reynolds decomposition

总计 **15 个 sections**,每个 sympy 脚本预期 50-150 行,对应 md 段落 200-500 字 + 方程显示。

## 执行工作流

```bash
# 开发一个 section:
cd docs/mhd_derivations
vim scripts/a1_ideal_mhd_equations.py
python3 scripts/a1_ideal_mhd_equations.py > output/a1_ideal_mhd_equations.log
# 脚本末尾应打印所有关键方程的 LaTeX,检查后拷到 sections/a1_ideal_mhd_equations.md

# 刷新全部:
bash run_all.sh

# 汇编 manuscript:
bash build_manuscript.sh    # 产出 manuscript.md 和 manuscript.pdf
```

## 依赖

- sympy ≥ 1.14(已加到 `pixi.toml`)
- pandoc(已有)
- xelatex(可选,若要 PDF 输出)

## Derivation ↔ code 的链接

每个 section 里给出 "✅ 验证 checkpoint":指向 `src/gpu/` 里未来的实现文件或 tests/ 里的 benchmark,说明这个 proposition 在代码里对应哪一段。例如:
- A3 eigensystem → `src/gpu/explicit/athena_mhd_kernels.cu::d_mhd_eigenvalues`
- A4 HLLD → `athena_mhd_kernels.cu::d_hlld_flux`
- A5 CT → `athena_mhd_solver.cu::update_face_B_with_emf`

这样 **手稿是 solver 实现的 single source of truth**,而不是 decorative document。
