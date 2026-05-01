# Anelastic 伪谱求解器：Sturm-Liouville 本征展开设计

日期：2026-05-01  
状态：探索性设计文档  
前置：`pseudo_spectral_solver`（2D 不可压 NS 伪谱法，已稳定）

---

## 1. 动机

当前 `pseudo_spectral` 求解器解的是均匀密度不可压 NS（ρ=const, ∇·u=0）。
恒星对流需要分层密度 ρ₀(y)，对应的压力 Poisson 方程变成**变系数椭圆方程**，
Fourier 模不再是本征函数，标准伪谱 O(N log N) 求解失效。

本文档推导基于 Liouville 正规化的 Sturm-Liouville 本征展开方法，
将变系数 Poisson 问题还原为"逐点除法"形式，保留伪谱法的结构优势。

---

## 2. 问题设定

### 2.1 变密度不可压流的压力方程

变密度不可压动量方程除以 ρ(x) 取散度，得到：

$$\nabla \cdot \left(\frac{1}{\rho_0(y)} \nabla p\right) = f(\mathbf{x})$$

其中 ρ₀(y) 是仅依赖 y 的背景分层密度（恒星内部的径向密度 profile）。

### 2.2 Fourier-in-x 分离

x 方向仍为周期均匀，做 Fourier 展开 p(x,y) = Σ p̂(kx,y) e^{ikx·x}，
每个 kx 得到 1D 变系数 ODE：

$$\frac{d}{dy}\left[\frac{1}{\rho_0(y)}\frac{d\hat{p}}{dy}\right] - \frac{k_x^2}{\rho_0(y)}\hat{p} = \hat{f}(k_x, y) \tag{2.1}$$

**问题**：kx² 项前面的 1/ρ₀(y) 系数使得不同 SL 模式耦合（非对角化）。
直接展开成 d/dy[(1/ρ₀)d/dy] 的本征函数，kx² 项会引入稠密耦合矩阵。

---

## 3. Liouville 正规化

### 3.1 变量代换

令 p̂(y) = √ρ₀(y) · q(y)。计算：

$$\frac{d}{dy}\left[\frac{1}{\rho_0}\frac{d\hat{p}}{dy}\right] = \rho_0^{-1/2}\left[q'' + W(y)q\right]$$

其中**势函数**：

$$W(y) = \frac{\rho_0''}{2\rho_0} - \frac{3(\rho_0')^2}{4\rho_0^2} \tag{3.1}$$

### 3.2 化简后的方程

代入 (2.1) 并乘以 √ρ₀：

$$q'' + W(y)q - k_x^2 q = g(y) \tag{3.2}$$

其中 g(y) = √ρ₀(y) · f̂(kx, y)。

等价地：

$$\left[\mathcal{T} - k_x^2\right] q = g, \qquad \mathcal{T} \equiv \frac{d^2}{dy^2} + W(y) \tag{3.3}$$

**关键**：变系数 1/ρ₀ 已完全吸收进变量代换。
算子 T 不依赖 kx，kx² 仅作为常数加在本征值上。

---

## 4. Sturm-Liouville 本征展开

### 4.1 本征问题

$$\mathcal{T}\psi_n = -\mu_n \psi_n, \qquad \psi_n''(y) + W(y)\psi_n(y) = -\mu_n\psi_n(y) \tag{4.1}$$

这是 1D **Schrödinger 方程**，势为 W(y)。

由 SL 理论保证：
- 本征值 {μ_n} 实数，μ₀ < μ₁ < μ₂ < ...
- 本征函数 {ψ_n} 在标准 L² 内积下构成完备正交基
- 当 W → 0（ρ₀ → const），ψ_n 退化为 Fourier 模，μ_n → n²π²/L²

### 4.2 对角化求解

展开 q = Σ q_n ψ_n，g = Σ g_n ψ_n：

$$[T - k_x^2]q = g \implies -(μ_n + k_x^2)q_n = g_n$$

$$\boxed{q_n = -\frac{g_n}{\mu_n + k_x^2}} \tag{4.2}$$

**同一组 {ψ_n} 对所有 kx 通用。** kx² 只是平移本征值。
结构完全类似于均匀密度下的 Fourier：p̂(k) = f̂(k) / |k|²。

---

## 5. 完整求解流程

### 5.1 预计算（一次性，仅在 ρ₀ 变化时重算）

```
1. 从 ρ₀(y) 计算 W(y) = ρ₀''/(2ρ₀) − 3ρ₀'²/(4ρ₀²)
2. 解 1D Schrödinger 本征问题 (4.1)，得 {μ_n, ψ_n}
3. 构造变换矩阵 Ψ[i,n] = ψ_n(y_i)              ← Ny × Ny
4. 预计算 √ρ₀(y_i) 向量
```

### 5.2 每步求解

```
1. FFT in x:        f(x,y) → f̂(kx, y)            ← cuFFT R2C, O(Nx·Ny·log Nx)
2. 权重:            g(kx,y) = √ρ₀(y) · f̂(kx,y)   ← 逐点乘, O(Nx·Ny)
3. 正 SL 变换:      G = Ψᵀ · g                     ← cuBLAS GEMM, O(Ny²·Nx)
4. 逐点除:          Q_n(kx) = −G_n(kx)/(μ_n+kx²)  ← O(Nx·Ny)
5. 逆 SL 变换:      q = Ψ · Q                      ← cuBLAS GEMM, O(Ny²·Nx)
6. 权重:            p̂(kx,y) = √ρ₀(y) · q(kx,y)   ← 逐点乘, O(Nx·Ny)
7. IFFT in x:       p̂(kx,y) → p(x,y)              ← cuFFT C2R, O(Nx·Ny·log Nx)
```

### 5.3 复杂度对比

| 方法 | y 方向操作 | 总复杂度 (N×N) | GPU 特性 |
|------|-----------|---------------|----------|
| FFT（ρ=const） | 逐点除法 | O(N² log N) | cuFFT 极快 |
| **SL-GEMM** | **矩阵乘** | **O(N³)** | **cuBLAS GEMM 极快** |
| Chebyshev + 三对角 | Thomas 算法 | O(N²) | 串行，GPU 不友好 |
| 迭代法 (CG/MG) | 多次 SpMV | O(N²·k_iter) | 收敛依赖 |

在 N ~ 2048 尺度上 GPU wall-clock 估算：

```
SL-GEMM:    (2048×2048) × (2048×1024) DGEMM ≈ 4.3×10⁹ FMA → ~3 ms (RTX 4090)
三对角:      1024 × Thomas(2048) 串行 sweep → ~2-5 ms (延迟受限)
FFT (ρ=const): cuFFT C2C 2048×1024 → ~0.5 ms
```

---

## 6. 势函数 W(y) 的物理意义

### 6.1 与 Brunt-Väisälä 频率的关系

W(y) = ρ₀''/(2ρ₀) − 3ρ₀'²/(4ρ₀²) 度量密度分层的"曲率"。

对理想气体 polytrope（P = Kρ^γ，HSE 下 ρ₀'/ρ₀ = −g/(γ K ρ₀^{γ−1})）：
- ρ₀ 平缓 → W ≈ 0 → ψ_n ≈ Fourier 模
- ρ₀ 缓变 → W 小 → ψ_n ≈ 微扰 Fourier 模（可用微扰论加速）
- ρ₀ 剧变（核-包层过渡） → W 有深势阱 → ψ_n 集中在梯度大的区域

### 6.2 与恒星 g-mode 的关系

SL 本征问题 (4.1) 的数学结构与恒星内部 g-mode（重力波）本征频率问题完全类似。
g-mode 频率由 Brunt-Väisälä 频率 N²(r) 决定，而 N²(r) 和 W(y) 具有相同的依赖关系。

**副产品**：SL 预计算在给出 Poisson 求解器的同时，也给出了 g-mode 本征频率分析。

### 6.3 潜在奇异性

当 ρ₀ → 0（恒星表面/大气），W(y) 可能发散。需要验证：
- Lane-Emden n=1.5 polytrope：ρ₀ = 0 在 r = R★，W → ?
- 可能需要表面 cutoff 或 sponge layer 截断
- **TODO**：拿真实 polytrope/MESA profile 数值计算 W(y)，检查行为

---

## 7. 文献调研：已有方法的空白

### 7.1 各社区的做法

| 社区 | 代表代码 | 径向求解方法 | 为什么不用 SL |
|------|---------|-------------|-------------|
| 恒星伪谱 | ASH, Rayleigh | 球谐(角向) + Chebyshev/FD(径向) + 带状矩阵 | CPU 上 GEMM 慢，Chebyshev 已够好 |
| GPU 伪谱 | hit3d, spectralDNS | Fourier 全方向 | 只做均匀密度 |
| 变密度 GPU CFD | 工程 LES 代码 | 有限体积 + multigrid | 不碰谱方法 |

### 7.2 文献空白

**没有人把以下三者结合过：**
1. Liouville 正规化 + SL 本征展开（经典数学物理）
2. GPU batched GEMM 做 SL 正/逆变换（GPU 算力优势）
3. Anelastic/Boussinesq 恒星对流伪谱求解器（天体物理应用）

原因：三个社区各自不会走到这条路上来——
- 做恒星伪谱的人在 CPU 上跑，GEMM 不划算
- 做 GPU 伪谱的人只做均匀密度
- 做变密度 GPU 的人用 FV + multigrid

---

## 8. 需要验证的关键问题

### 8.1 W(y) 在真实恒星 profile 下的行为

- [ ] 计算 Lane-Emden n=1.5 polytrope 的 W(y)
- [ ] 检查 ρ₀ → 0 处的奇异性
- [ ] 如果发散，确定截断/正则化策略

### 8.2 SL 本征函数的收敛性

- [ ] 给定 polytropic W(y)，需要多少本征模 N_mode 才能达到 10⁻¹⁰ 精度？
- [ ] 如果 W(y) 光滑且有界，期望指数收敛
- [ ] 如果 W(y) 有奇异性，可能退化为代数收敛

### 8.3 GPU wall-clock benchmark

- [ ] 实测 cuBLAS DGEMM (2048×2048)×(2048×1024) 的实际延迟
- [ ] 对比 cuSPARSE batched 三对角求解 (gtsv2StridedBatch)
- [ ] 对比 iterative CG with FFT preconditioner

### 8.4 数值稳定性

- [ ] μ_n + kx² 接近零时的共振处理（低 kx + 低 μ_n）
- [ ] 大 n 时 μ_n ~ n²π²/L²，高频模式自然被压制，应该稳定
- [ ] √ρ₀ 代换在 ρ₀ → 0 处的数值行为

---

## 9. 实现路线图

### Phase 0: 可行性验证（1 周）
1. 用 Python/numpy 算 polytropic W(y) 和前 N 个 SL 本征对
2. 测试 SL 展开对已知解的逼近精度
3. 估算 GPU GEMM 实际延迟

### Phase 1: Boussinesq 伪谱 + Fourier-Chebyshev（2-3 周）
不用 SL，先用标准 Chebyshev + 三对角做 y 方向，验证物理正确性：
- Rayleigh-Bénard 对流，Nu-Ra 标度律对标
- 保留为 baseline

### Phase 2: SL-GEMM 替换 y 方向求解（1-2 周）
- 预计算 SL 本征对（GPU Schrödinger solver 或 host 预算传入）
- 用 cuBLAS batched GEMM 替换 Chebyshev 三对角
- 精度/速度对比

### Phase 3: Anelastic 扩展（2-3 周）
- 从 Boussinesq 升级到 anelastic: ∇·(ρ₀ u) = 0
- SL 求解器天然适配（同一套 W(y) 和 {ψ_n}）

### Phase 4: 论文撰写
目标：JCP 或 Astronomy & Computing
标题草案："GPU-accelerated anelastic pseudo-spectral solver using
         Sturm-Liouville eigenfunction expansion for stratified convection"

---

## 10. Anelastic 方程组（Phase 3 参考）

### 10.1 控制方程

$$\nabla \cdot (\rho_0 \mathbf{u}) = 0 \tag{10.1}$$
$$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u} = -\frac{1}{\rho_0}\nabla p' + \frac{\theta'}{\theta_0}\mathbf{g} + \nu\nabla^2\mathbf{u} \tag{10.2}$$
$$\frac{\partial \theta'}{\partial t} + \mathbf{u}\cdot\nabla\theta' + u_y \frac{d\theta_0}{dy} = \kappa\nabla^2\theta' \tag{10.3}$$

其中 ρ₀(y), θ₀(y) 为背景分层，p', θ' 为扰动。

### 10.2 压力 Poisson 方程

对 (10.2)/ρ₀ 取散度，利用 (10.1)：

$$\nabla \cdot \left(\frac{1}{\rho_0}\nabla p'\right) = \text{RHS}(\mathbf{u}, \theta') \tag{10.4}$$

这正是 §2.1 的形式。SL 方法直接适用。

### 10.3 涡度-流函数形式（2D 特化）

定义流函数 ψ 使得 ρ₀ u = ∂ψ/∂y, ρ₀ v = −∂ψ/∂x，自动满足 (10.1)。
涡度方程 + 温度方程构成封闭系统。SL 展开同时用于 Poisson 反演和扩散项的隐式积分。

---

## 附录 A: Liouville 正规化推导细节

### 代换 p̂ = √ρ₀ · q

$$\frac{d\hat{p}}{dy} = \frac{\rho_0'}{2\sqrt{\rho_0}}q + \sqrt{\rho_0}\,q'$$

$$\frac{1}{\rho_0}\frac{d\hat{p}}{dy} = \frac{\rho_0'}{2\rho_0^{3/2}}q + \frac{q'}{\sqrt{\rho_0}}$$

$$\frac{d}{dy}\left[\frac{1}{\rho_0}\frac{d\hat{p}}{dy}\right] = \frac{q''}{\sqrt{\rho_0}} + \left[\frac{\rho_0''}{2\rho_0^{3/2}} - \frac{3(\rho_0')^2}{4\rho_0^{5/2}}\right]q$$

$$= \frac{1}{\sqrt{\rho_0}}\left[q'' + W(y)q\right]$$

$$W(y) = \frac{\rho_0''}{2\rho_0} - \frac{3(\rho_0')^2}{4\rho_0^2} \tag{A.1}$$

### 完整方程

$$\frac{1}{\sqrt{\rho_0}}\left[q'' + Wq\right] - \frac{k_x^2}{\rho_0}\sqrt{\rho_0}\,q = \hat{f}$$

$$\frac{1}{\sqrt{\rho_0}}\left[q'' + Wq - k_x^2 q\right] = \hat{f}$$

$$q'' + W(y)q - k_x^2 q = \sqrt{\rho_0}\,\hat{f} \equiv g(y) \tag{A.2} \qquad \square$$

---

## 附录 B: 对角化验证

设 T ψ_n = −μ_n ψ_n，展开 q = Σ aₙ ψ_n：

$$(T - k_x^2)\sum_n a_n \psi_n = \sum_n a_n(-\mu_n - k_x^2)\psi_n = g$$

内积 ⟨·, ψ_m⟩ （标准 L²，{ψ_n} 正交归一）：

$$-(\mu_m + k_x^2)a_m = g_m = \langle g, \psi_m \rangle$$

$$a_m = -\frac{g_m}{\mu_m + k_x^2} \tag{B.1}$$

**不存在模式耦合。** 变系数 1/ρ₀ 的效应完全编码在 {μ_n, ψ_n} 中。 □
