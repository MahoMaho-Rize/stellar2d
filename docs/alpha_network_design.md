# α-network(13 核)设计方案

**目的**: 为 Stage 2 复刻 Sato+2024 Fig 7 提供 alpha-chain 核反应。
**状态**: 设计阶段
**位置**: `src/physics/alpha_network.{h,cuh}`
**使用者**: radial1d(初期),cart_ale2(可选扩展)

---

## 1. 核素列表(13 核)

Timmes aprox13 的标准配方:

| idx | nucleus | A | Z | N |
|---|---|---|---|---|
| 0  | ⁴He   | 4  | 2  | 2 |
| 1  | ¹²C   | 12 | 6  | 6 |
| 2  | ¹⁶O   | 16 | 8  | 8 |
| 3  | ²⁰Ne  | 20 | 10 | 10 |
| 4  | ²⁴Mg  | 24 | 12 | 12 |
| 5  | ²⁸Si  | 28 | 14 | 14 |
| 6  | ³²S   | 32 | 16 | 16 |
| 7  | ³⁶Ar  | 36 | 18 | 18 |
| 8  | ⁴⁰Ca  | 40 | 20 | 20 |
| 9  | ⁴⁴Ti  | 44 | 22 | 22 |
| 10 | ⁴⁸Cr  | 48 | 24 | 24 |
| 11 | ⁵²Fe  | 52 | 26 | 26 |
| 12 | ⁵⁶Ni  | 56 | 28 | 28 |

**注**:aprox13 不含质子、中子,适合爆炸性核合成高温区(T₉>1),不适合早期 H-burning。符合 Sato+2024 需求(只关心 Si-burning 及以上)。

---

## 2. 反应清单(~28 反应)

### Alpha-captures (forward only — backward via detailed balance)
```
¹²C  + α → ¹⁶O          ¹⁶O  + α → ²⁰Ne
²⁰Ne + α → ²⁴Mg         ²⁴Mg + α → ²⁸Si
²⁸Si + α → ³²S          ³²S  + α → ³⁶Ar
³⁶Ar + α → ⁴⁰Ca         ⁴⁰Ca + α → ⁴⁴Ti
⁴⁴Ti + α → ⁴⁸Cr         ⁴⁸Cr + α → ⁵²Fe
⁵²Fe + α → ⁵⁶Ni
```

### Triple-alpha
```
3 ⁴He → ¹²C
```

### Heavy-ion (重要:Ne-burning 和 O-burning 的真正驱动机制)
```
¹²C  + ¹²C → ²⁰Ne + α     (主要)
¹²C  + ¹⁶O → ²⁴Mg + α
¹⁶O  + ¹⁶O → ²⁸Si + α
```

### Ne photodisintegration(Ne-burning 入口)
```
²⁰Ne(γ,α)¹⁶O   (逆向 α-capture 的光解)
```

**注**:所有 γ-disintegration 都通过 detailed balance 从 forward reaction 计算,不需要单独 fit。

---

## 3. 反应率来源

使用 **Caughlan & Fowler 1988 (CF88)** 解析 fit,公开代码有现成的 Timmes aprox13 F77 实现:
- https://cococubed.com/code_pages/burn_helium.shtml
- 反应率通常写成:
  ```
  λ(ρ, T) = const · ρ^n · T₉^α · exp(-β/T₉^{1/3})  (strong resonance)
  λ(ρ, T) = const · ρ^n · f(T₉)                       (photo-disintegration)
  ```

我们做的是 **porting + CUDA 化**,不重新推导物理。

---

## 4. ODE 结构

mass fraction Y_i = X_i / A_i(molar abundance,常用于核反应):

```
dY_i/dt = Σ_j (produced by j) λ_j Y_reactants
        - Σ_k (destroying i) λ_k Y_i Y_partner
```

**Stiff 程度**: 高。T₉ ~ 5 时 α-capture 时标 τ_nuc ~ 10⁻³ s,但整体 explosion 要跑 ~10 s → 10⁴ steps。必须隐式。

---

## 5. 实现策略

### Phase A (简化版,今晚目标)
- 只实现 **3α + ¹²C(α,γ)¹⁶O + ¹⁶O(α,γ)²⁰Ne + ²⁰Ne(α,γ)²⁴Mg + ²⁴Mg(α,γ)²⁸Si**(6 个反应)
- 覆盖 O/Ne/Mg/Si 演化(N49B 主要关心的核素)
- 不包括 Fe 族上游
- 结构:explicit ODE,和 hydro 通过 operator splitting 耦合(hydro sub-step dt → 核反应 sub-step 用 ODE solver 内部细分)
- ODE solver:**backward Euler with Newton iteration**(稳 + 简单),tolerance 1e-6

### Phase B (完整 aprox13,后续迭代)
- 加 Si-burning → Fe 族所有反应
- 用 radial1d 的 Dual<N> AD Jacobian
- 集成到 radial1d Newton 残差

### 今晚 scope:Phase A

---

## 6. 接口设计

```cpp
// src/physics/alpha_network.h

namespace alpha_net {
constexpr int N_SPEC = 13;  // or 6 for Phase A

struct AlphaNetParams {
    double rho;     // g/cc
    double T;       // K
    double dt;      // s (sub-cycle step)
    // ...
};

// Evolve Y[] from t to t+dt given (rho, T), using backward Euler + Newton.
// Returns energy release ε [erg/g] over this dt.
NUC_HD double advance(double Y[N_SPEC], const AlphaNetParams& p);

// Initialize Y[] from Sukhbold mass fractions X[].
NUC_HD void from_X(const double X[N_SPEC], double Y[N_SPEC]);

// Convert back
NUC_HD void to_X(const double Y[N_SPEC], double X[N_SPEC]);
}
```

---

## 7. 验证

### 单元测试
- **α-freezeout 平衡**:T₉=5, ρ=10⁷, 初始全 ⁴He,跑到 t=1 s,检查 NSE 附近分布(Ni/Fe 占优)
- **¹²C(α,γ)¹⁶O 单反应**:T₉=0.2, X(⁴He)=X(¹²C)=0.5,验证 energy release 速率和 CF88 数值

### 集成测试
- 15.90 M☉ progenitor IC → 加一个 light-bulb 后 → 跑到 shock 出表面,检查 post-SN Mg/Ne ≈ 0.75(论文数字)

---

## 8. 工作量分解

| 任务 | 时间 |
|---|---|
| 6-反应率公式 + CF88 ported | 1-2 小时 |
| Backward Euler Newton ODE solver | 1 小时 |
| from_X / to_X helpers | 30 分钟 |
| 单元测试 (α-freezeout + single reaction) | 1 小时 |
| CUDA 化 + 单独 bin 测试 | 1 小时 |
| **Phase A 总计** | **5-6 小时** |

---

## 9. 文件结构

```
src/physics/
├── alpha_network.h       # 接口 + CF88 反应率(header-only 模板)
├── alpha_network.cu      # 如需 CUDA device-side instantiation
tests/
└── test_alpha_network.cu # α-freezeout + single-reaction 单测
```
