# 球殼偽譜法天體物理路線圖 (sph_spectral roadmap)

**狀態**: 遠期規劃存檔 (Phase 4+). 目前 (2026-05-03) 主線是 2D Fourier-Chebyshev Boussinesq/Anelastic (Phase 1-3), 球殼版本是 2D 工作成熟後再開新 solver file 的目標.

**日期**: 2026-05-01 (原稿); 2026-05-03 狀態標註
**Scope**: Pseudo-spectral on spherical shell, incompressible / anelastic regime.
**戰略**: 一個一個打怪,低級小怪秒殺,把時間留給高 tier 的 boss。

---

> **狀態說明 (2026-05-03)**
>
> 本檔原擬在 pseudo_spectral 階段後直接跳球殼, 但 Phase 0 ext+ 後發現
> 2D Fourier-Chebyshev 在 Eddington $n=3$ 背景上是更自然的漸進路徑
> (Rayleigh-Bénard baseline 已是成熟 benchmark, 不需要 3D 球殼).
>
> 球殼擴展保留為 Phase 4+ 目標. 當前工作見:
> - `docs/spectral_solver_design.md`
> - `docs/spectral_stratified_poisson_report_2026-05-03.md` §8 (Phase 1-3)

## 核心原則

1. **不碰全可壓縮 + shock**。球殼 + 譜 + shock 是研究級難題,偽譜不是對的工具。
2. **遵守 CLAUDE.md 資產保留**。每開一個新 solver 就是新 `.cu/.cuh/.h` 檔案,不動既有 pseudo_spectral / strang / cart_ale* / lowmach 等。
3. **Boussinesq → Anelastic → MHD** 漸進擴展,每階段留下獨立 solver 作為基線。
4. **每個 case 要能對標文獻 / benchmark**,不做無人可驗的黑箱結果。

---

## 工具鏈現狀(2026-05-01 commit `de7766c` 後)

`pseudo_spectral`(2D Fourier 雙週期)已具備:
- IFRK3 積分因子(任意線性耗散算子解析積分)
- skew-symmetric + conservative + advective 三種對流形式
- linear drag / hyperviscosity / PI controller
- cuRAND forcing / checkpoint / Taylor-Green analytic verification
- eps_KE 物理-譜交叉檢查

球殼版本需要新加的基礎設施:
- **Legendre transform (LT)**:水平方向 Y_l^m 展開,FFT in φ + LT in θ
  - naive O(N²),生產級需 SHTns 或 Mortier transform ~ O(N² log N)
- **Chebyshev transform**:徑向 T_n(r) 展開,FFT-based O(N log N)
- **球諧算子對角化**:∇²_h Y_l^m = -l(l+1)/r² Y_l^m — IFRK3 天然對角積分
- **極軸正則性**:球諧天然處理,但數據轉置 layout 是性能命脈
- **邊界條件 (Chebyshev tau / Galerkin)**:no-slip / stress-free at r_in, r_out

---

## 打怪清單(按 tier 分)

### Tier 1:Boussinesq 球殼(最乾淨入門)

方程:
```
∇·u = 0
∂_t u + u·∇u = -∇p + ν∇²u + α·g·T'·r̂ + (2Ω×u if rotating)
∂_t T' + u·∇T' = κ∇²T' - u_r·dT₀/dr
```

| # | Case | 難度 | 驗證對象 |
|---|---|---|---|
| 1.1 | **Rayleigh-Bénard 球殼對流**(非旋轉,非磁) | ★ | 臨界 Ra_c 理論值 |
| 1.2 | **Rotating Rayleigh-Bénard 球殼** | ★★ | Taylor-Proudman 柱出現 |
| 1.3 | **Taylor-Couette in sphere**(兩層旋轉剪切) | ★★ | Stewartson 層結構 |
| 1.4 | **Tidal forcing**(外加週期性形變) | ★★ | 木衛一潮汐加熱 |
| 1.5 | **Christensen 2001 geodynamo benchmark** | ★★ | 標準答案 4 位有效數字 |

**Tier 1 戰略**:1.1 + 1.2 打完框架就穩。1.5 是 boss,但只比 1.2 多一個磁場方程。

### Tier 2:Anelastic 球殼(保留密度分層)

方程(加多 ρ₀(r) 參考剖面):
```
∇·(ρ₀·u) = 0           ← 聲波被濾掉但分層保留
∂_t (ρ₀ u) + ... = ... + buoyancy(熵擾動)
∂_t S' + u·∇S₀ = κ·∇²S' + heating
```

| # | Case | 難度 | 驗證對象 |
|---|---|---|---|
| 2.1 | **太陽對流層 differential rotation** | ★★★ | 日震赤道快極區慢 |
| 2.2 | **太陽 11 年 dynamo (α-Ω)** | ★★★★ | butterfly diagram |
| 2.3 | **紅巨星深對流**(Mach 0.01-0.1,分層 10⁶) | ★★★ | 星震學 ℓ=1 模 |
| 2.4 | **木星深內部對流**(高旋轉 → T-P 柱) | ★★★★ | Juno 重力場 |
| 2.5 | **太陽子午環流** | ★★★ | Helioseismology 測流 |

**Tier 2 戰略**:2.1 是門檻(需要穩定的 anelastic integrator),穩了後 2.2-2.5 都是參數擴展。

### Tier 3:2D 薄球殼(θ, φ only)

方程(Shallow-water on sphere 或 2D NS on sphere,徑向積分掉):
```
∂_t u_h + u_h·∇_h u_h = -∇_h p + Coriolis + β·v 類比
∇_h·u_h = 0  (or barotropic SW)
```

| # | Case | 難度 | 驗證對象 |
|---|---|---|---|
| 3.1 | **Rossby 波傳播**(線性) | ★ | 色散關係 |
| 3.2 | **Jovian zonal bands**(β 效應 + forced turb) | ★★ | Rhines 尺度 |
| 3.3 | **大紅斑類 coherent vortex** | ★★ | 湍流自組織壽命 |
| 3.4 | **Hadley cell 類環流**(熱梯度驅動) | ★★ | 地球大氣經向環流 |
| 3.5 | **Polar vortex**(極區強旋轉氣旋) | ★★ | 地球平流層 |
| 3.6 | **Held-Suarez benchmark**(GCM 標準測試) | ★★ | 社群對標數據 |

**Tier 3 戰略**:最便宜最快,2D only VRAM < 1 GB。3.2 視覺震撼 — 幾小時看到 band 自發形成。

### Tier 4:Dynamo / MHD 擴展

| # | Case | 難度 | 驗證對象 |
|---|---|---|---|
| 4.1 | **Kinematic dynamo**(u 給定,B 演化) | ★★ | α / Ω 效應檢驗 |
| 4.2 | **Full dynamo**(地核 Boussinesq MHD) | ★★★ | Christensen 1.5 的全版本 |
| 4.3 | **Solar dynamo**(anelastic + MHD) | ★★★★ | butterfly,週期反轉 |
| 4.4 | **Magnetic reconnection**(反向 B IC) | ★★★ | 重聯率 scaling |
| 4.5 | **Omega effect 可視化**(教學) | ★ | poloidal → toroidal 轉換 |

### Tier 5:雙擴散 / 熱鹽對流

| # | Case | 難度 | 驗證對象 |
|---|---|---|---|
| 5.1 | **恆星 semiconvection**(T + μ 雙擴散) | ★★★ | 恆星演化混合率 |
| 5.2 | **海洋溫鹽環流**(縮比) | ★★ | thermohaline 理論 |
| 5.3 | **Finger instability** | ★★ | 紅巨星殼層混合 |

### Tier 6:旋轉主導場景

| # | Case | 難度 | 驗證對象 |
|---|---|---|---|
| 6.1 | **Taylor-Proudman 柱 scaling** | ★★ | Ekman 數依賴 |
| 6.2 | **Inertial waves**(球面 Coriolis 波) | ★★ | 色散關係 |
| 6.3 | **Precession-driven flow** | ★★★ | 地核被月球擺動驅動 |
| 6.4 | **Libration-driven flow** | ★★★ | 水星 / 月球核 |

### Tier 7:高階 / 外圍(遠期)

| # | Case | 難度 | 備註 |
|---|---|---|---|
| 7.1 | **超流氦球殼**(兩流體) | ★★★★ | 中子星殼層 |
| 7.2 | **粘彈性對流** | ★★★★ | 冰巨行星地函 |
| 7.3 | **Hall MHD dynamo** | ★★★★ | 中子星磁場演化 |

---

## 發展 Phase(打怪時間軸)

### Phase A: 2D 薄殼框架(1-2 週)→ 秒殺 Tier 3 小怪

**新求解器**: `sph2d_spectral_solver.{cu,cuh}` + `sph2d_spectral_kernels.cu`

- 球諧 Y_l^m 展開,徑向被積分掉
- SHTns 或自寫 Legendre transform(先 O(N²) naive,perf 不夠再上 SHTns)
- 驗證:Rossby 波色散(Case 3.1) — 有解析 benchmark
- Deliverable:秒殺 3.1, 3.2, 3.3(木星 band + vortex)

Phase A 結束後會有一個**可驗證的球面 2D 譜求解器**,相當於當前 `pseudo_spectral` 的球面版。

### Phase B: 3D Boussinesq 球殼框架(2-3 週)→ Tier 1 掃蕩

**新求解器**: `sph3d_boussinesq_solver.{cu,cuh}` + `sph3d_boussinesq_kernels.cu`

- Chebyshev T_n(r) × Y_l^m(θ,φ) 雙基底
- 徑向 Chebyshev-tau method 處理 BC
- IMEX 時間積分:對流顯式,擴散 / Coriolis 隱式(對角)
- 驗證:Case 1.1 臨界 Ra,Case 1.2 T-P 柱
- Deliverable:秒殺 1.1, 1.2, 1.3

**VRAM 估算**(3D 128³ 級別): 多個 (N_r × (L+1)² × 16B) buffer ≈ 少 GB,消費卡可行

### Phase C: 加磁場 → Tier 4 主力 + Tier 1 boss(1-2 週)

**擴展求解器**: `sph3d_bdyn_solver.{cu,cuh}` (從 sph3d_boussinesq fork)

- 加磁向量場 B,用 poloidal/toroidal 分解(球諧下自動 ∇·B=0)
- 感應方程加入 IMEX 隱式處理
- 驗證:**Christensen 2001 benchmark**(Case 1.5 / 4.2)── 全球 code 共同基準
- Deliverable:秒殺 4.1, 4.2, 4.5,拿下 1.5 boss

### Phase D: Anelastic 擴展 → Tier 2 天體物理應用(2-4 週)

**新求解器**: `sph3d_anelastic_solver.{cu,cuh}`

- 加 ρ₀(r) 參考剖面(輸入自 MESA 或解析 polytrope)
- 動量方程改用 ρ₀·u,連續方程 ∇·(ρ₀·u)=0
- 熵方程替代溫度方程
- 驗證:Case 2.1 太陽差旋
- Deliverable:秒殺 2.1, 2.3, 2.5,挑戰 2.2 太陽 dynamo(+磁場)

### Phase E: Tier 5-7 自由探索(長期)

按興趣順序,每個 case 都是前面框架的擴展:
- 雙擴散:加第二個擾動場 + 不同 κ
- 旋轉主導:已有 Coriolis,改掃參數
- 外圍:每個需要方程層面擴展,但框架複用

---

## 反覆使用的打怪策略

**小怪秒殺(★ - ★★)**:
- Phase A 後,3.1 / 3.2 / 1.1 / 4.5 應該是**幾小時 wall time + 少量 LOC** 就能拿下
- 工具鏈已經提供 IFRK3 + 守恆形式 + cross-check,bug 很快就能抓
- Taylor-Green 風格的解析驗證要在每個 solver 加一個,作為「工具可信」的金標準

**中怪穩紮(★★★)**:
- 2.1 / 2.3 / 4.2 / 5.1 需要物理理解 + 參數試驗
- 每個建議給 1-2 週,搭配 diagnostic CSV + spectrum 分析腳本
- 不趕,保證對得上文獻

**Boss 戰(★★★★+)**:
- 2.2 太陽 dynamo、2.4 木星深內部、7.x 是真研究級問題
- 框架完備後再打,當時已有足夠 validation 基礎

---

## 基礎設施先做清單(Phase A 開始前)

1. `src/gpu/sph_transforms.{cu,cuh}`:球諧 / Chebyshev transform 封裝,所有 sph* solver 共用
2. `scripts/render_sph_*.py`:球面可視化(Mollweide / Orthographic / meridional slice)
3. `scripts/spectrum_sph_*.py`:球諧譜 E(l), E(l,m) 分析
4. Taylor-Green-類比 analytic benchmark 選一:
   - **2D 薄殼**:Rossby 波線性色散(Case 3.1)
   - **3D Boussinesq**:球殼 Marginal stability(臨界 Ra 有半解析值)

---

## 參考 / 對標 codes

| Code | 方法 | 主攻 |
|---|---|---|
| ASH | Boussinesq / anelastic 球殼譜 | 太陽對流(已經典) |
| Rayleigh | 現代 anelastic 譜 | 恆星對流 dynamo |
| MagIC | Boussinesq 球殼譜 | 地核 dynamo |
| Dedalus | 通用譜框架 | 任意 PDE 譜方法 |
| SHTns | 球諧 transform 庫 | 高性能基礎設施 |
| FastSphericalHarmonics | LT 優化 | 極快的球諧 transform |

---

## 不做的事(避免歧路)

- 不碰強激波 / Mach~1 全可壓縮 —— 用錯工具會白忙
- 不做 AMR / 非結構網格 —— 譜方法的前提是結構化基函數
- 不在 sph* solver 內塞輻射 / 化學反應 —— 那些要獨立 operator split 或新 solver
- 不把 anelastic 和 Boussinesq 硬塞同一個 solver —— 開新 file
