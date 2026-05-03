# Round-4 規劃：JCP/PRF 鋒利版本重排

**狀態**：規劃中，**待用戶拍板多個決策**
**上游**：
  - `REVIEW_ROUND1_RESULTS.md`（推導、experimental floor）—已歸檔
  - `REVIEW_ROUND2_RESULTS.md`（full-2D sweep、IMEX stability、ρ_cut）—已完成
  - `REVIEW_ROUND3_RESULTS.md`（**Proposition 1 v5**、三方法對比、complexity-accuracy、scaling law）—已完成

**性質**：本輪是 **論文章節重排 + Proposition 1 v5 落地**——把 Round-1/2/3 的所有推導與實驗按「現象 → 排除 → 機制 → 唯一解」四階段鋒利敘事組織起來。

---

## 1. 當前各章長度與重寫幅度

| 節 | 當前 | 目標 | 重寫幅度 |
|---|---|---|---|
| §1 Intro | 172 行 | ~140 行 | 小改（加 killer sentence、提前劇透 shock result）|
| §2 Setting | 152 行 | ~110 行 | 中改（移 Lane-Emden 推導細節到 Appendix）|
| §3 SL spatial | 223 行 | ~160 行 | 中改（收斂分類推導移 Appendix）|
| §4 GYRE EVP | 165 行 | ~130 行 | 小改（主要是 R1.3 error budget 併入 + 3.6e-5 → 9.1e-9 數字更新）|
| **§5 TD failure** | 229 行 | **~200 行** | **大改（按 "shock → rule-out → Proposition 1 → Dedalus 對比" 重排）** |
| **§6 Assembled** | 232 行 | **~150 行** | **中改（原理放前，CUDA 細節移 Appendix）** |
| §7 Nonlinear | 206 行 | ~140 行 | 中改（IMEX stability + fairness caveat + 能量診斷）|
| §8 Discussion | 245 行 | ~120 行 | **大改（精簡到 1.5 頁，抬拔到 "spectral accuracy ≠ operator consistency" level）** |
| §9 Conclusions | 92 行 | ~60 行 | 小改（三句話結構）|
| §10 Refs | 90 行 | ~95 行 | 小增（Higham 2002 等）|
| Appendix A (new) | 0 | ~100 行 | 新增（移轉內容）|
| Appendix B (new) | 0 | ~80 行 | 新增（reproducibility checklist）|

**總頁數**：目前約 25-26 頁 → 目標 21-22 頁正文 + 3 頁 Appendix。

---

## 2. 逐節變動細節

### §1 Intro — 提前劇透、killer sentence

**必改**：
  - 第一段首句換為：「Pseudo-spectral methods are widely assumed to retain spectral accuracy under smooth variable coefficients. We show that this assumption fails at the discrete operator level.」
  - 加「shock result」段：「Even when the eigenvalue problem is solved to machine precision, time-domain evolution fails to preserve eigenmodes, with an $\mathcal O(10^{-4})$ per-step deviation independent of resolution.」
  - Abstract 或 §1 結尾放 locked killer sentence。

**可刪**：
  - §1.3 Python prototyping methodology 壓縮 2→1 段（詳細案例推到 §8.5 或 Appendix）
  - 過長的天文物理背景段落

### §2 Setting — 移 Lane-Emden 推導

**保留**：anelastic 方程、背景定義、Brunt-Väisälä 物理。
**移 Appendix A**：Lane-Emden ODE (2.5) 詳細積分、$\rho_{\mathrm{cut}}$ 敏感性表格。
**加一句 framing**：「We deliberately adopt a standard pseudo-spectral formulation to isolate consistency rather than approximation errors.」

### §3 SL spatial — 收斂分類移 Appendix

**保留**：Liouville 代換、SL 基底、Poisson projection 七步流程。
**移 Appendix**：§3.5 polytropic-index dichotomy 的完整證明（保留結論表）。
**加一句**：「The spatial discretisation closes to machine precision and is externally validated (§4), isolating any subsequent failure to the time-stepping operator.」

### §4 GYRE EVP — Error budget 併入

**必改**：
  - 更新精度數字：$3.6\times 10^{-5} \to 9.1\times 10^{-9}$（cubic spline）
  - §4.2 加「Physical model equivalence」小段澄清四變數完整性
  - §4.5 改寫為 "Error budget decomposition"（R1.3 三軸掃描）

### §5 TD failure — **重排核心**

**新結構**（按用戶建議）：
```
5.1 Shock result (the empirical failure)
    → 直接上 6.9e-4 per-step、100 週期崩潰

5.2 Rule-out (three eliminations in bullets)
    → resolution independence (R1.2)
    → timestep independence (paper §5.1 現有)
    → boundary-condition robustness
    → Leibniz defect IS NOT the cause (new! R3-P1.7)

5.3 Proposition 1 (the mechanism)
    → locality gap: global L⁻¹ vs pointwise diag(1/ρ)
    → three clauses: resolution-independent floor + scaling law + structural character
    → proof sketch

5.4 Scaling-law verification (new subsection)
    → ε → 0 experiment, Fig 5.1 scaling plot

5.5 Why Galerkin/τ-method avoid this
    → 併入 §8.3 部分內容、加三方法對比數字
```

**內容來源**：
- §5.1/§5.2 現有大致保留 observation
- §5.2 新段 "Leibniz is not the leading cause"：來自 P1.7 推翻舊假說的數據
- §5.3 Proposition 1：來自 Round-3 P1.7 最終版
- §5.4 scaling-law：來自 Round-3 P1.7 scaling_plot.png
- §5.5 來自 Round-3 P2 三方法對比

**刪除**：§5.3 "changing basis" 和 §5.4 "changing state variable" 的長篇——壓縮為兩段要點。
**移 Appendix**：Leibniz identity (5.4) 的完整符號推導（保留公式）。

### §6 Assembled — 原理放前，細節移 Appendix

**新結構**：
```
6.1 Principle (one-line statement)
    → "We enforce closure by constructing M = L⁻¹R explicitly"
6.2 Mathematical properties (eigenbasis invariance, RK4 closure)
    → 保留 Round-1 R1.1 formal Theorem 6.1
6.3 Results (dev/step = 10⁻¹⁵)
    → 保留 Python + CUDA 數字
```

**移 Appendix**：CUDA memory layout、kernel、cuFFT 細節（50-60 行）。

### §7 Nonlinear — 壓縮 + IMEX stability 升級

**改動**：
  - §7.3 Finding 2：從量級估算 → 引用 Round-2 R2.1 IMEX 放大矩陣分析
  - §7.5 加 IMEX fairness caveat（R1.4 內容）
  - 壓縮 §7.4 能量診斷為半頁

### §8 Discussion — 抬拔到「定理級」

**新 §8 結構**（從 245 行壓到 ~120 行）：
```
8.1 Scope and limits (壓縮到半頁)
8.2 Relation to GYRE (併入 §4.5 error budget，只剩一段)
8.3 Relation to Dedalus / Galerkin
    → 主要內容：三方法對比表（R3 P2）+ complexity-accuracy 圖（R3 P3）
    → 加核心句："Galerkin frameworks naturally avoid this issue,
       but the necessity of operator assembly for primitive-node
       codes has not been explicitly identified."
8.4 Methodology (Python-first) — 壓縮 1 段
8.5 Open questions — 保留
```

**加抬拔句**（作為 §8 結尾）：
  - "Spectral accuracy does not imply operator consistency under variable coefficients."
  - "The distinction between basis accuracy and operator closure is essential."

### §9 Conclusions — 三句話結構

```
1. We identified a resolution-independent time-stepping failure in
   primitive-node pseudo-spectral methods under variable coefficients.
2. The failure is a structural operator-consistency error: a global
   elliptic inverse is replaced by a pointwise scaling.
3. The unique minimal-cost fix is assembled-operator time stepping;
   it restores discrete closure to machine precision and is
   externally validated against GYRE to 9.1e-9.
```

### Appendix A - 移轉內容

- §2.3 Lane-Emden 細節
- §3.5 polytropic-index dichotomy 推導
- §5 Leibniz identity 符號推導
- §6 CUDA implementation 詳細
- ρ_cut 敏感性完整表格（R2.2 數據）

### Appendix B - Reproducibility

列表：實驗 → 腳本 → 數據文件 → git commit。

---

## 3. 待拍板決策

### 決策 D.1：§5 重排激進度

- **D.1.a 激進版**（推薦）：完全按上面 5.1-5.5 新結構重寫，舊 §5.3/§5.4 的
  三個 rule-out 實驗壓縮為 bullet list（每條 1 段），結構上類似定理證明。
- **D.1.b 保守版**：保留現有 §5.3/§5.4 三段結構，只在 §5.5 後追加 Proposition 1。
  改動小但失去「一刀切進 reviewer」的鋒利度。

### 決策 D.2：Abstract 攻擊性

- **D.2.a 激進**（推薦，配 Round-3 user framing）：
  > A widely-used class of pseudo-spectral methods fails to preserve
  > eigenmodes under variable density, even at infinite resolution.
  > The failure arises from replacing the inverse of a global elliptic
  > operator by a pointwise scaling — a structural inconsistency, not
  > a singularity or a truncation effect. We identify the mechanism,
  > quantify it with a scaling law, and show that only explicit
  > operator assembly restores closure.
- **D.2.b 中等**：保留 Round-3 規劃的 "primitive-node variant" 限定。
- **D.2.c 保守**：仍保留「two ideas」舊 framing。

### 決策 D.3：CUDA / engineering 內容去留

- **D.3.a 徹底移 Appendix**：正文只講 "cuFFT + DGEMV per wavenumber" 一句，
  Appendix 細節。投 JCP 最合適。
- **D.3.b 保留 §6.4** 作為「實作可行性」節（當前做法）。投 ApJ/MNRAS 可保留。

### 決策 D.4：Proposition 1 三個 clause 全部放正文？

Proposition 1 有三個子條款（resolution-independent floor + scaling law +
structural character）。
- **D.4.a 三條全放 §5.3**（推薦）：完整定理，審稿人一目了然。
- **D.4.b 主條款 §5.3，scaling-law 推 §5.4**：分節更結構化，也可。

### 決策 D.5：Fig 5.1 / Fig 5.2 / Fig 8.1 是否都放正文？

- **Fig 5.1 matrix_heatmap.png**（$L^{-1}$ vs $\mathrm{diag}$ 並排）——強推薦放 §5.3。
  視覺衝擊最強。
- **Fig 5.2 scaling_plot.png**（$\varepsilon^{1.00}$ scaling + scale-invariant rel）——
  推薦放 §5.4 作為 Proposition 1 支撐。
- **Fig 8.1 complexity_accuracy.png**（三方法 Pareto）——推薦放 §8.3。

**D.5.a**：三圖全放正文。
**D.5.b**：Fig 5.1 正文，其他 Appendix。節省篇幅但弱化視覺。

### 決策 D.6：§4 GYRE 精度數字更新範圍

原 Round-1 的問題還在：$3.6\times 10^{-5} \to 9.1\times 10^{-9}$ 影響 §1.4、§4.4、§4.6、§8.2 四處。
- **D.6.a 全更新**（推薦）：cubic spline 當前是 default，論文應報最好數字。
- **D.6.b 保留舊數字**：改動小但精度溢價跑掉。

### 決策 D.7：投稿目標期刊定位

這決定論文寫作 tone 和 §8 篇幅：
- **D.7.a ApJ / MNRAS**：偏應用，保留天文 motivation，§2 設置詳細，~24 頁。
- **D.7.b JCP**（推薦 based on Round-3 user framing）：偏方法論，抬拔 Proposition，§5 $\to$ §6 是論文主線，~20 頁。
- **D.7.c PR Fluids**：偏 principle，更激進 framing，~18 頁。

這個決策回饋到 D.1、D.2、D.3 的激進度。

---

## 4. 推薦組合與工作量

**推薦**：D.1.a + D.2.a + D.3.a + D.4.a + D.5.a + D.6.a + D.7.b（JCP 鋒利版）

### Round-4 執行順序（七天預估）

| 天 | 工作 | 產出 |
|---|---|---|
| 1 | §5 完全重寫（shock → rule-out → Prop 1 → scaling → Galerkin）+ 插入 Fig 5.1, 5.2 | §5 new |
| 2 | §6 原理前置 + 移 CUDA 到 Appendix A + 保留 Theorem 6.1 formal | §6 new + App A 片段 |
| 3 | §4 error budget 併入 + 全文 3.6e-5 → 9.1e-9 替換 | §4 new |
| 4 | §7 IMEX stability 升級 + fairness caveat + 能量診斷壓縮 | §7 new |
| 5 | §8 抬拔重寫 + 加三方法對比 + Fig 8.1 + killer quotes | §8 new |
| 6 | §1 intro shock 提前 + abstract killer + §9 conclusion 三句 + §2/§3 壓縮 | §1/§2/§3/§9 new |
| 7 | Appendix A + B 組裝 + 99_concat.sh 調整 + PDF 重新渲染 + 整體 review | paper.pdf |

### 產出清單

- 重寫後的 §1-§9（~1800 行正文）
- 新 Appendix A/B（~200 行）
- 最終 `paper.pdf`（目標 ~22 頁主文 + 3 頁附錄）
- 本 Round-4 追溯文檔 `REVIEW_ROUND4_RESULTS.md`

---

## 5. 風險評估

**高風險**：§5 完全重寫有可能打破現有文本平衡、丟失某些重要細節。
→ 緩解：每節改動前先備份到 `paper/archive_pre_round4/`。

**中風險**：Proposition 1 v5 formal statement 可能在 LaTeX 渲染下需要多輪調試。
→ 緩解：在紙本推導完整後一次性寫入。

**低風險**：數字全文替換（3.6e-5 → 9.1e-9）有遺漏。
→ 緩解：grep 確認所有出現位置再替換。

---

## 6. 等待用戶的決策回饋

請從 D.1 到 D.7 拍板，或給「用推薦組合」三字指令。

拍板後我會：
1. 備份現版本到 `paper/archive_pre_round4/`
2. 建立 `REVIEW_ROUND4_RESULTS.md` 追溯骨架
3. 按上面七天順序開工
