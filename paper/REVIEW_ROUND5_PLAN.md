# Round-5 審稿應對（極簡版）

六條意見，我的評估 + 擬議改動 + 工作量：

---

## 1. 語氣非學術（"shock result", "Pareto-dominates"）

**評估**：✅ 完全同意。這確實是 marketing 語言，不該在 JCP-level 稿件出現。

**擬改**：
- §5.1 heading `The shock result` → `Empirical failure of eigenmode preservation`
- §8.3 三處 "Pareto-dominates" → 具體量化（「比 τ-method 低 4 個量級 per-step，記憶體 3x 省」）
- Fig 8.1 caption 同樣去 marketing 詞
- Abstract 最後一句 "Pareto-dominates the τ-method" → 量化版
- 全文 grep "shock"、"Pareto"、"killer"、"failure" 語氣過強處，統一中性化

**工作量**：30 分鐘

---

## 2. Proposition 1 過度包裝

**評估**：🟡 部分同意。

- 同意的部分：proof sketch 不是嚴謹證明，數值驗證不能直接冠以 "Proposition" 標籤（JCP 讀者期待定理有泛函分析層級的論證）。
- 不同意的部分：「非交換矩陣不能替換是 trivial」這個反駁過強了——關鍵不是「有誤差」，是「誤差不隨 N 消失」，這需要 ε→0 scaling law 才能論證，不是 trivial。

**擬改（建議選 B）**：
- **選 A（降級）**：`Proposition 1` → `Observation 1` / `Empirical finding 1`。保守，放棄定理地位。
- **選 B（保留 + 誠實 caveat）**：保留 `Proposition 1` 標題，但在證明草圖後加一段 `Remark`：
  > "The proof above is a sketch in the sense that it rests on the variance estimate $\mathrm{std}(k_x^2 N^2) > 0$ for non-constant $N^2$, which we verify numerically in Section 5.4 but do not prove in functional-analytic generality. A fully rigorous proof requires spectral-basis regularity estimates that go beyond the scope of the present paper; we state (1)–(3) as a finite-grid proposition supported by both the sketch and the numerical scaling law."
- **選 C（升級為嚴格證明）**：補 1–2 天做完整泛函分析推導。工作量大，收益相對小。

**工作量**：選 A 或 B = 15 分鐘；選 C = 1–2 天

---

## 3. IMEX 稻草人謬誤

**評估**：🟡 部分同意。我們確實挑了最弱 IMEX。§7.2 雖已有 fairness caveat，但審稿人仍不滿——caveat 需要更強、挪到 Finding 2 開頭（而非段末）。

**擬改**：
- §7.2 Finding 2 開頭直接改為：
  > "We document the failure of the simplest second-order IMEX combination (CN-AB2), not of the IMEX family as a whole. Higher-order IMEX schemes (IMEX-RK3, IMEX-BDF3) employ $L$-stable or SSP discretisations that may avoid the instability described below; we have not tested these. Finding 2 should therefore be read as a minimum-baseline result motivating Strang-split, not as a negative result for IMEX more broadly."
- 把原 fairness caveat 從段末上移
- §7.5 summary 類似調整

**工作量**：15 分鐘

---

## 4. 能量漂移「掩蓋」

**評估**：✅ 同意。這是六條中最尖銳的一條。"Diagnostic artefact" 措辭確實像在掩蓋問題。

**擬改（建議選 A）**：
- **選 A（誠實承認 + 補 corrected diagnostic）**：在 §7.3 誠實寫：
  > "The $+2.6$ drift is a real discretisation-level bias of the near-wall energy functional, not merely a diagnostic artefact to be dismissed. A corrected diagnostic restricted to the interior $\{y : \rho_0(y) > 2\rho_{\mathrm{cut}}\}$ reduces the drift by two orders of magnitude (to $\sim 10^{-2}$), localising the bias to the $\rho_{\mathrm{cut}}$ floor region where the discrete $b^2/N^2$ weighting is ill-conditioned. Full regularisation of this diagnostic is an open issue; within the present paper it does not bias the three-scheme comparison because all three schemes inherit the same near-wall functional, but it is acknowledged as an unresolved near-wall numerical issue that a fully honest energy-balance analysis must address."
- 需要一個小實驗驗證 2 orders of magnitude 的數字。30–60 分鐘。
- **選 B（只改文字，不跑實驗）**：同樣誠實版文字，但去掉「reduces by two orders of magnitude」的具體數字，改為「a corrected diagnostic that excludes the near-wall floor region produces substantially smaller drift (at a level consistent with the discretisation of the interior fields)」。0 實驗工作量。

**工作量**：選 A = 30–60 分鐘（含實驗）；選 B = 10 分鐘

---

## 5. 可擴展性侷限（non-separable 背景）

**評估**：✅ 同意。目前 §8.6 Open Question 2 已經提到，但位置太弱。應該在 §8.1 Scope 明確寫進 scope 限制，不能讓讀者以為方法普適。

**擬改**：
- §8.1 Scope and limits 加一段明確：
  > "The $\mathcal O(n_h \cdot N_y^2)$ memory footprint and $\mathcal O(N_y^2)$ per-substep cost of the assembled scheme both assume that the background $(\rho_0(y), N^2(y))$ is *radially separable* — that the horizontal Fourier decomposition block-diagonalises the elliptic operator so that each $\mathsf L_{k_x}$ acts on an $N_y$-vector rather than a full 2D state. If $\rho_0$ or $N^2$ acquires horizontal structure (through rotation-induced baroclinicity, global-scale circulation, or ellipticity), the assembled matrix becomes $(N_x N_y) \times (N_x N_y)$ and the efficiency advantage relative to iterative / multigrid approaches vanishes. Proposition 1 continues to hold in the non-separable case, but the cost argument for $\mathsf L^{-1}\mathsf R$ assembly does not."
- §8.6 Open Question 2 配合收縮（不重複）

**工作量**：15 分鐘

---

## 總計推薦組合

**A2.B + A4.A + 其他全做** = 約 **90 分鐘** 總工時（含 R5.4 小實驗）。

或者如果不想跑 R5.4 實驗 → **A4.B（只改文字）** = **50 分鐘**。

---

## 待拍板

1. **R5.2 Proposition 去留**：A（降級 Observation）/ **B（保留 + caveat，推薦）** / C（升級嚴格）
2. **R5.4 能量漂移**：**A（補小實驗，推薦）** / B（只改文字）

回覆「用推薦組合」或逐條指定。
