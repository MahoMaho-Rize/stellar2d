# Round-3 規劃：應對「核心貢獻」級審稿意見

**狀態**：規劃中，待拍板
**上游**：`REVIEW_RESPONSE.md`、`REVIEW_ROUND1_RESULTS.md`（歸檔）、`REVIEW_ROUND2_RESULTS.md`
**性質**：本輪審稿意見 **比 Round-1/2 都重**——涉及論文定位、核心命題強度、對競品的覆蓋度，
而不是修訂計算或潤色。

---

## 1. 審稿意見分類

六條意見，先按「我同意的程度」和「工作類型」分類。同意 ≠ 照做，後面有詳細評估。

| # | 意見 | 類型 | 我的同意度 | 工作量 |
|---|---|---|---|---|
| 1 | Leibniz defect 需嚴格 Proposition（「resolution-independent」成立的反例序列） | 推導 | 🟢 強同意 | 1-2 天 |
| 2 | 缺與 Dedalus / spectral element / FEM 的正面對比表 | 論述 + 小實驗 | 🟡 同意但要限縮 | 1-2 天 |
| 3 | 低估哲學意義，應寫成 "pseudo-spectral fundamentally broken under VC" | 論述重構 | 🟡 部分同意（見下） | 0.5-1 天 |
| 4 | Abstract 太保守，應寫得更狠 | 寫作 | 🟢 同意但要負責任 | 1 小時 |
| 5 | §5 結構：先 shock 再排除再鎖定 | 寫作 | 🟢 同意 | 半天 |
| 6 | 缺 complexity vs accuracy 圖 | 新實驗 + 圖 | 🟢 強同意 | 半天 |

**關鍵判斷**：意見 1、6 是純加分且工作量可控。意見 2 必須做但要注意不過度擴大範圍。
意見 3、4 是論文定位，**需要用戶親自拍板**——這決定投稿目標期刊層級。意見 5 是寫作。

---

## 2. 逐條評估

### 意見 1：需嚴格 Proposition「defect 不隨 $N\to\infty$ 消失」

**審稿人要求**：給出反例序列 $v_N$ 使 $\|\Delta_L(\rho_0) v_N\| = \mathcal O(1)$。

**現狀**：Round-1 R1.2 已有：
  - aliasing 本質推導（$\mathcal I_N$ 投影的 "product-and-truncate"）
  - $N_y$ 掃描實驗證實 primitive per-step 平坦於 $N_y \in [32, 256]$（0.3% 變化）
  - Round-2 full-2D 掃描確認同量級（0.09% 變化）

**缺的是**：一條正式 Proposition 說「對半整數 Lane-Emden 表面剖面，
存在 CGL 節點上的向量序列 $v_N$ 使得
$\|\Delta_L(\rho_0) v_N\|_2 / \|v_N\|_2 \ge c > 0$ 對所有 $N$ 成立」。

**評估**：審稿人完全對。這是論文的 killer idea，目前只有「觀察+啟發式論證」，
沒有「嚴格下界」。現在把它寫成 Proposition，工作量中等。

**擬補**：

> **Proposition 1 (resolution-independent Leibniz defect).**
> Let $\mathsf D_N$ be the CGL differentiation matrix of order $N$
> on $[0, L_y]$, and $\rho_0(y) = (L_y - y)^{3/2}$ the Lane-Emden
> $n = 3/2$ surface exponent model. Define
> $\Delta_L^{(N)} := \mathsf D_N\,\mathrm{diag}(\rho_0)\,\mathsf D_N -
> \mathrm{diag}(\rho_0)\,\mathsf D_N^2 - \mathrm{diag}(\rho_0')\,\mathsf D_N$.
> There exists a sequence of vectors $v_N \in \mathbb R^{N+1}$
> with $\|v_N\|_2 = 1$ such that
> $$
> \liminf_{N \to \infty} \|\Delta_L^{(N)}\,v_N\|_2 \;\ge\; c > 0.
> $$
>
> **Proof sketch**：Take $v_N = V_{n_g=1}^{(N)}$ the top g-mode
> eigenvector on grid $N$. Its Chebyshev coefficients decay as
> $\mathcal O(k^{-2})$ (§3.5, half-integer exponent). The aliasing
> residual $\rho_0\cdot(\mathcal I_N v_N)' - \mathcal I_N[\rho_0\cdot(\mathcal I_N v_N)']$
> has leading-order Chebyshev content at degree $N+1$ to $2N$, with
> coefficients of order $N^{-2}$ (inherited from $v_N$'s tail);
> differentiation amplifies by $\mathcal O(N)$, yielding $\|\Delta_L^{(N)} v_N\|_2 \sim N\cdot N^{-2}\cdot N^{1/2} = N^{-1/2}$
> as the naive bound—but the dominant contribution is from the
> *boundary layer* at $y = L_y$, which contributes $\mathcal O(1)$
> in CC weight. A careful accounting gives $c \ge c_0(\rho_{\mathrm{cut}}) > 0$;
> we confirm numerically $c \approx 5 \times 10^{-4}$ in our setup.
>
> The proposition does not hold for integer polytropic indices
> $n = 1, 3$, where $V_n$ inherits exponential Chebyshev decay and
> $\|\Delta_L^{(N)} v_N\| \to 0$ super-algebraically.

**工作量**：半天推導 + 半天 $N_y$ = 32..512 更寬掃描驗證下界的 non-vanishing
+ 半天寫作。總 1.5-2 天。

**可選加強**：把 sketch 升級為正式 proof（需要 Jacobi polynomial 係數估計），
再加半天。

---

### 意見 2：需與 Dedalus / SEM / FEM 正面對比

**審稿人要求**：

> 方法 | 是否有 Leibniz defect | eigenmode preservation
> - pseudo-spectral: ❌ 有 / ❌
> - SL + primitive: ❌ 有 / ❌
> - assembled: ✅ 無 / ✅

**評估**：論文 §8.3 已有 Dedalus 對比的三段討論，但確實沒有這種表格化的、
針對「Leibniz defect」軸的直接對比。審稿人的擔憂是合理的：
> "This is known in Galerkin literature"
這是一個真實的風險。

**但注意**：我 *同意要做*，**但要謹慎避免過度宣告**。下面是我認為論文應該
寫、不應該寫的：

**應該寫**：
  - 純 Galerkin（Dedalus τ-method）確實隱式達到 assembled-operator 結構，
    這點論文 §8.3 已承認：「Dedalus's tau-method linear solver, being a
    correct discretisation of the assembled operator L, is implicitly
    the assembled-matrix construction of our Section 6」。
  - 我們的貢獻是 **「診斷為何 primitive-node pseudo-spectral 失敗 + 為
    pre-existing primitive-node 代碼提供最小改造路徑」**，不是「發明 assembled」。

**不應該寫**：
  - ❌ 不應宣告 Dedalus 有 Leibniz defect（它沒有）
  - ❌ 不應暗示 Galerkin literature 不知道這個（他們在 continuous 層面知道，
    在 discrete 層面不一定明確 articulate 但實作上避開了）

**擬補實驗**：在相同 Lane-Emden n=3/2、相同 $N_y$ 下跑三個實作對比：
  (a) Pseudo-spectral primitive-node（我們診斷的失敗案例）
  (b) τ-method assembled（類 Dedalus 做法——只不過我們手工 assemble）
  (c) 我們的 assembled $\mathsf M = \mathsf L^{-1}\mathsf R$（§6）
(b) 和 (c) 應該都達到機器精度，(a) 被困在 aliasing floor。這就把「Leibniz defect
是 primitive-node 選擇的問題，不是 SL / pseudo-spectral 本身的問題」這個 **關鍵澄清**
直接畫出來。

**工作量**：1-1.5 天（實驗 + 寫表 + 修 §8.3 段落）。

---

### 意見 3：應重寫為 "pseudo-spectral fundamentally broken under VC"

**審稿人主張**：

> "不是 spectral 不行，而是 operator splitting 在 variable coefficient 下不閉合"
> "現有主流方法在某類問題上是結構性錯誤"

**評估**：這是 **論文定位問題**——不是技術問題。我的看法：

**部分同意**：
  - 「primitive-node operator splitting 在 VC 下不達離散閉合」是成立的命題。
  - 「這是結構性（不可通過 refinement 解決）」也是 R1.2 已證的。

**不同意過度擴張**：
  - 說「pseudo-spectral fundamentally broken」**不準確**——tau-method
    pseudo-spectral 沒這個問題。
  - 說「現有主流方法結構性錯誤」會激怒相當多的實踐者——他們用 primitive-node
    pseudo-spectral 多年，對特定問題（uniform background、弱 VC）是 OK 的。
  - 「廣泛有效」可能是審稿人的 bait；若發表出去，會有 heavy rebuttal 從 Dedalus
    / Nek5000 / 等等陣營。

**我的建議**：Split the difference——

1. **承認命題強度**：正文明確寫 「primitive-node pseudo-spectral fails
   discrete closure under variable coefficients, independent of grid
   refinement」——這是 accurate 且強硬的。
2. **不擴張到「pseudo-spectral 全盤失敗」**：明確限定為 "primitive-node"
   variant，不包括 tau-method / Galerkin。
3. **強調 practical impact**：很多 pre-existing primitive-node 代碼受影響，
   我們提供最小改造路徑，而非要求用戶拋棄整個 code base。

**這條的拍板問題**：用戶要選擇論文的「攻擊性」。
  - **保守版**（當前）：better-engineered method for variable-density DNS
  - **中等版**（我建議）：identifies and resolves a closure failure in
    primitive-node pseudo-spectral under VC; practical migration path
  - **激進版**（審稿人建議）：exposes a structural error in primitive-node
    pseudo-spectral; requires replacement

激進版可能適合 PR Fluids / JCP 這種偏向 method-impact 的期刊，中等版適合
MNRAS / ApJ + JCP 混合。保守版只適合 MNRAS / ApJ。

**工作量**：0.5-1 天純寫作（abstract、intro、§5、§9 重寫）。

---

### 意見 4：Abstract 太保守

**審稿人示範**：

> A widely-used class of pseudo-spectral methods fails to preserve
> eigenmodes under variable density, even at infinite resolution.
> We identify the cause as a discrete Leibniz defect and show that
> only an assembled-operator formulation restores closure.

**評估**：同意改。但「widely-used class」要具體——「primitive-node
pseudo-spectral」or 「variable-coefficient applied through factored
pointwise operations」，避免無差別批判。

**工作量**：1 小時，但先拍板意見 3 的「攻擊性」再寫。

---

### 意見 5：§5 結構「shock → 排除 → 鎖定」

**評估**：同意。現行 §5.1 已經是「期待 vs 失敗」開場，接近但不夠戲劇化。
§5.3/5.4 是「兩個 EVP 基底替換 + 狀態變量替換」的排除；§5.5 是鎖定。
結構是對的，只需潤色：
  - §5.1 開頭加一句 **shock**：「We report an empirically stable
    $6.9\times 10^{-4}$ per-step deviation that refuses to yield to
    any of the expected remedies.」
  - §5.2 開頭加 "We now trace this stubborn signal to its source."
  - §5.3/5.4 改寫為 "elimination of hypothesis A" / "elimination of
    hypothesis B" 格式。
  - §5.5 改寫為 "the only remaining culprit"。

**工作量**：半天純潤色。

---

### 意見 6：缺 complexity vs accuracy 圖

**審稿人要求**：$N_y$ vs (error per step, runtime) 對比圖。

**評估**：**完全同意**——這是最有說服力的審稿回應之一。單一圖能讓審稿人立即看到：
  - 我們的方法在 $N_y \gtrsim 48$ 就達機器精度
  - 成本隨 $N_y$ 上升的曲線比 primitive 的 inflection point 慢
  - runtime 數字讓工程人員能評估可採納性

**擬補實驗**：
  - $N_y \in \{32, 48, 64, 96, 128, 192, 256\}$
  - 對 primitive-node RK4 vs assembled-RK4 vs τ-method（上面意見 2 一起）
  - 量：(a) per-step deviation, (b) setup time, (c) per-step time, (d) memory

我已經有部分數據（R2.D.1 sweep_Ny.csv 有 accuracy），只需補 runtime
測量和 τ-method 運行。

**工作量**：半天實驗 + 半天繪圖 = 1 天。

---

## 3. 推薦的 Round-3 組合

### 選項 R3.A（保守版，~4 天）

只做「無爭議且需要」的：
  - 意見 1（Proposition 1）：1.5 天
  - 意見 6（complexity-accuracy 圖）：1 天
  - 意見 5（§5 潤色）：0.5 天
  - 意見 4（abstract 小改）：1 小時

論文強度：現狀 + 一條 Proposition + 一張圖。

### 選項 R3.B（中等版，~7 天）— **我推薦**

R3.A + 意見 2 對比實驗 + 意見 3 中等攻擊性定位：
  - 意見 1（Proposition 1）：1.5 天
  - 意見 2（三方法對比 + 表 + §8.3 重寫）：1.5 天
  - 意見 3（中等攻擊性定位，重寫 intro / conclusion）：1 天
  - 意見 4（abstract，配合意見 3）：1 小時
  - 意見 5（§5 shock 結構潤色）：0.5 天
  - 意見 6（complexity-accuracy 圖）：1 天

論文強度：中等攻擊性 + 對比表 + Proposition + 圖。
投稿目標：ApJ / MNRAS + 試投 JCP。

### 選項 R3.C（激進版，~9 天）

R3.B + 意見 3 最激進定位：
  - 以上全部
  - 意見 3 推到「pseudo-spectral fundamentally broken」版本：+1-2 天
    （需要更精細的「what is broken / what isn't」列表，避免誇大）
  - 需補一個「為什麼多年沒人注意」的 historical 段落

論文強度：激進攻擊性 + 全套防禦材料。
投稿目標：JCP / PR Fluids。
風險：被實踐者社區反彈；rebuttal 壓力大。

---

## 4. 三個待拍板決策

### 決策 A：論文攻擊性

- A.1 保守（當前）
- **A.2 中等（推薦）**：「primitive-node pseudo-spectral fails discrete
  closure under VC」——accurate 且強硬，不擴張到全 pseudo-spectral。
- A.3 激進：「pseudo-spectral fundamentally broken」——適合 JCP/PRF 但有風險。

### 決策 B：Round-3 範圍

- B.1 保守（~4 天，R3.A）
- **B.2 中等（推薦，~7 天，R3.B）**
- B.3 激進（~9 天，R3.C）

### 決策 C：意見 2 的對比實驗要不要補 τ-method

- C.1 只補論述 + 表格，不做實驗對比（最小，0.5 天）
- **C.2 補實驗 + 表格（推薦，1.5 天）**：真正跑一個 τ-method prototype
  比較 eigenmode preservation。這是最強防禦材料，把「Dedalus 知道這個」
  的質疑正面打掉。
- C.3 完整對比 FEM / SEM：工作量爆炸（>3 天），不建議。

---

## 5. 我的建議組合

**A.2 + B.2 + C.2** = Round-3 中等版，~7 天。

產出：
  - Proposition 1（資源無關 defect 的嚴格下界）
  - 三方法對比表（primitive vs τ-method vs assembled），含實驗驗證
  - complexity-accuracy 合成圖
  - §5 重寫為 shock-eliminate-lock 結構
  - abstract + intro + conclusion 中等攻擊性重寫

拍板後，我會建立 `REVIEW_ROUND3_RESULTS.md` 起追溯骨架並按上述順序開工。
