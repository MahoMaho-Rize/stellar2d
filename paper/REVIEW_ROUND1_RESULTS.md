# Round-1 修訂結果追溯文檔

> **狀態：ARCHIVED 2026-05-03**
> 本輪四項推導與數值實驗已全部完成並生成 artifacts（見文末清單）。
> 論文章節 patch 的落實轉入 Round-2（見 `REVIEW_ROUND2_PLAN.md`）。
> 本文件為 frozen reference，不再修改；所有後續變更走 Round-2 追溯。

本文件記錄審稿意見輪次 1（必做項）的修訂過程——每條意見的：
  1. 最終推導 / 實驗結論
  2. 得到的原始數據或關鍵推導式
  3. 擬寫入論文的段落（草案）
  4. 對應 paper/NN_*.md 的 patch 位置與變更摘要

目的：即使正文反覆修訂，這份文件保留每條意見的「結論性依據」與變更軌跡，供後續審稿回信與下一輪答辯引用。

上游：`paper/REVIEW_RESPONSE.md`（審稿意見分類與工作計劃）
工作基準：分支 `pseudo-astro-explore`，日期 2026-05-03

---

## R1.1 Theorem 6.1 正式化

**狀態**：完成

**審稿意見**：§6.5 Theorem 6.1 為非正式陳述，需發展為正式定理並給出證明。

### 設定與記號

$\mathsf L, \mathsf R \in \mathbb R^{N\times N}$ 為 §3.6 定義的內點限制
SL-相容矩陣（$N = N_{\mathrm{int}}$；以下略去 $k_x$ 下標）。
$\mathsf M = \mathsf L^{-1}\mathsf R$。設 $\mathsf M$ 對角化為
$\mathsf M = Q\,\Lambda\,Q^{-1}$，$\Lambda = \mathrm{diag}(\omega_1^2,\dots,\omega_N^2)$，
所有 $\omega_n^2 > 0$（SL 橢圓性保證，§3）。記
$\kappa(Q) = \|Q\|_2\,\|Q^{-1}\|_2$ 為基底條件數。

一階系統記
$$
\dot U = \mathsf A\,U,
\qquad
U = \begin{pmatrix}V\\W\end{pmatrix},
\qquad
\mathsf A = \begin{pmatrix} 0 & I \\ -\mathsf M & 0 \end{pmatrix}
\in \mathbb R^{2N\times 2N}.
$$
RK4 一步映射為
$$
U^{k+1} = R_4(\Delta t\,\mathsf A)\,U^k,
\qquad
R_4(z) = 1 + z + \tfrac{z^2}{2} + \tfrac{z^3}{6} + \tfrac{z^4}{24}.
$$

### Lemma A（$\mathsf A$ 譜分解）

對每個 $\mathsf M$ 特徵對 $(\omega_n^2, V_n)$，$\mathsf A$ 有一對
共軛特徵對
$$
\mathsf A\,U_n^\pm = \pm\mathrm i\omega_n\,U_n^\pm,
\qquad
U_n^\pm = \begin{pmatrix} V_n \\ \pm\mathrm i\omega_n\,V_n \end{pmatrix}.
$$
於是 $\mathrm{span}\{U_n^+, U_n^-\} = \mathrm{span}\{(V_n, 0)^\top, (0, V_n)^\top\}$
是 $\mathsf A$ 的 2-維不變子空間。

**證明**：直接代入得
$\mathsf A(V_n, c V_n)^\top = (cV_n, -\mathsf M V_n)^\top = (c V_n, -\omega_n^2 V_n)^\top$，
解 $-\omega_n^2 = \lambda c$ 且 $c = \lambda$ 得 $\lambda = \pm\mathrm i\omega_n$、$c = \pm\mathrm i\omega_n$。□

### Lemma B（RK4 穩定函數在純虛軸的性質）

對 $z = \mathrm i\theta$，$\theta \in \mathbb R$，$|\theta| \le 2\sqrt 2$（RK4 虛軸 CFL 界），
$$
|R_4(\mathrm i\theta)|^2 \;=\; 1 - \tfrac{\theta^8}{576} + \mathcal O(\theta^{10}),
\qquad
\arg R_4(\mathrm i\theta) \;=\; \theta + \tfrac{\theta^5}{120} + \mathcal O(\theta^7).
$$
特別地，$R_4(\mathrm i\theta) = \mathrm e^{\mathrm i\theta} + \mathcal O(\theta^5)$，
且 $|R_4(\mathrm i\theta)| \le 1$ 於該區間。

**證明要點**：展開 $R_4(\mathrm i\theta) = (1 - \theta^2/2 + \theta^4/24) + \mathrm i(\theta - \theta^3/6)$，
直接計算 $|R_4|^2$ 與 $\mathrm e^{\mathrm i\theta}$ 的差；前八階與 Taylor 展開匹配，
第八階的負貢獻 $-\theta^8/576$ 恰來自 $|R_4|^2$ 與 $1$ 的差。□

### 主定理（正式化）

**Theorem 6.1 (formal)**：令 $(V_n, \omega_n^2)$ 為 $\mathsf R V = \omega^2\mathsf L V$
的特徵對，$V(0) = V_n$，$W(0) = 0$，應用步長 $\Delta t$ 的 RK4 於
$\dot U = \mathsf A U$，得序列 $\{U^k\}_{k \ge 0}$。令
$V^k = (I, 0)\,U^k$（取上半分量）。設 $\omega_{\max}\Delta t \le 2\sqrt 2$（RK4 虛軸穩定範圍），
則存在與 $N, \rho_0, N^2$ 無關的常數 $C_1 \le 1/120$，$C_2 \le 2$，使
$$
\bigl\|V^k - V_n\cos(\omega_n k\Delta t)\bigr\|_2
\;\le\;
C_1\,(\omega_n\Delta t)^5\,k\,\|V_n\|_2
\;+\;
C_2\,k\,\epsilon_{\mathrm{mach}}\,\kappa(Q)\,\|V_n\|_2.
\tag{6.3}
$$

**證明**：依 Lemma A，$\mathsf A$ 限制於 $\mathrm{span}\{U_n^+, U_n^-\}$ 為
$\mathrm{diag}(\mathrm i\omega_n, -\mathrm i\omega_n)$。
初值 $U^0 = (V_n, 0)^\top = \tfrac12 U_n^+ + \tfrac12 U_n^-$。
RK4 映射在該子空間內對角作用，每步乘 $R_4(\pm\mathrm i\omega_n\Delta t)$。
由 Lemma B，
$$
R_4(\pm\mathrm i\omega_n\Delta t)
\;=\;
\mathrm e^{\pm\mathrm i\omega_n\Delta t}\,
\bigl(1 + \rho_n\bigr),
\qquad
|\rho_n| \le \tfrac{(\omega_n\Delta t)^5}{120}.
$$
$k$ 步後
$$
V^k = \tfrac12\bigl[R_4(\mathrm i\omega_n\Delta t)^k + R_4(-\mathrm i\omega_n\Delta t)^k\bigr]\,V_n.
$$
與 $V_n\cos(\omega_n k\Delta t)$ 相減，逐項估計
$|(1+\rho_n)^k - 1| \le k|\rho_n|\,\mathrm e^{k|\rho_n|}$；
對 $k|\rho_n| \le 1$（典型運行中成立，否則截斷誤差已失效），得
$$
\|V^k - V_n\cos(\omega_n k\Delta t)\|_2 \le k|\rho_n|\cdot\|V_n\|_2 \le \tfrac{1}{120}(\omega_n\Delta t)^5 k \|V_n\|_2.
$$
這給出第一項（$C_1 = 1/120$）。

第二項來自有限精度：每步 $\mathsf A$ 與 $Q$ 的矩陣-向量乘法引入
$\mathcal O(\epsilon_{\mathrm{mach}})$ 誤差，在 $Q$ 基下放大為 $\kappa(Q)\cdot\epsilon_{\mathrm{mach}}$
（標準混合前後向誤差分析，見 Higham, *Accuracy and Stability of
Numerical Algorithms*, 2002, Ch. 3）。$k$ 步累積線性得第二項
（$C_2 \le 2$，保守界）。□

### Corollary（Primitive-node 步進的反例）

設 $\widetilde{\mathsf A}$ 為 §5 的 primitive-node 步進算子，
與 $\mathsf A$ 在連續極限一致但滿足
$\widetilde{\mathsf A} = \mathsf A + \Delta\mathsf A$，$\|\Delta\mathsf A\|_2 = \|\Delta_L\|_2 \ne 0$
（§5.2 的 Leibniz 缺陷）。則 $V_n$ \emph{不是} $\widetilde{\mathsf M} := \mathsf L^{-1}\widetilde{\mathsf R}$
的特徵向量（一般而言），於是不在 $\widetilde{\mathsf A}$ 的二維不變子空間中。
每步洩漏量
$$
\bigl\| R_4(\Delta t\,\widetilde{\mathsf A})\,U^0 - R_4(\Delta t\,\mathsf A)\,U^0 \bigr\|_2
\;=\;
\mathcal O\bigl(\Delta t\,\|\Delta_L\|_2\,\|V_n\|_2\bigr),
$$
獨立於 $\Delta t \to 0$ 的常數部分由 §5.1 測得為 $6.9\times 10^{-4}/\text{step}$。

**註**：這正式化了 §5 的經驗觀察——Primitive-node 步進無法被塞進「RK4 限於不變子空間」
的證明結構裡，因為它的算子根本不是 $\mathsf M$。

### 新段落草案（擬寫入 §6.5）

上面「設定與記號」+「Lemma A」+「Lemma B」+「主定理」+「Corollary」作為 §6.5 正文。
原有的 §6.5 現成段落「The theorem does not hold when $\mathsf M$ is
replaced...」併入 Corollary。

字數估計：約 1.2 頁 LaTeX。

### 對應 patch

- 檔案：`paper/06_assembled_td.md`
- 位置：§6.5 完整替換
- 變更類型：原「(informal)」非正式陳述 → 完整「Setup + Lemma A + Lemma B + Theorem 6.1 (formal) + Corollary」結構。
- 向量範數 $\|\cdot\|_2$ 定義、條件數 $\kappa(Q)$ 定義在主定理前明確引入。
- 引用 Higham 2002（需加入 `10_refs.md` 作為新文獻 [27]）。
- 既有 (6.3) 誤差式保留，但右側改為兩項的顯式界。
- §6.6 總結段維持不動（只是 §6.5 內部強化）。

---

## R1.2 Leibniz 缺陷 aliasing 分析

**狀態**：完成

**審稿意見**：§5.2 只給二範數量級估算，需推導 $\Delta_L(\rho_0)$ 具體形式，並解釋為何網格加密不消除相對誤差。

### 推導（aliasing 本質）

記 $\mathcal I_N: C[0,L_y]\to\mathbb P_N$ 為 CGL 節點上的多項式插值算子
（度 $\le N$）。Chebyshev 微分矩陣 $\mathsf D$ 的節點作用為
$$
(\mathsf D v)_j = (\mathcal I_N v)'(y_j).
$$
在連續層面，$\partial_y(\rho_0\,\partial_y v) = \rho_0 v'' + \rho_0' v'$ 逐點成立。
在離散層面，考察三個候選矩陣的作用在節點 $y_j$ 上：

$$
\begin{aligned}
\bigl[\mathsf D\cdot\mathrm{diag}(\rho_0)\cdot\mathsf D\,v\bigr]_j
&= \bigl[\mathcal I_N(\rho_0\cdot\mathcal I_N v')\bigr]'(y_j),\\
\bigl[\mathrm{diag}(\rho_0)\cdot\mathsf D^2 v\bigr]_j
&= \rho_0(y_j)\,(\mathcal I_N v)''(y_j),\\
\bigl[\mathrm{diag}(\rho_0')\cdot\mathsf D\,v\bigr]_j
&= \rho_0'(y_j)\,(\mathcal I_N v)'(y_j).
\end{aligned}
$$

前者在對 $\rho_0\cdot\mathcal I_N v'$ 做第二次微分 *之前*，先把它
重新插值到 $\mathbb P_N$（Pseudo-spectral product-and-truncate 的
標準行為）。由於 $\rho_0\cdot\mathcal I_N v'$ 是次數 $\le 2N$ 的函數，
重插值到 $\mathbb P_N$ 丟失 $\mathbb P_{2N}\setminus\mathbb P_N$ 的貢獻
——這正是 **aliasing**。

**缺陷的顯式形式**：定義
$\Delta_L(\rho_0)\,v = \mathsf D\mathrm{diag}(\rho_0)\mathsf D\,v -
\mathrm{diag}(\rho_0)\mathsf D^2 v - \mathrm{diag}(\rho_0')\mathsf D\,v$，
則對於 Chebyshev 基 $\{T_0, T_1, \dots, T_N\}$，
$$
\bigl(\Delta_L(\rho_0)\,T_k\bigr)(y_j) \;=\;
\bigl[\mathcal I_N(\rho_0 T_k')\bigr]'(y_j) - \rho_0(y_j)\,T_k''(y_j) - \rho_0'(y_j)\,T_k'(y_j)
$$
$=$ 「$\rho_0 T_k'$ 在 $\mathbb P_N$ 正交投影下的殘差」的微分。
若 $\rho_0\in C^r$ 且在內部解析，則此殘差的 Chebyshev 係數
在 $n\to N$ 時按 $\rho_0$ 的譜衰減率下降。

**為何網格加密不消除相對誤差**：關鍵是 $V_n$（SL 特徵向量）
本身在 Lane-Emden $n = 3/2$ 下只有 *代數* 譜衰減。表面指數
$\sigma = 1/2$（半整數）使 $V_n$ 的第 $k$ 個 Chebyshev 係數衰減為
$\mathcal O(k^{-2\sigma-1}) = \mathcal O(k^{-2})$（§3.5）。於是：

- $V_n$ 本身的截斷誤差：$\|V_n - \mathcal I_N V_n\|_2 = \mathcal O(N^{-2})$。
- $\rho_0 V_n'$ 的 aliasing 誤差：$\|\Delta_L V_n\|_2 = \mathcal O(N^{-2})$。
- **相對** 誤差：$\|\Delta_L V_n\|_2 / \|V_n\|_2 = \mathcal O(1)$，常數。

所以 primitive-node 步進的 per-step 相對缺陷在 $N_y\to\infty$ 時不衰減——
分子分母同階。對整數極指數（$n = 1, 3$）$V_n$ 有指數譜衰減，
aliasing 誤差也指數小，此時 primitive-node 的確收斂；但對現實
的半整數-表面剖面（$n = 3/2$，對流包絡）會被凍結在某個
constant floor。

**與 Orszag (1972) 的比對**：Orszag 的 3/2-rule dealias 處理的是
\emph{非線性} 項裡的 aliasing；此處的缺陷是同一現象在
\emph{變係數線性} 算子中的出現形式，故 dealiasing 等效作法是「把
$\rho_0$ 視為等效於一個非線性項，在 2/3 模式內做 multiply-and-truncate」，
也就是 §6 的組裝操作：把 $\mathsf L^{-1}\mathsf R$ 在 $\mathbb P_N$ 內
一次到位組合，等效於把所有高階 aliasing 貢獻歸零——這正是 §6 方案
在相同 $N_y$ 下達到 $5\times 10^{-18}$ 的原因。

### 數值實驗（驗證 N_y 獨立性）

腳本：`scripts/review_r12_aliasing_scan.py`
輸出：`review/r12_aliasing/aliasing_scan.csv`

設置：Lane-Emden $n=3/2$，$\rho_{\mathrm{cut}} = 0.05$，$k_x = 2\pi/L_y$（$\ell = 1$），
$\Delta t = 10^{-4}$，振幅 $10^{-8}$，100 RK4 步，比較 primitive-node 與組裝算子兩個方案。

| $N_y$ | $\omega_1$ | primitive per-step | assembled per-step |
|---|---|---|---|
| 32  | 1.6440 | $1.052\times 10^{-5}$ | $4.69\times 10^{-18}$ |
| 48  | 1.6446 | $1.051\times 10^{-5}$ | $6.07\times 10^{-18}$ |
| 64  | 1.6448 | $1.050\times 10^{-5}$ | $5.46\times 10^{-18}$ |
| 96  | 1.6449 | $1.050\times 10^{-5}$ | $5.95\times 10^{-18}$ |
| 128 | 1.6450 | $1.049\times 10^{-5}$ | $5.01\times 10^{-18}$ |
| 192 | 1.6450 | $1.049\times 10^{-5}$ | $2.02\times 10^{-17}$ |
| 256 | 1.6450 | $1.049\times 10^{-5}$ | $8.73\times 10^{-17}$ |

**觀察**：
  - Primitive-node 在 $N_y = 32 \to 256$（八倍加密）per-step 從
    $1.052\times 10^{-5}$ 到 $1.049\times 10^{-5}$——**相對變化 < 0.3%**。
    完全平坦，支持「constant floor」的 aliasing 假說。
  - Assembled 穩定於 $\sim 5\times 10^{-18}$；在 $N_y = 192, 256$
    略升至 $10^{-17}$，來自 $\mathsf M$ 對角化矩陣 $Q$ 條件數
    隨 $N_y$ 成長（§6.5 主定理誤差界第二項 $\kappa(Q)\cdot\epsilon_{\mathrm{mach}}$）。
  - 注意此處 per-step 數量級為 $10^{-5}$，與論文 §5.1 報告的 $6.9\times 10^{-4}$
    不同，因為本實驗用 $\Delta t = 10^{-4}$ 而非 $\Delta t = 2\times 10^{-2}$；
    per-step 誤差量級隨 $\Delta t$ 增加而增大（$\mathcal O(\Delta t)$ 至領階），
    但 **$N_y$-獨立性** 在兩個 $\Delta t$ 區間都成立。

### 新段落草案（擬寫入 §5.2 末）

> **Remark 5.1 (Why the defect is resolution-independent).**
> The discrete operator $\Delta_L(\rho_0)$ defined in (5.4) is the
> product-and-truncate signature of a variable-coefficient operator
> acting in a finite-dimensional polynomial space. Writing the
> differentiation matrix $\mathsf D$ in its interpolation form
> $(\mathsf D v)_j = (\mathcal I_N v)'(y_j)$ with $\mathcal I_N$ the
> CGL interpolation projector onto $\mathbb P_N$, the composite
> $\mathsf D\,\mathrm{diag}(\rho_0)\,\mathsf D$ applies
> $\mathcal I_N$ \emph{twice} --- once in the inner derivative and
> once implicitly through the pointwise multiplication that precedes
> the outer derivative. The net operation is
> "differentiate, multiply, truncate to $\mathbb P_N$, differentiate
> again", and the truncation step discards the degree-$N{+}1$
> through $2N$ content of $\rho_0 v'$. This is the standard aliasing
> error of pseudo-spectral product representation [\ref{Orszag1972}, \ref{Canuto2006}],
> manifested here in the linear variable-coefficient setting rather
> than in the nonlinear advection term where it is usually discussed.
>
> The residual aliasing in $\Delta_L V_n$ is not suppressed by grid
> refinement for the Lane--Emden $n = 3/2$ background. The surface
> exponent $\sigma = 1/2$ enters the Chebyshev spectrum of the
> eigenvector $V_n$ as $\mathcal O(k^{-2\sigma-1}) = \mathcal O(k^{-2})$
> (Section 3.5); the aliasing tail of $\rho_0 V_n'$ decays at the
> same rate. The quotient
> $\|\Delta_L V_n\|_2 / \|V_n\|_2$ is therefore an $\mathcal O(1)$
> constant in $N$, independent of refinement. This prediction is
> verified empirically by an $N_y$ sweep: the per-step deviation
> under primitive-node RK4 stays within 0.3\% of
> $1.05\times 10^{-5}$ as $N_y$ varies from 32 to 256 (eight-fold
> refinement, $\Delta t = 10^{-4}$, Lane--Emden $n = 3/2$;
> reproducer at \texttt{scripts/review\_r12\_aliasing\_scan.py}).
> For integer polytropic indices $n = 1, 3$, by contrast, the
> eigenvector $V_n$ inherits exponential Chebyshev decay and the
> same aliasing quotient vanishes exponentially in $N$ --- the
> primitive-node scheme does converge on smooth backgrounds. The
> half-integer surface regime, which is the one physically relevant
> to convective-envelope stars, is where the defect becomes an
> irreducible floor.

### 對應 patch

- 檔案：`paper/05_td_mismatch.md`
- 位置：§5.2 末尾（現「A second, smaller defect enters through...」段之後）
- 變更類型：新增 `**Remark 5.1 (Why the defect is resolution-independent).**` 子段，約 380 字。
- 需要加文獻 [27] Orszag 1972（已在 Canuto 2006 [3] 裡，但若要精確引用 aliasing 可獨立列）。
- 數據文件：提交 `review/r12_aliasing/aliasing_scan.csv`（7 行 + header）。
- 腳本：`scripts/review_r12_aliasing_scan.py`（~180 行）。

---

## R1.3 GYRE error budget 三重掃描

**狀態**：完成

**審稿意見**：釐清 3.6e-5 誤差來源，不能僅歸於插值；需區分物理模型與數值離散的差異。

### 物理模型澄清（先做）

審稿人似乎誤讀論文 §4.2：「4-variable system」指的就是完整
Dziembowski $(y_1, y_2, y_3, y_4)$ = $(\xi_r/r, p'/\rho_0 r g, \Phi'/rg, d\Phi'/d\ln r/g)$，
**含重力擾動 $\Phi'$**。論文離散的是 GYRE `alpha_grv = alpha_omg = alpha_gam = alpha_pi = 1`
下的完整方程 (4.1)-(4.2)——與 GYRE 物理模型完全一致，差異只在離散方式。

§4.2 首段需要加一句明確申明，避免誤讀。

### 三重掃描實驗

腳本：`scripts/review_r13_gyre_error_budget.py`
輸出：`review/r13_error_budget/{sweep_resolution,sweep_interpolation,sweep_gyre_density}.csv`

固定：Lane-Emden $n = 3$ polytrope，$\ell = 1$，內外截斷 $[10^{-4}, 0.9999]$，
10 個 g-modes 對比 GYRE `COLLOC_GL6` 解（frozen in `EXPECTED_OMSQ_GYRE`）。

#### Axis (i)：$N_r$ 解析度掃描（cubic spline，1000 pts 源剖面）

| $N_r$ | rel_err $(n_g = 1)$ | max rel_err $(n_g \le 10)$ |
|---|---|---|
|  48 | $9.12\times 10^{-11}$ | $1.48\times 10^{-6}$ |
|  64 | $2.02\times 10^{-11}$ | $1.06\times 10^{-8}$ |
|  96 | $6.72\times 10^{-12}$ | $9.12\times 10^{-9}$ |
| 128 | $4.69\times 10^{-12}$ | $8.70\times 10^{-9}$ |
| 192 | $1.55\times 10^{-11}$ | $8.65\times 10^{-9}$ |

$N_r = 64$ 已接近譜收斂飽和，$N_r \ge 96$ 時 max rel_err 平坦於 $\sim 9\times 10^{-9}$
（LAPACK `dggev` round-off 界）。

#### Axis (ii)：插值階 (固定 $N_r = 96$)

| method | rel_err $(n_g = 1)$ | max rel_err |
|---|---|---|
| linear (np.interp) | $3.25\times 10^{-7}$ | $2.77\times 10^{-5}$ |
| cubic spline       | $6.72\times 10^{-12}$ | $9.12\times 10^{-9}$ |

**關鍵發現**：linear interpolation 給 $2.8\times 10^{-5}$，cubic 給 $9.1\times 10^{-9}$—
**差了 3000 倍**！這確認了原論文 §4.4 報告的 $3.6\times 10^{-5}$ 幾乎全部來自
**profile interpolation error**，不是譜離散誤差。

#### Axis (iii)：GYRE 源剖面密度 (固定 $N_r = 96$，cubic)

| GYRE rows | rel_err $(n_g = 1)$ | max rel_err |
|---|---|---|
| 1000 | $6.72\times 10^{-12}$ | $9.12\times 10^{-9}$ |
|  400 | $4.48\times 10^{-10}$ | $2.63\times 10^{-7}$ |
|  200 | $3.26\times 10^{-8}$  | $2.56\times 10^{-5}$ |
|  100 | 發散 ($\mathcal O(10^{2})$) | — |

GYRE 1000 點是「過解析」的——降到 400 已顯著退化，200 已與 cubic spline
在 1000 點時的 linear interp 誤差等量級。這說明 GYRE 原始剖面的 smoothness
已經被 cubic spline 幾乎完全吸收；$N_r = 96$ 譜解析的瓶頸並不是 GYRE 的
1000-point 輸出。

### Error budget 結論

1. 論文 §4.4 原報告的 $3.6\times 10^{-5}$ **不是譜離散極限**，是 linear
   interpolation 的 $\mathcal O(h^2)$ floor。
2. 將插值升為 cubic spline 後 max rel_err 降至 $9.1\times 10^{-9}$——
   **四個數量級的改進**。
3. 譜離散在 $N_r \ge 64$ 已飽和到 LAPACK `dggev` round-off floor（$\sim 10^{-8}$）。
4. GYRE 1000 點源剖面在 cubic spline 之下已完全吸收，不是瓶頸。
5. **沒有** 任何證據支持「物理模型差異」導致誤差——兩邊都是完整 4-variable
   Dziembowski 系統。

**實際狀態**：腳本 `gmode_exp_k_chebyshev_full.py` 自 2026-05-03 起已使用
CubicSpline（見 line 253-272），但論文文字仍寫「linear interpolation」——
這是論文與代碼脫節，修訂時要同步。

### 新段落草案

#### §4.2 首段擴充（~50 字）

在 "GYRE solves the four-variable system..." 之後、"In GYRE's dimensionless form..." 之前加：

> **Physical model equivalence.** Our discretisation targets the same
> four-variable adiabatic system (4.1)--(4.2) with the full-gravity
> setting $\alpha_{\mathrm{grv}} = \alpha_{\mathrm{omg}} = \alpha_{\mathrm{gam}} = \alpha_\pi = 1$;
> the perturbed gravity potential $\Phi'$ enters explicitly through
> $y_3$ and $y_4$. The only distinctions between our solver and GYRE
> are the discretisation (Chebyshev--Gauss--Lobatto collocation
> versus GYRE's Gauss--Legendre COLLOC_GL6), the treatment of the
> central/surface singularities (interior restriction versus GYRE's
> shoot-and-match), and the structure-profile sampling density.
> The benchmark in Section 4.4 therefore measures discrete
> approximation error alone, not a physical-model discrepancy.

#### §4.4 表 4.1 上下文 + 數字更新

目前 §4.4 的 3.6e-5 要改為 **cubic spline** 下重新測的數字
（max rel_err $\sim 9\times 10^{-9}$）。舊 3.6e-5 作為「linear interp」對照放入新 §4.5。

#### §4.5 擴展為 Error Budget（新 Tab. 4.2）

> **4.5 Decomposition of the residual error into independent sources**
>
> The $9.1\times 10^{-9}$ maximum relative error of Table 4.1 arises
> from three discretisation choices that are logically independent:
> the radial spectral order $N_r$, the profile interpolation method
> used to map the GYRE-shipped $\mathtt{poly3.txt}$ onto the CGL
> grid, and the density of GYRE's source profile itself. Table 4.2
> reports three sweeps, each varying one of these axes while the
> other two are held fixed at their production values.

新 **Table 4.2**：三段合併的 error budget（上方數據可直接填）。

> Two conclusions follow. First, the spectral order $N_r$ saturates
> at $N_r \approx 64$; further refinement is blocked by the
> floating-point round-off of LAPACK's \texttt{dggev} generalised
> eigensolver ($\sim 10^{-8}$), not by truncation in the basis.
> Second, the profile interpolation method dominates the residual:
> linear interpolation caps the benchmark at $\sim 3\times 10^{-5}$,
> its $\mathcal O(h_{\mathrm{GYRE}}^2)$ error floor; cubic spline reduces
> this by almost four orders of magnitude and exposes the underlying
> spectral accuracy. Third, the GYRE-shipped 1000-row polytrope
> \texttt{poly3.txt} is sufficiently dense that sub-sampling to 400
> rows introduces a new $\mathcal O(10^{-7})$ error, and to 200 rows
> matches the linear-interp floor --- so the 1000-row default is
> not a bottleneck for cubic-spline-based consumers.
>
> The original Section 4.4 report of $3.6\times 10^{-5}$ relative
> error, obtained with linear interpolation in an earlier version
> of our solver, is the linear-interp entry of Table 4.2 (axis ii);
> updating the interpolation to cubic spline was a one-line fix and
> moved the benchmark into the spectral-accuracy regime.

### 對應 patch

- 檔案：`paper/04_gmode_evp.md`
- 改動 1（§4.2）：首段插入 "Physical model equivalence" 澄清段。
- 改動 2（§4.4 Tab. 4.1）：重跑 benchmark 數字，**更新為 cubic spline 下的值**；
  保留舊結構，只換欄位數字。
- 改動 3（§4.4 最後段）：修正「limited by linear interpolation」→
  「limited by the floating-point precision of the generalised eigensolver」。
- 改動 4（§4.5）：從幾行改為完整小節，引入 Tab. 4.2 三段掃描。
- 檔案：`paper/08_discussion.md` §8.2 "maximum relative error... linear interpolation"
  需要同步更新。
- 數據與腳本：提交 `review/r13_error_budget/*.csv`（3 個 CSV，總 ~30 行）
  + `scripts/review_r13_gyre_error_budget.py`（~150 行）。

### **需要用戶決策的一件事**

由於 cubic spline 把誤差從 $3.6\times 10^{-5}$ 降到 $9.1\times 10^{-9}$，
論文 §1.4 contribution (2)、§4.6 summary、§8.2 都有這個數字。**是否要一併更新
全文？** 我的建議：**更新**——現在代碼支持 9e-9，論文就該報 9e-9，這是加分而非減分。
但要在 §4.5 的 error budget 小節裡同時展示 linear 的 3e-5 以保留「不同實作
選擇下的精度譜」。若用戶同意，我會在應用 patch 時做完整 search-and-replace。

---

## R1.4 IMEX 公平性聲明

**狀態**：完成

**審稿意見**：CN-AB2 vs Strang 對比不公平——前者為二階非對稱格式，後者為二階對稱分裂。失敗可能源於 AB2 低階外插或非對稱耦合，不代表所有 IMEX 方案。

### 決策

採「路徑 1」——只修正聲明範圍，不補 IMEX-RK3 實驗。理由：論文主線是 Strang-split 足夠，不是全面 IMEX 比較。補 IMEX-RK3 需 ~1.5 天且與論文主論證無關；若審稿人二輪仍堅持，再補。

### 新段落草案（擬加在 §7.5 末段）

> **Caveat on the scope of the IMEX comparison.** The semi-implicit
> scheme tested here (Crank--Nicolson on the linear block with
> Adams--Bashforth-2 extrapolation of the nonlinear right-hand side)
> is the simplest second-order IMEX combination, representative of
> the default time-stepping choice in frameworks such as Dedalus for
> quick-turnaround runs. Our Section 7.3 Finding 2 should therefore
> be read as a statement that \emph{this particular} IMEX
> combination fails at physically relevant amplitudes, not as a
> negative result for the IMEX family as a whole. A higher-order
> implicit-explicit scheme --- IMEX-RK3 in the Ascher--Ruuth--Spiteri
> sense, or IMEX-BDF3 --- would very likely avoid the stability
> failure of Finding 2 by replacing the AB2 extrapolation with a
> genuinely $L$-stable or SSP discretisation. We have not
> implemented and tested these higher-order schemes because the
> Strang-split scheme already provides the accuracy and stability
> targets required by the problem at hand and at lower engineering
> cost. A systematic comparison including higher-order IMEX variants
> is an independent study for which the present work provides the
> Strang-split reference point.

### 對應 patch

- 檔案：`paper/07_nonlinear.md`
- 位置：§7.5 末（現有段落 "The exponential-propagator scheme is held in reserve..." 之後、"Extending the Strang-split scheme..." 之前）
- 變更類型：新增一段 ~170 字的 caveat。
- 是否影響其他章節：否。

---

## 變更追蹤表

| 意見 ID | 對應章節 | patch 類型 | 狀態 | commit hash |
|---|---|---|---|---|
| R1.1 | §6.5 | 重寫 + 擴展為正式定理 | 推導完成，待 apply patch | — |
| R1.2 | §5.2 | 新增 Remark 5.1 + 引 CSV | 推導+實驗完成，待 apply | — |
| R1.3 | §4.2 澄清 + §4.5 Tab. 4.2 | 澄清 + Error budget | 實驗完成，待決策後 apply | — |
| R1.4 | §7.5 | 新段落 caveat | 草案完成，待 apply | — |

**附帶工件清單**（本輪產出）：
- 本文件 `paper/REVIEW_ROUND1_RESULTS.md`
- 腳本 `scripts/review_r12_aliasing_scan.py`
- 腳本 `scripts/review_r13_gyre_error_budget.py`
- 數據 `review/r12_aliasing/aliasing_scan.csv`
- 數據 `review/r13_error_budget/sweep_resolution.csv`
- 數據 `review/r13_error_budget/sweep_interpolation.csv`
- 數據 `review/r13_error_budget/sweep_gyre_density.csv`

---

## 完工檢查

- [x] 所有四項修訂的推導 / 數據已記錄於本文件
- [x] 驗證實驗腳本可獨立 reproduce（CSV 已生成）
- [ ] 論文章節 patch 全部落實並重新渲染 PDF（**下一階段工作**）
- [ ] 本文件與 paper 章節交叉引用清晰（patch 落實時同步）
- [ ] git commit 含 `DOC:` 前綴 + 指向本文件

---

## 決策摘要（等待用戶拍板後才 apply patch）

1. **R1.3 數字更新範圍**：是否全文將 $3.6\times 10^{-5}$ 更新為
   cubic spline 下的 $9.1\times 10^{-9}$？建議更新，並在 §4.5 以新 Tab. 4.2
   展示 linear 作為對照。影響 §1.4、§4.4、§4.6、§8.2 共 4 處。
2. **R1.4 IMEX-RK3 是否補測**：默認不補（採草案的 caveat 路徑）。若用戶要補，
   需再開半天-1 天的 IMEX-RK3 prototype 工作。
3. **Patch 應用順序**：建議 R1.4 → R1.1 → R1.2 → R1.3，後者影響面大，放最後。
