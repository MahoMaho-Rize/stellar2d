---
title: |
  Spectral Solver Design — Liouville / Sturm--Liouville / Chebyshev Methods
  for the Stratified Pressure Equation
author: |
  Kiriko, Tsinghua University
date: 2026-05-03 (合併版, 綜合 2026-05-01..03 之四份前期文檔)
geometry: margin=1in
fontsize: 11pt
mainfont: "Times New Roman"
header-includes: |
  \usepackage{amsmath,amssymb,amsthm}
  \usepackage{bm}
  \usepackage{booktabs}
  \newtheorem{theorem}{Theorem}
  \newtheorem{proposition}{Proposition}
  \newtheorem{remark}{Remark}
  \newcommand{\dd}{\mathrm{d}}
  \newcommand{\pp}{\partial}
  \newcommand{\rhob}{\rho_{0}}
  \newcommand{\phat}{\hat{p}}
  \newcommand{\pihat}{\hat{\pi}}
  \newcommand{\fhat}{\hat{f}}
---

# 0. About this document

本檔是 stellar2d 項目 2026-05-01..03 期間四份前期技術文檔合併、
更新、並補入 Phase 0 ext+ 結論後的整合版. 原始四份文檔
(`anelastic_SL_spectral_design.md`, `liouville_SL_spectral_derivation.md`,
`reduced_pressure_liouville.md`, `liouville_singularity_causality.md`)
在 repo 中保留作為推理軌跡.

**與正式技術報告的關係**: 正式英文報告
`docs/spectral_stratified_poisson_report_2026-05-03.md` 是投稿/引用級文檔,
18 頁,署名 Kiriko / Tsinghua, 綜合全部定量證據. 本檔是**倉庫內部的
完整設計記錄**, 保留中英文混合, 強調數學推導過程與工程考慮, 供後續
開發者 onboarding 使用.

**三份正式姊妹文檔**:
- `docs/spectral_stratified_poisson_report_2026-05-03.md` — 正式英文技術報告
- `docs/spectral_experiments.md` — 實驗紀錄合併檔 (Phase 0 / reduced-pressure / g-mode / polytropic index)
- `docs/singular_basis_survey_2026-05-02.md` — GYRE / Dedalus 調查

---


# Part I  Motivation and Problem Setup

## 1. Motivation

當前 `pseudo_spectral` 求解器解的是均勻密度 2D 不可壓 NS
($\rho=\mathrm{const}$, $\nabla\cdot\bm{u}=0$), 雙週期域, 渦度-流函數形式.
恆星對流需要分層密度 $\rhob(y)$, 對應的壓力 Poisson 方程變成
**變係數橢圓方程**, Fourier 模不再是本徵函數, 標準偽譜法
$\mathcal{O}(N\log N)$ 求解失效.

本文檔整合兩條技術路線:

1. **Liouville-SL 基底 (Part II-III)**: 用 $\hat p = \sqrt{\rhob}\,q$ 代換
   把變係數 Poisson 還原成 Schrödinger 形式, 其本徵函數對所有 $k_x$
   通用, Poisson 求解降為 "cuFFT + GEMM + 逐點除法 + GEMM + cuFFT"
   管線.
2. **直接 Chebyshev collocation (Part IV)**: Phase 0 ext+ 結果顯示, 對
   Eddington 標準模型 ($n=3$, $\sigma=3$), raw Chebyshev 直接給出譜收斂
   到機器精度, 不需要 SL 基底. **這是 Phase 1 的主路線**.

兩條路線使用同一套 Chebyshev 網格基礎設施, 差別只在算子組裝方式.
SL 路線保留為 Phase 2+ 可選優化後端.


## 2. Problem formulation

### 2.1 變密度不可壓流的壓力方程

變密度不可壓動量方程除以 $\rhob$ 取散度, 得到

$$\nabla \cdot \left(\frac{1}{\rhob(y)} \nabla p\right) = f(\bm{x})
\tag{2.1}$$

其中 $\rhob(y)$ 是僅依賴 $y$ 的背景分層密度. 對 anelastic 系統
$\nabla\cdot(\rhob\bm{u})=0$, 同一算子在類似投影後出現; 數學結構一致.

### 2.2 Fourier-in-$x$ 分離

$x$ 方向仍為周期均勻, 做 Fourier 展開
$p(x,y) = \sum_{k_x} \phat(k_x,y)\,e^{ik_x x}$, 每個 $k_x$ 得到 1D 變係數 ODE:

$$\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob(y)}\,\frac{\dd \phat}{\dd y}\right] - \frac{k_x^{2}}{\rhob(y)}\,\phat = \fhat(k_x,y).
\tag{2.2}$$

**問題**: $k_x^2$ 項前面的 $1/\rhob(y)$ 係數使得不同 SL 模式耦合. 直接
展開成 $\tfrac{\dd}{\dd y}[(1/\rhob)\tfrac{\dd}{\dd y}]$ 的本徵函數,
$k_x^2$ 項會引入稠密耦合矩陣.

這個問題的兩種解法正是本文 Part II-III (Liouville-SL) 與 Part IV
(Chebyshev collocation) 的起點.


# Part II  Liouville Normal-Form Reduction

## 3. The Liouville substitution

### 3.1 變數代換

令
$$\boxed{\phat(y) = \sqrt{\rhob(y)}\;q(y),}\tag{3.1}$$
計算轉換後的算子.

\begin{proposition}[Liouville reduction]
在代換 (3.1) 下,
$$\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob}\,\frac{\dd \phat}{\dd y}\right] = \frac{1}{\sqrt{\rhob}}\,\bigl[q'' + W(y)\,q\bigr], \tag{3.2}$$
其中 \textbf{Liouville 勢} 為
$$W(y) \;=\; \frac{\rhob''}{2\rhob} - \frac{3(\rhob')^{2}}{4\rhob^{2}}. \tag{3.3}$$
\end{proposition}

\begin{proof}
微分 $\phat = \sqrt{\rhob}\,q$:
\begin{align*}
\phat' &= \frac{\rhob'}{2\sqrt{\rhob}}\,q + \sqrt{\rhob}\,q',\\
\frac{1}{\rhob}\,\phat' &= \frac{\rhob'}{2\,\rhob^{3/2}}\,q + \frac{q'}{\sqrt{\rhob}}.
\end{align*}
微分上式:
\begin{align*}
\frac{\dd}{\dd y}\!\left[\frac{1}{\rhob}\,\phat'\right] &= I_1 + I_2,\\
I_1 &= \left[\frac{\rhob''}{2\,\rhob^{3/2}} - \frac{3(\rhob')^{2}}{4\,\rhob^{5/2}}\right]q + \frac{\rhob'}{2\,\rhob^{3/2}}\,q',\\
I_2 &= -\frac{\rhob'}{2\,\rhob^{3/2}}\,q' + \frac{q''}{\sqrt{\rhob}}.
\end{align*}
$q'$ 項對消, 得
$$I_1 + I_2 = \frac{1}{\sqrt{\rhob}}\left[q'' + W(y)\,q\right]. \qed$$
\end{proof}

### 3.2 化簡後的方程

代入 (2.2) 並乘以 $\sqrt{\rhob}$:

$$\boxed{q'' + W(y)\,q - k_x^{2}\,q = g(y), \qquad g \equiv \sqrt{\rhob}\,\fhat.}\tag{3.4}$$

定義 **Liouville--Schrödinger 算子**
$$\mathcal{T} \equiv \frac{\dd^{2}}{\dd y^{2}} + W(y),\tag{3.5}$$
方程 (3.4) 化為
$$\bigl[\mathcal{T} - k_x^{2}\bigr]\,q = g. \tag{3.6}$$

**關鍵**: $\mathcal{T}$ 不依賴 $k_x$. 水平波數僅作為本徵值的加法平移.
這是使單一預計算譜基底服務所有 $k_x$ 的結構性質.


## 4. Spectral diagonalisation

### 4.1 本徵問題

考慮 Sturm-Liouville (等價地, 時間無關 Schrödinger) 本徵值問題

$$\mathcal{T}\,\psi_n \;=\; \psi_n'' + W(y)\,\psi_n \;=\; -\mu_n\,\psi_n,\qquad n=0,1,2,\ldots \tag{4.1}$$

在適當邊界條件 (週期 / Dirichlet / Neumann) 下, SL 理論保證:

1. 本徵值 $\{\mu_n\}$ 實數且不遞減: $\mu_0 \le \mu_1 \le \mu_2 \le \cdots$
2. 本徵函數 $\{\psi_n\}$ 在 $L^2([0,L_y])$ 構成完備正交基
3. 當 $W\equiv 0$ (均勻密度), $\psi_n$ 退化為標準 Fourier 模,
   $\mu_n = n^{2}\pi^{2}/L_y^{2}$

\begin{theorem}[通用對角化]
$\{\psi_n\}$ 對每個 $k_x$ 同時對角化 $\mathcal{T} - k_x^{2}$:
$$\bigl[\mathcal{T} - k_x^{2}\bigr]\,\psi_n = -(\mu_n + k_x^{2})\,\psi_n. \tag{4.2}$$
\end{theorem}

### 4.2 本徵函數展開解

展開 $q = \sum_n a_n\,\psi_n$, $g = \sum_n g_n\,\psi_n$, 代入 (3.6):

$$\boxed{a_n(k_x) = -\frac{g_n(k_x)}{\mu_n + k_x^{2}}.} \tag{4.3}$$

**無模式耦合**. 變係數 $1/\rhob$ 完全編碼在本徵資料 $\{(\mu_n,\psi_n)\}$
中. 每個 $(k_x, n)$ 對的求解是一次標量除法, 結構上等同於均勻密度
Fourier-Poisson 的 $\phat(\bm{k}) = -\fhat(\bm{k})/|\bm{k}|^{2}$.


## 5. Complete SL-GEMM algorithm

### 5.1 預計算 (一次性, $\rhob$ 變化時重算)

1. 從 $\rhob(y)$ 計算 $W(y)$ 由 (3.3)
2. 解 1D Schrödinger 本徵問題 (4.1), 得 $\{\mu_n, \psi_n\}$
3. 構造變換矩陣 $\Psi_{in} = \psi_n(y_i)$, $\Psi\in\mathbb{R}^{N_y\times N_y}$
4. 預計算 $\sqrt{\rhob(y_i)}$ 向量

### 5.2 每步求解

\begin{center}
\begin{tabular}{clll}\toprule
Step & Operation & Formula & Cost\\\midrule
1 & FFT in $x$          & $f(x,y) \to \fhat(k_x,y)$                   & $\mathcal{O}(N_x N_y \log N_x)$\\
2 & Weight              & $g = \sqrt{\rhob}\cdot\fhat$                 & $\mathcal{O}(N_x N_y)$\\
3 & Fwd SL transform    & $G = \Psi^{\!\top} g$                        & $\mathcal{O}(N_y^{2} N_x)$\\
4 & Pointwise divide    & $Q_n(k_x) = -G_n/({\mu_n+k_x^{2}})$          & $\mathcal{O}(N_x N_y)$\\
5 & Inv SL transform    & $q = \Psi\,Q$                                & $\mathcal{O}(N_y^{2} N_x)$\\
6 & Weight              & $\phat = \sqrt{\rhob}\cdot q$                & $\mathcal{O}(N_x N_y)$\\
7 & IFFT in $x$         & $\phat(k_x,y) \to p(x,y)$                    & $\mathcal{O}(N_x N_y \log N_x)$\\\bottomrule
\end{tabular}
\end{center}

主導成本: 步驟 3, 5 的稠密矩陣乘法, 總複雜度 $\mathcal{O}(N_y^{2}N_x)$
每次 Poisson 解.

### 5.3 複雜度對比

\begin{center}
\begin{tabular}{lccl}\toprule
方法 & $y$ 方向 & 總 ($N\times N$) & GPU 特性\\\midrule
Fourier ($\rhob=\mathrm{const}$) & 除法 & $N^{2}\log N$ & cuFFT (最優)\\
Chebyshev + 三對角 & Thomas & $N^{2}$ & 每列串行\\
\textbf{SL-GEMM}   & \textbf{GEMM} & $\bm{N^{3}}$ & \textbf{cuBLAS batched (tensor cores)}\\
Iterative (CG/MG)  & SpMV $\times k$ & $N^{2}k$ & 收斂依賴\\\bottomrule
\end{tabular}
\end{center}

雖然 SL-GEMM 的漸近複雜度較高, 稠密矩陣乘法在 GPU 上達接近峰值
算術吞吐 (高算術密度, 完全平行, 可用 tensor cores). 對
$N_y \lesssim 4096$, 批次 DGEMM 的 wall-clock 可以與 latency-bound
Thomas 算法競爭甚至更快.


# Part III  Reduced-Pressure Formulation

## 6. Motivation: 改變因變數可以弱化奇異性

回顧: (2.1) 的 $1/\rhob$ 係數是"除以 $\rhob$ 後取散度"這個特定代數
操作引入的. **改變因變數會改變奇異性**.

定義 **約化壓力** $\pi \equiv p/\rhob$ (比焓擾動). 代入
anelastic 動量方程 + 連續性約束, 得
$$\nabla\cdot(\rhob\,\nabla\pi) = \tilde f,\tag{6.1}$$
其中 $\tilde f$ 吸收浮力、對流、$\nabla\rhob$ 項. **關鍵結構性差別**:
(6.1) 的橢圓算子係數是 $\rhob$ (不是 $1/\rhob$), 密度在**分子**.

## 7. Reduced-pressure Liouville 分析

### 7.1 Fourier 分離

$$\frac{\dd}{\dd y}\!\left[\rhob\,\frac{\dd\pihat}{\dd y}\right] - k_x^{2}\,\rhob\,\pihat = \tilde f(k_x,y).\tag{7.1}$$

### 7.2 代換 $\pihat = \rhob^{-1/2}\,q$

直接符號計算 (SymPy 驗證) 得
$$q'' + \widetilde W(y)\,q - k_x^{2}\,q = \tilde g,\tag{7.2}$$
其中
$$\widetilde W(y) = \frac{\rhob''}{2\rhob} - \frac{(\rhob')^{2}}{4\rhob^{2}}.\tag{7.3}$$

**與原始勢 $W$ 的差別**: $(\rhob')^2/\rhob^2$ 項係數為 $-1/4$ 而非 $-3/4$.

### 7.3 表面奇異性比較

對 Lane-Emden $n=3/2$, $\rhob\propto(R-y)^{3/2}$:

\begin{center}
\begin{tabular}{lccc}\toprule
Formulation & 算子 & 表面勢 & $|C|$\\\midrule
Original & $\nabla\cdot(\rhob^{-1}\nabla p)$ & $W \approx -21/(16t^{2})$ & $1.3125$\\
\textbf{Reduced-pressure} & $\nabla\cdot(\rhob\nabla\pi)$ & $\widetilde W \approx +3/(16t^{2})$ & $\bm{0.1875}$\\\bottomrule
\end{tabular}
\end{center}

**奇異強度降低 7 倍**, 且 $C$ 符號從負 (吸引) 翻成正 (排斥).
Frobenius 指數分析:

\begin{center}
\begin{tabular}{lccc}\toprule
Formulation & $C$ & Indicial exponents & 平方可積?\\\midrule
Original & $-21/16$ & $7/4,\;-3/4$ & 只有 $\alpha=7/4$ 分支可積\\
\textbf{Reduced-pressure} & $+3/16$ & $3/4,\;1/4$ & \textbf{兩個分支都可積}\\\bottomrule
\end{tabular}
\end{center>

這是質的改進: 兩個線性無關解都平方可積, SL 理論直接適用.

**Scope 聲明 (2026-05-03 更新)**: 以上分析的 7× 降低只對 $\sigma=3/2$
有意義. 對項目主要情境 $\sigma=3$ (Eddington, $\rhob$ 是多項式),
raw Chebyshev 直接給機器精度, 這個優化沒有 operational 價值. 見 Part IV.


# Part IV  Direct Chebyshev Collocation (Phase 1 主路線)

## 8. 整數 vs 分數 $\sigma$ 的譜收斂斷崖

Phase 0 ext+ (2026-05-03) 的核心發現: 直接對 reduced-pressure 算子
(2.2 或 6.1) 做 Chebyshev collocation, 不做任何代換, 其收斂行為在
表面指數 $\sigma$ 處有**斷崖式**差異.

### 8.1 數值證據

解 manufactured-solution Poisson 問題
$\rhob(r) = (1-r)^{\sigma}$, $\pi_\text{exact} = \sin(2\pi r)$,
$f = [\rhob\pi']' - k_x^{2}\rhob\pi$ (SymPy 符號精確).

\begin{center}
\begin{tabular}{rll}\toprule
$N$ & $\sigma=3$ 誤差 & $\sigma=3/2$ 誤差 (raw)\\\midrule
16  & $8.5\times 10^{-8}$  & $2.7\times 10^{-3}$\\
32  & $\sim 10^{-10}$      & $5.1\times 10^{-4}$\\
64  & $6.7\times 10^{-11}$ & $1.8\times 10^{-4}$\\
128 & $\sim 10^{-9}$ (roundoff) & $5.2\times 10^{-5}$\\
256 & $3.2\times 10^{-9}$  & $1.1\times 10^{-5}$\\\bottomrule
\end{tabular}
\end{center>

- $\sigma=3$: 指數收斂到機器精度 ($\sim 10^{-10}$ at $N=64$), 之後
  roundoff 緩慢爬升 — 這是 Chebyshev 在多項式係數下的理想行為.
- $\sigma=3/2$: 固定 $N^{-2}$ 代數收斂, 任何 $r^\alpha$ 前因子無效.

### 8.2 逼近論解釋 (Trefethen Thm 7.2)

Chebyshev 多項式係數衰減率由係數的解析性決定. 對
$\rhob(r) = (1-r)^{\sigma}$:

- $\sigma\in\mathbb{Z}_{\ge 0}$: $\rhob$ 是**多項式**, Chebyshev 展開
  有限項 ($\sigma+1$ 項), 整個係數被機器精度解析. 譜收斂完全不受
  表面影響.
- $\sigma\notin\mathbb{Z}$: $(1-r)^{\sigma}$ 有**分數代數分支點**,
  Chebyshev 係數只以 $N^{-\sigma-1/2}$ 衰減. $L_N$ 近似 $L$ 的精度
  受限於此, 傳播到解即 $N^{-\sigma-1/2}$ 代數收斂.

### 8.3 對恆星結構的物理意義

Lane-Emden 方程表面行為 $\theta\sim(\xi_1-\xi)$, $\rhob\propto\theta^n$
所以 $\rhob\propto(R-r)^n$. **多方指數 $n$ 就是表面指數 $\sigma$**.

\begin{center}
\begin{tabular}{llcl}\toprule
指數 & 物理情境 & 表面 $\sigma$ & 收斂率\\\midrule
$n=1$ & 外層白矮星 & 1 & $N^{-3/2}$ 代數\\
$n=3/2$ & 對流核 / 完全對流 & 3/2 & $N^{-2}$ 代數\\
$n=2$ & 主序包層近似 & 2 & $N^{-5/2}$ 代數\\
$n=3$ & \textbf{Eddington 輻射模型} & \textbf{3} & \textbf{譜}\\
$n=7/2$ & 巨星氫包層 & 7/2 & $N^{-4}$ 代數\\\bottomrule
\end{tabular}
\end{center>

**$n=3$ Eddington 模型是唯一標準物理多方指數在譜收斂分支**. 這是
"歷史上研究最多的多方模型"與"Chebyshev 收斂性最好的情境"之間的
機緣巧合, 也是本項目主路線選擇 $n=3$ 作為 Phase 1 背景的原因.


## 9. Chebyshev discretisation

### 9.1 Chebyshev-Gauss-Lobatto 網格

$$\xi_j = \cos\frac{j\pi}{N}, \qquad j=0,\ldots,N,\tag{9.1}$$
affine 映射到 $r\in[a,b]$. Trefethen 譜微分矩陣 $D$, 大小 $(N+1)\times(N+1)$.

### 9.2 Reduced-pressure 算子

設 $R_\rho = \operatorname{diag}(\rhob(r_j))$,
$$L_N = D\,R_\rho\,D - k_x^{2}\,R_\rho.\tag{9.2}$$
Dirichlet BC 強 collocation 施加 (節點 0 與 $N$ 用單位行, 右端設邊界值).

### 9.3 對稱性注意

$D$ 在 Euclidean 內積下**非對稱**; 在 Chebyshev-Gauss-Lobatto quadrature
加權內積下對稱. 本徵值問題必須用 `numpy.linalg.eig` (LAPACK `geev`),
不能用 `eigvalsh`. 這是 Phase 0 ext+ 解析解天花板測試 (Test B/C)
的關鍵教訓, 早期用 `eigvalsh` 給出發散/虛假的本徵值.


# Part V  Unified Basis Claim — Critical Examination

## 10. The "g-mode as free by-product" claim revisited

### 10.1 兩個算子的奇點位置不同

- **Poisson 算子**: 奇點在**表面 $r=R$** (因 $\rhob\to 0$).
  Liouville 勢 $\widetilde W\sim \sigma(\sigma-2)/[4(R-r)^2]$.
  SymPy 推導最優前因子:
  $$\pi = (R-r)^{\alpha_\star}\,u, \qquad
    \alpha_\star(\text{Poisson}) = 1 - \sigma/2.\tag{10.1}$$
- **g-mode 算子**: 奇點在**原點 $r=0$** (離心項 $\ell(\ell+1)/r^2$).
  最優前因子:
  $$y_1 \sim r^{\beta_\star},\qquad
    \beta_\star(\text{g-mode}) = \ell + 1.\tag{10.2}$$

$\alpha_\star \ne \beta_\star$: **不存在一組本徵函數同時對角化兩個算子**.

### 10.2 退化後的正確陳述

"Unified SL basis simultaneously diagonalises Poisson and g-mode operators"
的強形式**不成立**. 退化為:

- Chebyshev **網格**可共用 (同一 CGL 節點)
- Dense linear algebra 基礎設施 (cuBLAS GEMM, VRAM 佈局) 共用
- Poisson 對每個 $k_x$ 重用 LU 預分解 (標量平移)
- g-mode 是**同網格的獨立 EVP**, 不是免費副產品

工程效益保留, 數學免費性降級. 這是 Phase 0 ext+ (E5/E6) 的重要結論.


# Part VI  Liouville Singularity — Physical Interpretation

## 11. 因果反轉: QM vs 天體物理

Liouville 勢在 $\rhob\to 0$ 處的 $1/t^2$ 奇異, 數學上與量子力學的
Coulomb/centrifugal barrier 完全相同, 但**因果方向反轉**.

### 11.1 QM 中: 勢是原始物理量

$$V(r) \;\text{(物理輸入)} \;\longrightarrow\; \psi(r)\to 0 \;\text{(數學後果)}$$

Coulomb 勢或 centrifugal barrier 是**根本物理量**, $|\psi|^2$ 在邊界
消失是"因為"勢強迫其消失. Coulomb-Sturmian 基底、Laguerre DVR、
R-matrix matching 等 QM 技術都是為了"不能移除的物理奇點"設計的.

### 11.2 天體物理中: 密度是原始物理量

$$\rhob(y)\to 0 \;\text{(物理輸入)} \;\longrightarrow\; W(y)\to-\infty \;\text{(數學人工產物)}$$

密度剖面 $\rhob(y)$ 由靜水平衡決定 (Lane-Emden, MESA). 恆星表面只是
氣體跑完了 — 沒有"無限勢壁". 原始算子 $\nabla\cdot(\rhob^{-1}\nabla p)$
是表面**退化橢圓** (degenerate elliptic), 平滑失去橢圓性,
不是真奇異. $W$ 的發散**完全由 $\sqrt{\rhob}$ 代換製造**, 是坐標奇點,
不是物理奇點.

### 11.3 對基底設計的啟示

1. **QM 方法可用但過度工程**. Coulomb-Sturmian 為"有真實奇點"設計,
   把它套在"人工奇點"上相當於殺雞用牛刀.
2. **Path A 的局限由此而來**. 試圖用 $r^\alpha$ 吸收 $W$ 的奇異,
   在 QM 中對應於 Coulomb-Sturmian 的 $r^\ell e^{-\alpha r}$,
   那裡是正確方法, 這裡是"反向移動"一個本不該存在的奇點.
3. **最自然的做法是回到原算子**. Reduced-pressure (Part III) 把
   奇異性從 "不可積吸引" 變成 "兩分支可積排斥", 削弱但未消除;
   raw Chebyshev (Part IV) 直接對著係數 $\rhob$ 做多項式逼近,
   對多項式 $\sigma$ 完全無奇異, 這才是"對味"的方法.

**結論**: 相較於 QM 社群為 Coulomb 奇點設計的複雜技術棧, 天體物理
分層 Poisson 問題應該用**最少機械的方法**. 對 $\sigma\in\mathbb{Z}$
這甚至不需要任何特殊處理; 對 $\sigma\notin\mathbb{Z}$ 則 Jacobi 加權
基底 (Dedalus) 是唯一需要的額外機械.


# Part VII  Phase Roadmap

## 12. Current status and forward plan

### 12.1 已完成 (Phase 0 ext+, 2026-05-02..03)

- ✓ Lane-Emden $n=3/2$ 與 $n=3$ 的譜收斂斷崖驗證
- ✓ Chebyshev $N=48$ ($192$ DOF) 對 GYRE full-gravity 4-var 系統 benchmark,
  max_rd $1.5\times 10^{-6}$, 比 FD $N_r=1024$ (4096 DOF) 快 21× DOF /
  精度 350×
- ✓ 三組解析解天花板測試 (manufactured Poisson, 量子諧振子, Dirichlet
  Laplacian) 全部達雙精度機器精度 ($10^{-13}$-$10^{-15}$)
- ✓ Barycentric Lagrange 驗證 "N 係數 ≠ N 像素"
- ✓ Path A/B/C 決策定論: Path A (raw Chebyshev) 對 $n=3$ 足夠

全部實驗證據見 `docs/spectral_experiments.md` 與
`docs/spectral_stratified_poisson_report_2026-05-03.md`.

### 12.2 Phase 1: 2D Fourier-Chebyshev Boussinesq

- **x 方向**: Fourier (週期), 沿用 `pseudo_spectral` 的 cuFFT 基礎設施
- **y 方向**: Chebyshev collocation, GPU 上用 cuBLAS dense GEMM 做 $D^{(2)}$
- **物理**: 2D Boussinesq + buoyancy. Poisson via Chebyshev + dense solve
- **Benchmark**: Rayleigh-Bénard Nu-Ra scaling (對標 Ahlers 等人)
- **背景**: Gaussian 過渡 → Eddington $n=3$ polytrope

設計文檔 `docs/phase1_2d_spectral_design.md` (待寫).

### 12.3 Phase 2: Anelastic 升級

- 從 Boussinesq 升級到 anelastic: $\nabla\cdot(\rhob\bm{u})=0$
- Chebyshev 對變密度 Poisson 直接適配 (raw 或 reduced-pressure)
- **SL-GEMM 後端作為可選優化**, 若 dense solve 成為 GPU 瓶頸才啟用

### 12.4 Phase 3: 線上模式投影 (差異化賣點)

- 同網格獨立 EVP 求 g-mode / p-mode 本徵對
- 瞬時流場投影到模式空間作為 runtime diagnostic
- **這是項目真正的 novelty**: 對流-脈動耦合的 2D 非線性 DNS
  + 線上模式投影

### 12.5 Phase 4+: 球殼擴展

見 `docs/sph_spectral_roadmap.md` (遠期).

### 12.6 論文規劃

- **方法類** (JCP): "Sturm-Liouville spectral methods for stratified
  astrophysical flows — convergence regimes and GPU implementation"
- **應用類** (A&C / ApJS): "GPU anelastic pseudo-spectral with live
  eigenmode projection for convection-pulsation coupling diagnostics"

主推應用類 (真 novelty); 方法類作為支持文獻.


# Appendix A  GPU Implementation Considerations

## A.1 Batched GEMM 形式

步驟 3, 5 可合併為單次矩陣矩陣乘法:
$$\bm{G} = \Psi^{\!\top}\bm{g}, \qquad \bm{q} = \Psi\bm{Q},$$
其中 $\bm{g},\bm{G},\bm{Q},\bm{q}\in\mathbb{R}^{N_y\times N_x}$.
現代 GPU (NVIDIA Ampere/Hopper) FP64 GEMM 超過 1 TFLOP/s, 算術密度
$\mathcal{O}(N)$ flops/byte 保證 $N_y\ge 256$ 時 compute-bound.

## A.2 工作流整合

SL-GEMM Poisson solver 嵌入現有 pseudo-spectral 時間積分器 (例如 IFRK3
+ cuFFT), 把譜空間除法 $\phat = -\fhat/|\bm{k}|^2$ 替換為:

$$\text{cuFFT (R2C in }x\text{)} \to \text{cuBLAS DGEMM} \to \text{逐點除} \to \text{cuBLAS DGEMM} \to \text{cuFFT (C2R in }x\text{)}.$$

所有既有基礎設施 — IFRK3, skew-symmetric 對流, 2/3 dealias, VRAM
frame buffer — 無須修改即可復用.

## A.3 記憶體需求

變換矩陣 $\Psi$ 需 $N_y^{2}$ doubles = 32 MiB at $N_y=2048$. 相比
流場陣列 (100 MiB each at $2048^2$) 與 VRAM frame buffer (~10 GiB) 極小.


# Appendix B  相關方法與文獻定位

## B.1 本方法的三社群空白

| 社群 | 代表代碼 | 徑向求解 | 為何沒做這個 |
|------|---------|---------|-----------|
| 恆星偽譜 | ASH, Rayleigh | 球諧(角向) + Chebyshev/FD(徑向) + 帶狀 | CPU 上 GEMM 慢 |
| GPU 偽譜 | hit3d, spectralDNS | Fourier 全方向 | 只做均勻密度 |
| 變密度 GPU CFD | 工程 LES | FV + multigrid | 不碰譜方法 |

三個社群不會自發走到 "Liouville + SL + GPU GEMM + anelastic" 的交點.

## B.2 SL 理論脈絡

- **Liouville normal form**: Sturm 1836; Liouville 1837; 見 Zettl (2005)
  現代處理
- **SL 本徵展開解橢圓 PDE**: Boyd (2001) 論過理論優雅但缺快速變換
- **Chebyshev 快速收斂論**: Trefethen (2013) Ch. 7 (整數 vs 分數
  指數的漸近係數衰減)
- **Berrut-Trefethen 2004**: barycentric Lagrange, 讓譜表達可以在
  任意細密網格上穩定評估到機器精度

## B.3 與 GYRE / Dedalus 的比較

詳見 `docs/singular_basis_survey_2026-05-02.md`. 本項目的 **novelty
定位最終修正為** "GPU 2D anelastic DNS + 線上模式投影", 不跟 GYRE
(1D 星震成熟基石) 或 Dedalus (通用譜法 PDE 框架) 在 1D 星震 benchmark
上競爭.


# Appendix C  Historical record — 4 份前身文檔的對應關係

| Section in this doc | 前身文檔 | 前身原始章節 |
|--------------------|---------|------------|
| Part I §1-2        | anelastic_SL_spectral_design.md | §1-2 |
| Part II §3-4       | liouville_SL_spectral_derivation.md | §3-4 |
| Part II §5         | liouville_SL_spectral_derivation.md §5 + anelastic_SL_spectral_design.md §5 |
| Part III §6-7      | reduced_pressure_liouville.md 全文 |
| Part IV §8-9       | polytropic_index_spectral_convergence_2026-05-03.md + 正式報告 §3 |
| Part V §10         | 新增 (Phase 0 ext+ E5/E6 結論) |
| Part VI §11        | liouville_singularity_causality.md 全文 |
| Part VII §12       | 新增 (Phase 0 ext+ 後路線圖) |
| Appendix A         | anelastic_SL_spectral_design.md §8 + liouville_SL_spectral_derivation.md §8 |
| Appendix B         | anelastic_SL_spectral_design.md §7 + liouville_SL_spectral_derivation.md §9 |

四份前身文檔全部保留在 repo 作為推理軌跡存檔, 頭部都標註了 Update
區塊指向本文與正式報告. 新工作以本檔 + 正式報告為準.
