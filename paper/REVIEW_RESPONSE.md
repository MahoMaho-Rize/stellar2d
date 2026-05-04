# 審稿意見回應草案 (v1, 2026-05-03)

本文件整理對第一輪審稿意見的逐條回應。分三類：
(A) **已可在現有材料內回應**——需重新論述或補強既有段落；
(B) **需補充數學推導**——目前只有經驗觀察，要補嚴格論證；
(C) **需補充數值實驗**——目前數據不足以支撐，要跑新算例。

對每條意見，先判明類型，再列出**最小可行補丁**（修改哪一節、大致多少字、需不需新圖新表）。
最後匯總為一份可執行的修訂計劃，供下一步開工使用。

---

## 1. 數學分析與理論深度

### 1.1 Theorem 6.1 非正式 → 需正式化

**審稿人建議**：將 Theorem 6.1 發展為正式定理，證明 Galerkin 投影下時間步進運算元與空間離散運算元一致時，為何保證特徵模態保持，而原始節點形式則不能。

**類型**：(B) 需補充數學推導。

**回應策略**：
目前 §6.1 已給了「proof sketch」，關鍵論點（RK4 多項式在 $\mathsf M$ 的每個特徵子空間內作用、RK4 對 $\exp(\mathrm i\omega\Delta t)$ 的 $\Delta t^5$ 逼近）已經到位。缺的是：
  - **引理 A**：$\mathsf A = \begin{bmatrix}0&I\\-\mathsf M&0\end{bmatrix}$ 的譜分解——每個 $\mathsf M$ 特徵對 $(\omega_n^2, V_n)$ 對應一對 $\mathsf A$ 特徵對 $(\pm\mathrm i\omega_n, (V_n, \pm\mathrm i\omega_n V_n))$。
  - **引理 B**：RK4 穩定函數 $R_4(z) = 1 + z + z^2/2 + z^3/6 + z^4/24$ 在純虛軸 $z = \mathrm i\omega\Delta t$ 上滿足 $|R_4| = 1 + \mathcal O((\omega\Delta t)^{10})$（非耗散），且 $\arg R_4 = \omega\Delta t + \mathcal O((\omega\Delta t)^5)$。
  - **主定理**：結合二者，得 $k$ 步後的誤差為 $\|V_{k}-V_n\cos(\omega_n k\Delta t)\| \le C_1 (\omega_n\Delta t)^5 k + C_2\,k\,\epsilon_{\mathrm{mach}}\,\kappa(Q)$，其中 $Q$ 為 $\mathsf M$ 的對角化矩陣。
  - **反例**：primitive-node 步進算子 $\widetilde{\mathsf A}$ 的特徵向量與 $V_n$ 差一個 $\mathcal O(\|\Delta_L\|)$ 擾動；RK4 仍保相位，但 $V_n$ 不是其不變子空間，故每步洩漏 $\|\Delta_L\|\cdot\|V_n\|$。

**補丁位置**：§6.5 替換為 §6.5（新）「Theorem 6.1 (formal) 與其證明」，約 2 頁。三個引理顯式寫出，主定理保留現 (6.3) 式，但常數 $C_1, C_2$ 具體給出界。反例放 §6.5 末作 Corollary。

**不需要新數據**：現有 Tab. 6.1（Python 5e-18）和 Tab. 6.2（CUDA 3e-15）已經是 $C_2\cdot k\cdot\epsilon_{\mathrm{mach}}\cdot\kappa(Q)$ 的實證；可在證明中點出 $\kappa(Q)\approx 10^2$ 時兩者一致。

**工作量估計**：半天推導 + 半天寫作 = 1 天。

---

### 1.2 離散萊布尼茲缺陷 → 需理論化分析

**審稿人建議**：目前 §5.2 只給經驗觀察和 $\|\rho_0'\|_\infty\cdot\mathrm{cond}(\mathsf D)$ 的量級估算。要求：
  (a) 推導 $\Delta_L(\rho_0)$ 的具體形式及其在基函數空間中的作用；
  (b) 闡明為什麼不會因網格加密而消失（從多項式插值和微分角度）。

**類型**：(B) 需補充數學推導。

**回應策略**：

**(a) 具體形式**：$\mathsf D$ 是在 CGL 節點上 *精確多項式微分*——對次數 $\le N$ 的多項式 $p$，$(\mathsf D p)_j = p'(y_j)$ 機器精度成立。但 $\mathsf D\,\mathrm{diag}(\rho_0)\,\mathsf D$ 作用在向量 $v$ 上，首先計算 $(\mathsf D v)_j = \mathcal I_N'(v)(y_j)$（$\mathcal I_N$ 為 $N$ 次插值多項式），然後乘 $\rho_0(y_j)$，再微分——第二次微分作用在 $\rho_0(y)\cdot\mathcal I_N'(v)(y)$ 的插值上，這是 $N$ 次多項式（因為插值已經降到 $N$ 次），而 $\rho_0\cdot v'$ 的真實值是 $N+r$ 次（$r$ 是 $\rho_0$ 的插值階）——所以 $\mathsf D^2$ 前後兩次等效於把 $\rho_0 v'$ *截斷* 到 $N$ 次多項式空間，丟掉 $\mathcal O(\|\rho_0\|_{C^r}\cdot N^{-r})$。

所以 $\Delta_L(\rho_0) v$ 的元素是 $\rho_0\cdot v'$ 去掉其在 Chebyshev 多項式基 $\{T_0,\dots, T_N\}$ 上投影的殘差——這是 **aliasing 誤差**，跟 pseudo-spectral 的 dealiasing 問題同源。

**(b) 為什麼網格加密不消**：RHS $= k_x^2 N^2 \rho_0 V$ 裡 $V$ 是 SL 特徵向量。$V$ 本身的 Chebyshev 譜在 $n\to N$ 時的衰減率 *與背景規則度相同*；當 $n=3/2$ 表面指數為半整數，$V_n$ 的第 $k$ 個 Chebyshev 係數衰減為 $\mathcal O(k^{-2\sigma-1})$，$\sigma = 1/2$ 對應 $k^{-2}$，*代數* 衰減。aliasing 誤差 $\Delta_L V_n$ 的第 $N$ 個 Chebyshev 係數也是 $\mathcal O(N^{-2})$，所以 $\|\Delta_L V_n\| = \mathcal O(N^{-2})$——但此時 $V_n$ 的 truncation 誤差也是 $N^{-2}$，於是 *相對* 誤差 $\|\Delta_L V_n\|/\|V_n\|$ 是 $\mathcal O(1)$ 常數，不隨 $N$ 下降。

這解釋了 §5.1 觀察到的 "primitive-node deviation 不隨 $N_y$ 從 48 改到 256 而變"：分子分母同階。

**補丁位置**：§5.2 末尾新增「**Remark 5.1 (Why the defect is resolution-independent).**」子段，約 400 字 + 一條引用 Orszag (1972) 的 aliasing 分析、Canuto et al. Ch.4.2。

**可選新實驗**：若要加強實證，跑 $N_y = \{48, 64, 96, 128, 192, 256\}$ 掃描 primitive-node deviation，預期曲線 **平坦**（~$6\times 10^{-4}$）而非下降。現有 §5.1 文字已提過 48→256 沒變，加個新圖把這個點坐實。約 1 小時實驗 + 繪圖。

**工作量估計**：推導半天 + 實驗半天 + 寫作半天 = 1.5 天。

---

### 1.3 非線性延伸的分析 → IMEX 失穩需理論解釋

**審稿人建議**：§7 只展示哪種分裂有效，未解釋為什麼。IMEX 失穩的理論根源是數值共振？還是 AB2 絕對不穩定性被 CN 放大？

**類型**：(B) 推導 + (C) 若推導得出可驗證的頻率/振幅邊界，跑掃描。

**回應策略**：
既有分析在 §7.3 Finding 2 已經提了一個量級估算（$\mathcal O((\omega\Delta t)^2)$ AB2 外插誤差），但沒做穩定性域。要補的是：

**線性穩定性域分析**：對 IMEX(CN, AB2) 應用到 $\dot x = -\mathrm i\omega x + f_{\mathrm{nl}}(x)$ 的測試方程，求放大因子 $R(z, w) = (1 + z/2)/(1 - z/2)$（CN 部分）$+ \Delta t\cdot (3 w^n/2 - w^{n-1}/2)$（AB2 部分），其中 $z = -\mathrm i\omega\Delta t$，$w = \partial f_{\mathrm{nl}}/\partial x$。設非線性 Jacobian 的純虛部分（耦合到 $\omega$ 的部分）為 $\mathrm i\alpha\omega$（線性化的模式耦合係數，$\alpha$ 與振幅成比例），則放大矩陣在 $\alpha\to 0$ 時 $|R|=1$，但 $\alpha\ne 0$ 時 AB2 在純虛軸上有絕對不穩定性（經典結果：AB2 穩定域不包含純虛軸除原點外的點）。

這解釋了：
  - Strang/exp.-prop. 方案把 $\omega$ 部分 *完全* 移到 RK4 塊或 exp 塊內部，非線性塊只看到 $f_{\mathrm{nl}}$ 本身；RK4 穩定域包含一段純虛軸，所以穩定。
  - IMEX 方案把 $\omega$ 留在 CN 裡——CN *保* $|R|=1$（中性穩定），但 AB2 又外插了線性-非線性耦合，這部分就命中了 AB2 的不穩定區。

**補丁位置**：§7.3 Finding 2 擴展，從目前 ~150 字增至 ~400 字，加一個 **穩定性圖**（$\alpha\omega\Delta t$ 平面上的放大因子 contour）。

**可選新實驗**：振幅掃描 $a = 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}$ 下每種方案的 $\max|\lambda_k|$，與理論預測的穩定邊界對照。~2 小時。

**工作量估計**：推導一天（要仔細做 $2\times 2$ IMEX 矩陣特徵值分析）+ 實驗半天 + 繪圖半天 = 2 天。

---

## 2. 方法論的完整性

### 2.1 GYRE 比對的對等性

**審稿人建議**：GYRE 用完整四階方程 + 1000 網格，我們用簡化二階方程 + 96 插值網格。3.6e-5 不能僅歸於插值，要論證物理模型差異（重力擾動 $\Phi'$ 忽略與否）。

**類型**：(A) 論述補強 + (C) 對照實驗。

**回應策略**：
**需要先釐清一個事實**：§4.3 裡我們 *已經* 離散了 GYRE 完整四變數 Dziembowski 系統（見 (4.1)--(4.2)，含 $y_3 = \Phi'/rg$, $y_4$），使用 GYRE 的所有五個結構係數 $V_2, A^\star, U, c_1, \Gamma_1$——所以物理模型 *沒有* 忽略 $\Phi'$，僅離散方式與網格不同。這點在目前的 §4 裡寫得不夠清楚，審稿人可能誤讀了「4-variable」。

**補丁 A（必做，~1 段）**：§4.2 首句明確寫「The present work uses the full four-variable formulation including $\Phi'$, identical to GYRE's standard adiabatic non-rotating case ($\alpha_{\mathrm{grv}} = \alpha_{\mathrm{omg}} = 1$).」

**補丁 B（收斂性實驗）**：既然物理模型相同，誤差來源只剩：
  (i) 徑向離散網格（我們 96 CGL vs GYRE 1000 GL + 6 階 COLLOC）；
  (ii) 結構剖面插值（poly3.txt 1000 點線性插值到 96 CGL）；
  (iii) 邊界截斷（$x \in [10^{-4}, 0.9999]$ 避奇點）。

提議做一個 **三重掃描**：
  - 固定 profile interpolation 方式（linear），掃 $N_r \in \{48, 64, 96, 128, 192\}$；
  - 固定 $N_r = 96$，換 profile interpolation 為 cubic spline；
  - 固定兩者，把 GYRE 自己的解析度從 1000 降到 200、100，看 GYRE 解本身對網格的收斂。

這三個掃描能 *分離* 三個誤差源。預期結果：(i) 影響小（指數收斂），(ii) 主導，(iii) 影響 $10^{-7}$ 級。

**補丁位置**：§4.5 現在只有幾行 "$N_r = 48$ 有 spurious modes, $N_r = 96$ 乾淨"——擴展為完整的 §4.5 「Error budget decomposition」，約一頁，含 Tab. 4.2（三列掃描）。

**工作量估計**：實驗 4 小時（三個掃描，每個掃 5--6 個點）+ 寫作半天 = 1.5 天。

---

### 2.2 非線性比較的公平性

**審稿人建議**：Strang/exp.-prop. 都用 Strang 對稱分裂，IMEX 用非對稱 CN-AB2。對比不公平——IMEX 失敗可能是低階外插，不是 CN 本身的問題。

**類型**：(A) 論述補強 + 可選 (C) 公平對比實驗。

**回應策略**：
這是合理的方法論批評。有兩條路：

**路徑 1（最小修補）**：在 §7 結尾加一個 **Caveat** 段，明確寫「本節比較並非對所有 IMEX 方案的終審；我們選 CN-AB2 因其是 Dedalus 等框架的缺省簡單 IMEX，實測表明在本問題上不可用。更高階的 IMEX-BDF3、IMEX-RK3 可能避免本節的失穩模式，但實作複雜度顯著提升；由於 Strang-split 已達目標精度，我們未進一步評估。」——這保留現有實驗，只修正聲明範圍。

**路徑 2（公平對比實驗）**：補一個 IMEX-RK3（Ascher-Ruuth-Spiteri 三階 SSP）prototype，掃相同振幅，看是否穩定。若穩定但慢於 Strang，強化「Strang 勝在工程成本」的結論；若也失穩，強化「高振幅下這類問題本質 stiff」的結論。約 300 行 Python + 半天調試。

**建議採路徑 1 + 補充材料註明 IMEX-RK3 未測**，因為論文主線是 Strang-split 足以解決問題，不是要對 IMEX 做全面比較。若審稿人堅持要路徑 2，再補。

**補丁位置**：§7.5 summary 末尾加一段，約 100 字。

**工作量估計**：路徑 1 = 半小時；路徑 2 = 1.5 天。

---

### 2.3 Python 原型優先方法論的定位

**審稿人建議**：§8.5 更像經驗心得，不是嚴格方法論。建議移至引言或作為「development methodology」討論，並分析適用邊界。

**類型**：(A) 論述補強。

**回應策略**：
同意。提議：
  - §1.3 目前已有該主題的簡短介紹（~30 行）；把 §8.5 完整內容縮到兩段（保留兩個具體 reversal 案例 + 適用邊界），放回 §8.5 但重命名為「**Development methodology: Python prototyping as gatekeeper**」。
  - 在 §1.3 加一句「A full discussion is deferred to §8.5」。
  - §8.5 增加對適用邊界的條列（現在已經有一段，但可以更清晰）：
    - ✅ 適用：小矩陣 per wavenumber，FFT-friendly geometry，bandwidth-bound。
    - ✗ 不適用：communication-bound，大稀疏矩陣求解，與 legacy C++/CUDA API 重度耦合。

**補丁位置**：§8.5 改寫，約半頁。

**工作量估計**：半天寫作。

---

## 3. 技術細節與再現性

### 3.1 SL 求解器的可擴展性

**審稿人建議**：用 LAPACK dgeev 求稠密 SL 特徵問題只適用小 $N_r$。要在 §8.1 討論此瓶頸 + 替代方案。

**類型**：(A) 論述補強。

**回應策略**：
目前 §8.1 沒討論這點。提議加一子段「**Scalability of the SL precomputation.**」：
  - 當前：LAPACK `dgeev` 成本 $\mathcal O(N_y^3)$，一次性，$N_y = 64$ 約 40 ms，$N_y = 128$ 約 300 ms——仍可接受。
  - 瓶頸：$N_y \gtrsim 512$ 時 setup 開銷 > 1 秒；對時間步進長度 $10^3$--$10^4$ 這仍是小頭，但若做 3D 或參數掃描則相關。
  - 替代方案：
    - (a) **快速 SL 變換**：Rokhlin & Tygert、Townsend 等近年做法，$\mathcal O(N \log N)$ 成本；已有開源實作（fastSL.jl）。
    - (b) **Krylov 特徵值 + 位移-反迭代**：只解前 $k$ 個特徵對，$\mathcal O(k N_y^2)$。
    - (c) **Riccati-sweep / Prufer 方法**：一維 SL 可 $\mathcal O(N_y)$ 解所有特徵值。
  - 本文選 `dgeev` 因 $N_y \le 128$ 是目標區間，且 setup 可完全預計算、bit-reproducible；若將來需 $N_y = 512$ 級，Prufer 是首選。

**補丁位置**：§8.1 末尾新子段，約 300 字。

**工作量估計**：半天（含引用查找）。

---

### 3.2 表面截斷的處理

**審稿人建議**：$\rho_{\mathrm{cut}}$ 引入人為反射邊界，需：
  (a) 參數敏感性研究（$\rho_{\mathrm{cut}} \in [0.01, 0.1]$）；
  (b) 邊界條件實現細節。

**類型**：(A) + (C) 敏感性實驗。

**回應策略**：
**(a) 敏感性**：提議掃 $\rho_{\mathrm{cut}} \in \{0.01, 0.02, 0.05, 0.10\}$，計算前 5 個 g-mode 頻率和本徵函數 $L^2$ 變化。既有數據裡應該已有 $\rho_{\mathrm{cut}} = 0.05$ 的結果，補另外三個點即可。約 2 小時實驗。

**(b) 實現細節**：§2.3 提了「$\max(-\rho_0'/\rho_0, 0)$」截斷，但沒說壁上如何處理。實際上（查 `src/gpu/stellar_profile.*`）：
  - 將 CGL 網格 $y \in [0, L_y]$ 映射到 $\xi \in [0, \xi_1]$；
  - 凡 $\rho_0(\xi) < \rho_{\mathrm{cut}}$ 的 $\xi$ 點，被 clip 到 $\rho_{\mathrm{cut}}$；
  - $N^2$ 同樣 clip（取 $\max$ 避免負值）；
  - Dirichlet $v = 0$ 邊界條件施加於 $y = L_y$（CGL 最外網格點），*不* 是 $\rho_0 = \rho_{\mathrm{cut}}$ 處。

**補丁位置**：
  - §2.3 末尾擴充實現細節，約 5 行。
  - §3.5（或新 §3.7）加敏感性表 Tab. 3.1。

**工作量估計**：實驗 2 小時 + 寫作 2 小時 = 半天。

---

### 3.3 程式碼與數據可及性

**審稿人建議**：poly3.txt 插值步驟及程式碼必須清楚提供。

**類型**：(A) 論述補強（GitHub release 已含數據）。

**回應策略**：
GitHub release `gyre-benchmark-2026-05-03` 已包含 poly3.txt、summary h5、CSV、PNG、SHA256SUMS。要在論文裡：
  - §4.4 明確指向具體 release 與檔名；
  - §4.4 點出插值程式碼位置（`src/gpu/stellar_profile.cpp::read_gyre_structure_txt`）；
  - 附錄新增「**Appendix B: Reproducibility checklist**」，表列每個實驗需要的 (腳本, 輸入, 輸出, git commit) 四元組。

**補丁位置**：§4.4 + 新附錄 B。

**工作量估計**：半天。

---

## 4. 寫作與結構

### 4.1 核心章節抬頭

**審稿人建議**：§3（SL 空間）與 §6（組裝矩陣時間步進）是兩個核心貢獻，目前和實驗章節 (§4, §5, §7) 混在一起。

**類型**：(A) 結構調整。

**回應策略**：
兩個選項：

**選項 1（最小改動）**：在 §3 和 §6 的節標題前加「**Core contribution I: ...**」「**Core contribution II: ...**」。讀者立刻看到這是貢獻章節。

**選項 2（重組結構）**：引入兩部分劃分：
  - **Part I: Spatial machinery** — §2, §3, §4
  - **Part II: Temporal machinery** — §5, §6, §7
  - **Part III: Discussion** — §8, §9

**建議選項 2**——結構更清晰，但需要改 ToC 和節編號（章節內文字不動）。

**補丁位置**：目錄 + 每部分首頁加過渡段（約 50 字 × 3）。

**工作量估計**：1 小時。

---

### 4.2 語氣中立化

**審稿人建議**：「the primitive-node time-stepping... cannot reproduce」過於絕對，改為「fails to reproduce... due to a discrete operator inconsistency」。

**類型**：(A) 論述補強。

**回應策略**：
同意。做一次全文 "cannot" / "impossible" / "never" 的掃描，凡是涉及數值方法評斷的都改為「fails to...」、「does not in general...」、「is empirically observed to...」等較學術表述。

**補丁位置**：全文 search-and-revise。

**工作量估計**：1 小時（grep + 逐條判斷）。

---

## 5. 次要問題

### 5.1 文獻引用格式

**審稿人建議**：引言提「Gough (1969)」「Braginsky & Roberts (1995)」作者-年，但正文用 [11][12] 數字格式，不一致。

**類型**：(A) 技術修補。

**回應策略**：
JCP 格式用數字；但引言文字裡提作者名是沒問題的（標準做法：「Gough [11] first proposed...」）。要做的是統一——檢查 §1、§8 所有人名出現處，確保後面都有 `[n]` 編號。

**工作量估計**：15 分鐘。

---

### 5.2 Tab. 7.1 的 $\Delta E/E_0$ 數值讀法

**審稿人建議**：$\Delta E/E_0$ 從 +2.6 到 +2.6e79 的震盪要在文字更明確提示。

**類型**：(A) 論述補強。

**回應策略**：
§7.4「Total energy drift as a diagnostic」已經解釋了 +2.6 是診斷假象（interpolation noise）；但沒單獨拉出來強調 +2.6e79 的物理意義——這是 IMEX 失穩的 smoking gun。

補一句：「The single entry $\Delta E/E_0 = 2.6\times 10^{79}$ at IMEX / amp = $10^{-1}$ is the numerical signature of the instability diagnosed in Finding 2: over 800 steps the nonlinear feedback amplifies the round-off injected by the CN/AB2 coupling by a factor $(1+\varepsilon)^{800}$ with $\varepsilon \approx 0.25$.」

**工作量估計**：20 分鐘。

---

### 5.3 附錄化

**審稿人建議**：§2.3 Lane--Emden 推導、§4.2 GYRE 係數定義搬附錄。

**類型**：(A) 結構調整。

**回應策略**：
同意。提議：
  - §2.3 保留背景與 $\rho_{\mathrm{cut}}$ 選擇的動機（半頁），把 (2.5) 求解細節搬 **Appendix A**。
  - §4.2 保留方程 (4.1)--(4.2) 但把 $V_2, A^\star, U, c_1, \Gamma_1$ 的逐一背景定義搬 **Appendix C**。
  - Appendix B（上面 3.3 提的 reproducibility checklist）也一起建。

**補丁位置**：新增 `11_appendix.md`。

**工作量估計**：半天（純搬運 + 銜接句）。

---

## 6. 匯總：修訂計劃

按優先級分三輪：

### 輪次 1（必做，核心學術可信度）

| # | 意見 | 類型 | 位置 | 時間 |
|---|---|---|---|---|
| 1.1 | Theorem 6.1 正式化 | 推導 | §6.5 新 | 1 天 |
| 1.2 | Leibniz 缺陷 aliasing 分析 | 推導 | §5.2 Remark 5.1 | 1.5 天 |
| 2.1 | GYRE 對等性 + error budget 實驗 | 論述+實驗 | §4.2 澄清 + §4.5 擴展 | 1.5 天 |
| 2.2 | 非線性 IMEX 公平性聲明 | 論述 | §7.5 路徑 1 | 0.5 小時 |

**小計：~4 天**

### 輪次 2（強化，應對可能的二輪審稿）

| # | 意見 | 類型 | 位置 | 時間 |
|---|---|---|---|---|
| 1.3 | IMEX 失穩穩定性域分析 | 推導+實驗 | §7.3 | 2 天 |
| 3.2 | $\rho_{\mathrm{cut}}$ 敏感性 | 實驗 | §2.3 + §3 新表 | 0.5 天 |
| 3.1 | SL 求解器可擴展性討論 | 論述 | §8.1 | 0.5 天 |

**小計：~3 天**

### 輪次 3（潤色）

| # | 意見 | 類型 | 位置 | 時間 |
|---|---|---|---|---|
| 2.3 | Python 原型方法論重定位 | 論述 | §1.3 + §8.5 | 0.5 天 |
| 3.3 | 再現性檢核表 | 附錄 | Appendix B | 0.5 天 |
| 4.1 | 三部分結構 | 結構 | ToC | 1 小時 |
| 4.2 | 語氣中立化 | 潤色 | 全文 | 1 小時 |
| 5.1 | 引用格式統一 | 格式 | §1, §8 | 15 分鐘 |
| 5.2 | Tab. 7.1 $\Delta E$ 備註 | 潤色 | §7.4 | 20 分鐘 |
| 5.3 | 附錄化 | 結構 | 新 §11 | 0.5 天 |

**小計：~2 天**

**總計：~9 工作日**

---

## 7. 幾個未解開的判斷題（需要決策）

以下三項是方法論選擇，不是技術修補，建議先拍板再動手：

1. **非線性比較路徑 1 vs 路徑 2**（§2.2 上面）：要不要補 IMEX-RK3？我建議先走路徑 1（純聲明），若審稿堅持再補。

2. **結構選項 1 vs 2**（§4.1 上面）：加「Core contribution」標籤還是做三部分劃分？我建議選項 2。

3. **新附錄數量**：擬新增 A（Lane-Emden 細節）、B（再現性 checklist）、C（GYRE 係數）。若不想篇幅膨脹，可把 A+C 合成一個「Appendix: Background definitions」。

這三個決策做完，就可以進入輪次 1 開工。

---

## 附：對審稿人整體評估的看法

審稿意見整體合理且有建設性。核心兩條（Theorem 6.1 正式化、Leibniz 缺陷 aliasing 解釋）確實是現階段的弱點，修補之後論文的理論深度會明顯提升。

GYRE 對等性這條有一半是我們自己論述不清（我們確實用了四變數完整模型），澄清後其實是加分項。

方法論公平性、再現性、結構、潤色這些都是標準二輪意見，照做就好。

我的判斷：**這些意見全部可在現有材料 + 約 9 個工作日內回應完**。沒有一條需要推倒重來。
