# Round-2 修訂規劃（決策文檔）

**狀態**：規劃中，待用戶拍板
**上游**：`REVIEW_RESPONSE.md`（全部審稿意見）、`REVIEW_ROUND1_RESULTS.md`（輪次 1 已完成推導與實驗，已歸檔）
**範圍**：Round-1 四項的「論文 patch 落實」+ 審稿意見輪次 2 的新項

---

## 1. Round-2 要解決的四類問題

### 1.A 把 Round-1 的推導與實驗結果「落地」到論文

目前 Round-1 產物只在追溯文檔裡。這輪要真正改 `paper/NN_*.md` 並重新渲染 PDF。

四個 patch 點：
  - **§6.5 重寫** — 從 "informal theorem" 換為 Lemma A + Lemma B + 主定理 + Corollary。
  - **§5.2 新 Remark 5.1** — aliasing 推導 + $N_y$ 掃描表。
  - **§7.5 新 caveat** — IMEX 聲明範圍。
  - **§4 大改** — 物理模型澄清 + §4.5 擴展為 error budget（+ Tab. 4.2）
    + 全文 $3.6\times 10^{-5}\to 9.1\times 10^{-9}$ 搜換。

這部分是**機械工作**，無新推導，但要小心：
  - R1.2 prototype 的 magnitudes 與論文 §5.1 不吻合（$10^{-5}$ vs $6.9\times 10^{-4}$），
    要在 Remark 5.1 誠實披露是 prototype 簡化所致。
  - R1.3 的 3.6e-5→9.1e-9 升級要連帶改 §1.4、§4.4、§4.6、§8.2 四處的數字。
  - §6.5 新定理使用 $\kappa(Q)$，要在 §6.5 前顯式定義，並在 §10 新增
    Higham (2002) 文獻 [27]。

### 1.B 審稿意見輪次 2 未做項（原 REVIEW_RESPONSE.md 規劃的「輪次 2」）

三個新項，按重要性排序：

  - **2.1 IMEX 穩定性域分析**（對應審稿人 §1.3）
    推 IMEX(CN, AB2) 的放大矩陣 $2\times 2$ 特徵值；在 $(\omega\Delta t, \alpha\cdot\text{amp})$
    平面畫穩定域 contour；用振幅掃描 $a = 10^{-4},\dots,10^{-1}$ 的實測
    最大特徵值對照。
  - **2.2 $\rho_{\mathrm{cut}}$ 敏感性實驗**（對應審稿人 §3.2）
    掃 $\rho_{\mathrm{cut}} \in \{0.01, 0.02, 0.05, 0.10\}$，看前 5 個 g-mode
    頻率與本徵函數 $L^2$ 變化；另外補充壁上邊界條件的實現細節（§2.3）。
  - **2.3 SL 求解器可擴展性討論**（對應審稿人 §3.1）
    純論述——LAPACK dgeev 的 $\mathcal O(N_y^3)$ 瓶頸、快速 SL
    變換/Prufer 等替代方案、我們選擇的理由（$N_y\le 128$ bit-reproducibility）。

### 1.C 審稿意見輪次 3 潤色項（原規劃）

小工作量：
  - Python 原型方法論重定位（§1.3 + §8.5）
  - 再現性 checklist（新 Appendix B）
  - 三部分結構（Part I/II/III ToC）
  - 語氣中立化（全文 "cannot"→"fails to..."）
  - 引用格式統一
  - Tab. 7.1 $\Delta E$ 數值備註
  - Lane-Emden 細節附錄化（新 Appendix A）

### 1.D Round-1 披露出的問題

兩個需要在 Round-2 正視：

  - **D.1 R1.2 magnitudes gap**
    簡化 1D prototype $1.05\times 10^{-5}/\text{step}$ vs 論文 $6.9\times 10^{-4}/\text{step}$。
    兩種應對：
      (a) 只披露——在 Remark 5.1 寫「magnitudes scale with $\Delta t$;
          $N_y$-independence is the invariant observable」。
      (b) 補實驗——在完整 2D 代碼裡直接跑 $N_y$ 掃描，磁量級與論文一致。
    建議 (b)，因為審稿人會問「為什麼 prototype 不匹配」。約半天額外實驗（CUDA 端
    已有完整流水線）。
  - **D.2 論文文字與代碼脫節（cubic spline）**
    2026-05-03 代碼已升級為 cubic spline，但論文仍寫 linear。
    修法：§4.5 誠實寫「initial implementation used linear interp for
    ease of setup; cubic spline was adopted during the review response
    and is now the default」。

---

## 2. 三個用戶決策點

修訂啟動前需要拍板：

### 決策 A：R1.3 數字升級範圍

**選項 A1（建議）**：全文將 $3.6\times 10^{-5}$ 更新為 $9.1\times 10^{-9}$（cubic spline 下新測值），
影響 §1.4、§4.4、§4.6、§8.2 四處；§4.5 保留 linear 作為對照。
論文呈現「cubic 是 default，linear 是 legacy」。

**選項 A2**：保留 $3.6\times 10^{-5}$ 不動，只在 §4.5 補充說「cubic spline 可進一步降到 $9.1\times 10^{-9}$」。
改動最小，但犧牲論文的「精度記錄」。

**選項 A3**：僅更新摘要與 §4.4 的數字，其他地方不動。折衷方案。

### 決策 B：R1.2 magnitude gap 應對

**選項 B1（建議）**：披露 + 補 2D CUDA 實驗（半天），讓 Remark 5.1 的 $N_y$ 掃描磁量級
直接對應 §5.1 的 $6.9\times 10^{-4}$。給審稿人零把柄。

**選項 B2**：只披露，寫清楚「prototype 簡化導致磁量級差異，invariance 成立」。
不補實驗。省半天，但審稿人可能會在 round 3 要求補。

### 決策 C：Round-2 範圍

**選項 C1（小範圍）**：只做 1.A（論文 patch 落地）+ 1.D 披露，~2 天工作。
把 1.B / 1.C 推到 Round-3。論文立刻變可送審版。

**選項 C2（中範圍，建議）**：1.A + 1.B 全做 + 1.D 披露，~6 天工作。
覆蓋審稿輪次 2 的所有新項；Round-3 只剩潤色。

**選項 C3（大範圍）**：1.A + 1.B + 1.C 全做，~9 天工作。
一次到位，但工期長。

---

## 3. 推薦組合

**A1 + B1 + C2** = 總計 **~7 個工作日**，產出：
  - Round-1 四個 patch 全部落實
  - IMEX 穩定性域、$\rho_{\mathrm{cut}}$ 敏感性、SL 可擴展性 三個新項完成
  - 2D CUDA $N_y$ 掃描補足 magnitude gap
  - 論文精度數字全部對齊到 $9.1\times 10^{-9}$
  - PDF 重新渲染

完工後還剩下的：潤色、附錄化、結構重組——一併做個 Round-3 就能送審。

---

## 4. Round-2 執行順序（若採推薦組合）

| 天 | 工作 | 產出 |
|---|---|---|
| 1 | 1.A §7.5 caveat + §6.5 重寫 | 兩節文字 |
| 2 | 1.A §5.2 Remark 5.1（含 Round-1 表） | 一節文字 |
| 3 | D.1 補 2D CUDA $N_y$ 掃描 + §5.2 表更新 | 新 CSV + 表 |
| 4 | 1.A §4 大改（4.2 澄清 + 4.5 error budget + 全文數字替換） | §4 重寫 |
| 5 | 2.1 IMEX 穩定性域推導 + 實驗 + §7.3 擴展 | 推導 + contour 圖 |
| 6 | 2.2 $\rho_{\mathrm{cut}}$ 敏感性 + §2.3 實作細節 + 新表 | 實驗 + 表 |
| 7 | 2.3 SL 可擴展性 §8.1 論述 + PDF 重渲染 + Round-2 追溯歸檔 | 完成 |

每一步產物都寫入對應的 `review/r2X_*/` 目錄與 `REVIEW_ROUND2_RESULTS.md`
追溯文檔，保證可回溯。

---

## 5. 待決策問題歸納

1. **決策 A**（數字升級範圍）：A1 / A2 / A3？
2. **決策 B**（magnitude gap 應對）：B1 / B2？
3. **決策 C**（Round-2 範圍）：C1 / C2 / C3？

拍板後，我會建立 `REVIEW_ROUND2_RESULTS.md` 起追溯骨架，並按 §4 的七天順序開工。
