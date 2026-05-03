# Round-4 修訂結果追溯文檔

**狀態**：進行中
**決策組合**：D.1.a + D.2.a + D.3.a + D.4.a + D.5.a + D.6.a + D.7.b（JCP 鋒利版本）
**備份位置**：`paper/archive_pre_round4/` (10 個 .md 檔)

本輪是論文章節落地 —— 把 Round-1/2/3 的實驗與推導按「現象 → 排除 → 機制 → 唯一解」重排。

---

## 執行順序 & 狀態

| 天 | 工作 | 狀態 | 實際章節長度 |
|---|---|---|---|
| 1 | §5 完全重寫 | ✅ | 267 行（5.1 shock / 5.2 rule-out / 5.3 Prop 1 / 5.4 scaling / 5.5 Galerkin）|
| 2 | §6 原理前置 + CUDA 移 Appendix | ✅ | 145 行（6.1 原理 / 6.2 Theorem + 證明 / 6.3 結果 / 6.4 summary）|
| 3 | §4 error budget + 3.6e-5 → 9.1e-9 | ✅ | 120 行（含新 Table 4.2 error budget）|
| 4 | §7 IMEX stability 升級 | ✅ | 142 行（含新 equation 7.1 放大矩陣）|
| 5 | §8 抬拔 + Fig 8.1 | ✅ | 156 行（含 Table 8.1 三方法對比 + Fig 8.1 Pareto）|
| 6 | §1/§9/§2/§3 壓縮 | ✅ | §1 120行, §9 46行, §2 改§2.3, §3 改§3.5 |
| 7 | Appendix A/B + PDF 渲染 | ✅ | **28 頁 PDF, 529 KB** |

---

## 關鍵 patch 記錄（填入每節完成時的主要改動）

### §5 — 待完成

### §6 — 待完成

### §4 — 待完成

### §7 — 待完成

### §8 — 待完成

### §1/§9/§2/§3 — 待完成

### Appendix — 待完成

---

## 遇到的問題與決策回溯

（留給執行中記錄）

---

## 最終 PDF 檢查清單

- [x] 總頁數 28 頁（目標 ~25，稍超但可接受）
- [x] Fig 5.1 (locality heatmap), Fig 5.2 (scaling law), Fig 8.1 (complexity) 在正文
- [x] Abstract 含 killer sentence："The failure arises from replacing the inverse of a global elliptic operator by a pointwise scaling"
- [x] Proposition 1 三子句完整：resolution-independent floor + scaling law + structural character
- [x] Theorem 6.1 含 Lemma A + Lemma B + 主定理 + Corollary 完整證明
- [x] 全文 $3.6\times 10^{-5}$ 已替換為 $9.1\times 10^{-9}$（§1, §4, §6, §8, §9）
- [x] §9 conclusion 三句話結構（Identification / Mechanism / Resolution）
- [x] Appendix A/B 組裝完整（A.1 Lane-Emden, A.2 收斂, A.3 CUDA, A.4 NL prototypes, A.5 IMEX stability; B reproducibility table）
- [x] `99_concat.sh` 更新（abstract 新版 + appendix 加入）
- [x] `build_paper_pdf.sh` 成功（圖片絕對路徑注入）
- [x] 備份 `archive_pre_round4/` 保留舊版本

## 關鍵指標對比

| 指標 | Round-3 前 (舊版) | Round-4 後 (新版) |
|---|---|---|
| 標題 | "Sturm–Liouville and Assembled-Matrix..." | "Operator-Consistency Failure and its Resolution..." |
| Abstract framing | "two ideas must be combined" | "Pseudo-spectral methods... fails at the discrete operator level" |
| §5 結構 | 現象 → 三種 rule-out → 結論 | shock → rule-out → Proposition 1 → scaling → Galerkin |
| 核心命題 | Informal Theorem 6.1 only | Proposition 1 (三子句) + formal Theorem 6.1 + Lemmas A/B + Corollary |
| GYRE 精度 | $3.6\times 10^{-5}$ | $9.1\times 10^{-9}$ |
| 圖表數量 | 0 圖 | 3 圖 (Fig 5.1, 5.2, 8.1) + 8 表 |
| 總頁數 | 27 | 28 |
| 定位 | "a better method" | "a discrete-operator theorem + a practical resolution" |
