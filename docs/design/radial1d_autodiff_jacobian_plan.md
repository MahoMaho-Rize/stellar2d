# radial1d autodiff Jacobian — 設計備忘

**狀態:暫緩(pending trigger)**,非當前衝刺目標;本文記錄方案、觸發條件、與 JFNK 的關係,待觸發時直接上。

---

## TL;DR

我們現在的 JFNK + Viallet + row-scaled Armijo 在 cgs 下 **1–3 iter/Newton step 收斂**,但 **GMRES 每步都頂 30 iter 上限**。原因不是 matvec 不夠好,而是 **preconditioner 是恆等矩陣**。

加 autodiff **不是取代 JFNK 的 matvec**,而是為 preconditioner 快速建一個 **block-tridiagonal 近似 Jacobian**,讓 GMRES 從 30 iter 降到 3–5 iter,總成本大約 **4× 加速**。

**現在不做**,因為:

1. Newton 已經可用,ignition demo 和 Tier-2 驗證主線不卡這裡。
2. Dual\<T\> 模板化要改 ~20 個 kernel,投入 2–3 天;會中斷現有驗證。
3. 真正的下一步獨特價值在 `radial1d → cart_ale2` 2D 橋樑,不在把 1D Newton 從 1 iter 優化成 0.3 iter。

**觸發條件(任一)** 滿足時立即執行:

- 要把 dt 從目前 1e6 s 推到 ≥ 1e10 s(真正跨完整 τ_KH)→ GMRES 成本必須砍
- 引入 aprox13 或更完整的核反應網路 → stiff 導致 FD noise 放大,exact matvec 變關鍵
- `cart_ale2` 也要寫 implicit Newton → 在 1D 先練基礎設施更合算

---

## 現狀:為什麼 GMRES 卡 30

`src/gpu/radial1d_implicit.cu`:

```cpp
void Radial1DSolver::apply_precond_implicit(const double* d_v_in, double* d_Mv,
                                            double /*inv_dt*/) {
    int N = N_dof, B = 256;
    k_r1di_copy<<<(N+B-1)/B, B>>>(d_Mv, d_v_in, N);   // ← M⁻¹ = I
}
```

GMRES 對 `J δU = −F` 從零初始化、每 iter 做 1 次 matvec + 1 次 dot。我們的 matvec 是 JFNK FD:

```
J · v ≈ (F(U + α v) − F(U)) / α
```

**沒有 preconditioner** 時,GMRES 收斂率完全由 J 的條件數決定。radial1d 的 J 在 cgs 下條件數大(v、r、e 三個場跨 10 位),即便用了 Viallet 兩側 scaling,仍然需要 ~30 iter 才達到 1e-3 relative residual。

典型 Newton step 成本(64 zones):

- 1× `compute_F` (baseline)
- 30× FD matvec = 60× `compute_F` (每次 FD 要 2 次 F eval)
- 1× Armijo line search F eval
- **總計 ≈ 62 次 F eval per Newton step**

所以就算 Newton 只需 1 iter,每 dt step 仍然 ≈ 62 次 F eval。這是我們現在時間成本的主要項。

---

## 方案 A:exact matvec(取代 FD)

**做法:** 把 F 裡所有 double 模板化成 `Dual<T>`,forward-mode autodiff 算精確 `J · v`,一次 pass。

**好處:**

- 消除 α 選擇這個 hack(今天花一堆功夫在 Viallet 兩側 scaling,本質就是讓單一 α 能 work)
- 消除 FD noise floor:√εₘ · ||U|| / ||v|| 在 cgs 下是 1e−8 · 1e17 = 1e9,對 GMRES 收尾殘差是污染
- Newton quadratic convergence 更乾淨 —— stiff 核反應進來時尤其明顯

**壞處:**

- **不加速** —— matvec 次數不變,只是每次 matvec 從 2 次 F eval 變成 1 次 dual F eval(略快,但不改變 O(30) 複雜度)
- Dual\<T\> 模板化成本高 —— 每個 `__device__` function 都要適配

**單獨做 A,收益有限。** A 的真正價值是跟 B 配套。

---

## 方案 B(真正有用):tridiag autodiff J → preconditioner

### 核心觀察

Lagrangian 1D 裡,每個 `F_i(U)` 只依賴 `U_{i-1}, U_i, U_{i+1}` 三個 zone 的狀態(face velocity、zone density、zone energy 都在 ±1 zone 範圍內耦合)。所以 Jacobian 是 **block-tridiagonal**,每 block 3×3(for v, r, e)或 4×4(加 species)。

完整 J 的 non-zero 元素:`3 · 3 · nz = 9·nz`,對 nz=128 是 1152 個元素 —— tiny 到直接存成 3 個 nz·3·3 block 數組。

### 建 J 的 autodiff 做法

用 **forward-mode seeded AD**:

1. 對每個 zone 索引 `i ∈ {0, ..., nz-1}`,seed dual 在 `U_{i-1}, U_i, U_{i+1}` 的 3 個 zone × 3 個 field = 9 個分量,每次一組(3 個 zone stride 掃過整個網格)。
2. 跑一次 `compute_F(U)` 回傳每個 F_j 的 9 個偏導數。
3. 組成 `J_diag[i]`, `J_lower[i]`, `J_upper[i]` 的 3×3 blocks。

**成本:9 次 forward-mode F eval 全部 zone 同時計算(一次性),每 Newton step 建一次。**

對 nz=128:9 次 F eval vs 60 次 FD matvec → 建 J 比建 matvec 還便宜(factor 6.7)。

### preconditioner:block-Thomas

有了 block-tridiag J,preconditioner 就是 **block-Thomas 算法解 `J_approx · δU = rhs`**:

- O(nz) 成本(no iteration)
- 近似 J 對 radial Lagrangian 問題非常接近真實 J(唯一漏掉的是跨 zone ≥ 2 的耦合,很小)
- 我們 BE radiation 已經有 scalar Thomas 的 host 實現,block 版本抄一次就行

### 預期效果

GMRES 收斂率由 `κ(M⁻¹ J)` 決定,其中 M 是 preconditioner。block-tridiag J 跟真實 J 差距主要在:

- radiation diffusion 的 MLT conductivity 讓 off-tridiag elements 非零(但 K_conv Picard-lagged,所以實際 J 看起來仍是 tridiag)
- nuclear burning 是 point-wise(不引入任何新耦合)
- advection 是嚴格 tridiag

所以 `M ≈ J` 到 1% 級別,**GMRES 2–4 iter 就能收斂到 1e-3 tol**。

### 新的 per-Newton-step 成本

- 1× baseline F eval
- 9× forward-mode F eval (建 J)
- 1× block-Thomas solve(O(nz) 標量運算,可忽略)
- 3× FD matvec for GMRES = 6 F eval
- 1× Armijo
- **總計 ≈ 17 F eval** vs 目前 62 → **~3.6× 加速**

### 如果方案 A 已經做

如果已經把 F 模板化成 Dual\<T\> 了,matvec 也可以換成 exact J·v,每次 matvec 1 次 dual F eval。GMRES 2–4 iter × 1 eval = 3 eval,總 per-step:

- 1 baseline + 9 autodiff J build + 3 exact matvec + 1 Armijo = **14 F eval** → **~4.4× 加速**

---

## 模板化成本估算

`src/gpu/radial1d_*.cu` 裡的 device function 約 20 個,最主要需要 Dual 相容的:

- `k_rad1d_zone_primitives_eos` — EOS 呼叫
- `k_rad1d_cfl_eos`
- `k_rad1d_face_flux` / 相關 momentum / energy RHS
- `k_r1di_compute_F` — 最外層包裝,autodiff 都要穿過這裡
- `helm_eval` / `helm_T_from_rho_e` — **最棘手**,biquintic Hermite + Newton 反解需要二階導數

**工作量估計:**

- 建立 `Dual<T>` 含 cgs 常數的 POD + 基本四則運算:0.5 天
- Helm EOS `helm_eval` dual 相容:1 天(biquintic 本來就是多項式,但反解 Newton 的 dual 展開要想)
- `compute_F` 及所有流量 kernel:1 天
- block-Thomas + 整合測試:0.5 天
- **合計:3 天**

---

## 為什麼不先加 preconditioner 再決定

可以。**最小努力路徑:先手寫一個 approximate block-tridiag J**(不靠 autodiff,直接從 F 的 analytical 結構推出 block entries),接 block-Thomas 當 preconditioner,看 GMRES iter 數到底降到多少。

- 如果 GMRES 從 30 → 5:證明 tridiag 結構對 —— 這時才值得上 autodiff 讓 J build 自動化
- 如果 GMRES 從 30 → 20:tridiag 不夠近似,要加 block-pentadiag —— autodiff 立即解救,手寫死活都難維護

這個「手寫先探路」的做法只需 **0.5 天**,風險最低。**觸發條件滿足時先做這個,再決定下一步。**

---

## 關係到 MESA

MESA Fortran Newton 用 **analytic Jacobian 手寫**(在 `star/private/hydro_eqns.f90` 及其朋友裡),因為:

- 純 Fortran,固定物理,手寫一次就行
- 沒有 GPU 考量,不用擔心 device-side Dual 傳遞

我們 CUDA + 多 EOS/核網/對流拼接 → **手寫 J 不現實**,autodiff 正是對的工具。等價地,autodiff 是 MESA 手寫 J 的 "自動化升級版",而 JFNK 是另一軸(不存儲 J)。三者不衝突,按需配合。

---

## 動手時的 checklist(觸發時直接跟)

1. 先手寫 block-tridiag preconditioner 探路(**0.5 天**,只動 implicit 模塊)
   - 複用 BE radiation tridiag 的 Thomas kernel
   - 3×3 block 版本:直接 `solve_3x3` 替 scalar 除法
   - benchmark GMRES iter 數改善
2. 如果 preconditioner 明顯有效 → 判斷是否上 autodiff 自動化 J build
3. autodiff 上的話,分階段:
   - (a) `Dual<T>` POD + 運算 + 測試
   - (b) Helm EOS 相容
   - (c) F eval 相容
   - (d) 取代 handwritten J
4. 每階段通過現有 `test_helm_eos` + cgs Newton 回歸,確保不退

---

## 引用

- Knoll & Keyes 2004, "Jacobian-Free Newton-Krylov methods" — 我們現在的骨幹
- Weaver & Fryxell 2006 (MESA-style handwritten J) — 傳統路線
- `ForwardDiff.jl`, `StanMathLibrary`, `Enzyme` — autodiff 現代工具鏈(CUDA 有 Enzyme 支援)
- Kopecz & Meister 2018 "Stiff ODEs with autodiff Jacobian" — 理論背景
