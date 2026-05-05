# lowmach JFNK 救援方案(基於 MUSIC 實現細節)

> 2026-05-01,基於 deep-research 對 Viallet+2011/2016、Goffrey+2017、Miczek+2015 的實現層總結。
>
> **目標**:讓 `lowmach` solver 能真正跨越擾動時標(現狀:HSE ✅,1e-4 擾動 ✅,1e-3 擾動 ❌ dt 塌到 10⁻⁵ 以下)。

## 當前診斷

- **HSE**:38 step, tend=5, M/E 守恆到 10 位精度 ✅
- **1e-4 擾動**:76 step, tend=5 ✅(在 base state scaling revert 前)
- **1e-3 擾動**:dt 卡在 ~1e-5,F_ρ 約 1e-4,F_mr 約 1e-8,Newton 迭代 Fnorm 不下降 ❌
- **已完成且確定對的修復**(keep):
  - S_E = ρv·g 物理 bug 刪除(內能方程不該有 gravity 源)
  - CFL 改純對流(JFNK 不受聲速限制)
  - `--lm-muscl-order` 可選 MUSCL 對流
- **已 revert 的錯誤嘗試**:base state scaling 改 scR、JFNK matvec 改 L/R scaled、dt_cap 改 0.1 + CFL 加聲速 floor — 都被 revert

## 根因(來自文獻,不是猜)

**F_ρ 比 F_mr 大 10⁴× 不是 Newton 的問題**,是 **1 階 upwind / HLLC 在低馬赫的 O(1/M) 數值耗散** signature:

- Guillard-Viozat 1999、Miczek-Röpke-Edelmann 2015 定理:Godunov-type 的密度通量耗散項正比於 `ρ·c_s·Δu`
- 物理連續性項正比於 `ρ·u·∂/∂x`,比值 `c_s/u = 1/M`
- M=1e-3 時,F_ρ 1000× 大於物理項
- Newton 無法消除 — 沒有 δU 能 zero out 純 scheme dissipation
- 1e-4 擾動跑通是因為 M 足夠小時,絕對 F_ρ 也足夠小,判據 Fnorm_per_cell < 1e-4 被滿足(偶然過關)

## 三階段方案

風險/回報遞增,每階段獨立驗證,成功就停。

---

### 階段 1 — Mach-scaled 通量耗散修正(Miczek+2015 / Thornber+2008)

**假設**:F_ρ 污染源是 1 階 upwind 的 scheme dissipation。

**改動**:`lm_residual.cu` 的 `upwind_r / upwind_t` lambda,把密度通量的**有效速度**按 Mach 數降:

```cuda
// Thornber-style low-Mach correction: scale velocity jump by f(M)
double M_face = fmax(fabs(ul)/cs_l, fabs(ur)/cs_r);
double f_M = fmin(1.0, M_face);
// Apply only to the dissipative part of the flux
// For pure upwind: F = vf·q_up.  Modified:
//   F = 0.5·(vf_L + vf_R)·q_avg + 0.5·f_M·|vf|·(q_R - q_L)
// At M=1 reduces to standard upwind; at M→0 pressure-jump dissipation vanishes.
```

注意:只改密度/能量通量,**不要動動量通量**(那裡的壓力項是物理的,不是耗散)。

**驗證**(exit criteria):
1. HSE 回歸:38 step ≈ tend=5,M/E 守恆 10 位精度(不能變壞)
2. 1e-4 擾動:tend=5 跑完,≤100 step
3. 1e-3 擾動:**至少 tend=1 跑完**,dt 能穩定在 ~1e-3 級別,E 漂移 < 1e-3
4. F_ρ 診斷(加臨時 print)與 F_mr 量級相差 < 10×

如果 (3) 滿足:**階段 1 勝利,收工**。如果不滿足但有明顯改善,進階段 2。

**工作量**:1-2 小時。

---

### 階段 2 — MUSIC 風格 L/R scaling(Viallet 2016 eq 63-65,α₁=1e-5,α₂=1)

**假設**:階段 1 不夠,還需要 Newton convergence 判據對不同方程權重不均。

**改動**:`lm_precond.cu` 的 `k_lm_compute_music_scale`,採用**不對稱參數**:

```cuda
// Right scaling (unknowns) — Viallet 2016 eq 63-64
double alpha1 = 1e-5, alpha2 = 1.0;
scR[ρ]  = rho_c;
scR[e]  = e_c;                   // 注意:MUSIC 用比內能 e, 不是 ρe
scR[u]  = fmax(fabs(u), alpha2 * cs);

// Left scaling (residuals) — eq 65, 關鍵是 alpha1 << alpha2
scL[ρ]  = rho_c;
scL[e]  = rho_c * e_c;
scL[u]  = rho_c * fmax(fabs(u), alpha1 * cs);   // alpha1 = 1e-5 才是 fix
```

**關鍵**:之前我改 scaling 破了 HSE 是因為同時動了 scR 和 Newton 架構。這次**只動 scL**,`α₁=1e-5 << α₂=1` 的**不對稱性是 Viallet 2016 §5.1.1 明確記錄的 low-Mach fix**。

**還要做**:
- `jfnk_matvec` 的 ε 改為 `λ(λ + ||U||/||v||)`,λ=1e-7(Viallet 2016 eq 77,低馬赫驗證過的唯一對形式)
- Newton 收斂判據改用 `||R⁻¹ δX||_∞ < 1e-6`(Viallet 2016 eq 73)

**驗證**:同階段 1 的四項,加上:
5. 1e-3 擾動能跑到 tend=5,E 漂移 < 1e-4

**工作量**:3-4 小時(含調試)。

---

### 階段 3 — Park-2009 δ-form 預處理器(Viallet 2016 §3.3, §4.3)

**假設**:階段 1+2 還不夠,需要替換整個預處理器結構。

**不是 SIMPLE 不是 Schur**,是在 preconditioner 內部**變換到原始變量 (p, e, u)**,然後解一個**標量 δp 的 Helmholtz**:

```
δp/Δt − a²·Δt·∇²δp = −F̃_p,   a² = Γ₁·p/ρ
```

back-substitute:
- `δe` from eq 37
- `δu = −Δt/ρ · ∇(δp)` from eq 38

然後 `δU_primitive → δU_conservative` 通過 `∂U/∂V`(Viallet 2016 eq 21, 23)。

**這個 Helmholtz 在 Δt→∞ 退化為 Laplacian**(係數 a²·Δt² 主導),GMG 收斂極快。

**複雜度**:
- 需要重寫 `apply_preconditioner` for a new `PrecondType::MUSIC_PBP`
- 用現有 `gmg_pressure` 解 Helmholtz(`∇²u - σu = f` 形式,已有 `gmg_schur` 接口)
- 變量變換矩陣 `∂V/∂U`、`∂U/∂V` 按 Viallet 2016 Appendix B 實現
- 保留 line_jacobi / pbp 作為後備

**驗證**:
1-5 同階段 2,加上:
6. GMRES 迭代數在 1e-3 擾動下 < 20(當前 ~120 還不收斂)
7. 單步 wall-clock 不明顯變慢(Helmholtz 一次 GMG vs SIMPLE 兩次)

**工作量**:2-3 天,需要仔細對照 Viallet 2016 §3.3 公式。

---

## 其他已確認有用但**不**是主攻方向的小修

放入 backlog,階段結束後可一起 commit:

- **Newton 收斂要求 `newton >= 1`**(防 trivial convergence):已加但之前 revert,重做
- **Velocity floor in tolerance**:`max(|u|, c_s)` 作 Newton `tol = eps × U₀`(Viallet 2011 §3.2.3)
- **CFL_adv ≤ 0.5 硬 cap**:Goffrey 2017 §2.1 明確警告 preconditioner 超過此值不穩定
- **dt 增長率 + 步數**:`converge ≤ 5 iter → dt × 1.5`,`> 10 iter → dt × 0.5`(Viallet 2011 §3.2.6)
- **ε=1e-7 而非 sqrt(ulp)**:Viallet 2016 §5.1.1

## 不要做的事(之前踩過的坑)

1. **不要在 scR/scL 都放 base state split**:破壞了 HSE。如果要做 perturbation split,只改 `scR`,且要完整驗證 HSE
2. **不要硬改 `dt_cap`**:保留 1.0,讓 CFL 和 Newton iter count 決定
3. **不要把 HLLC 替代 1 階 upwind**:MUSIC 本身用 1 階迎風 + Park-2009 preconditioner,HLLC 在低馬赫耗散更糟

## 關鍵文獻位置(供下次 session 無需 re-search)

- Viallet 2016 A&A 586 A153(JFNK 最完整):https://www.aanda.org/articles/aa/pdf/2016/02/aa27339-15.pdf
  - §3.3 δ-form SI 方程(eq 30-38)
  - §4.1 L/R scaling(eq 63-72)+ α₁=1e-5 參數(§5.1.1)
  - §4.2 FD Jacobian ε formula(eq 77)
  - §4.3 preconditioner variable transform
  - Appendix B `∂V/∂U` 顯式公式
- Viallet 2011 A&A 531 A86(MUSIC 首版):http://www.the-a-maze.net/papers/walder/aa16374-10.pdf
  - §3.2.3 velocity floor U₀=max(u, c_s)
  - §3.2.6 dt 控制器(5/10 threshold)
- Goffrey 2017 A&A 600 A7(benchmarking):https://arxiv.org/abs/1610.10053
  - §2.1 CFL_adv ≤ 0.5 硬 cap
- Miczek-Röpke-Edelmann 2015 A&A 576 A50:https://arxiv.org/abs/1409.8289
  - Godunov 低馬赫耗散定量分析(階段 1 的依據)
- Guillard-Viozat 1999 Comp. Fluids 28(最早的低馬赫耗散論斷)

## 下次 session 快速 kick-off

當前 branch `lowmach-muscl`,基於 main。

當前已修改文件(截至這個 plan 寫作時):
- `src/gpu/lm_residual.cu`:MUSCL 重建 + S_E 刪除 + 純對流 CFL
- `src/gpu/lowmach_solver.cu`:`cfl_max_factor=2`
- `src/gpu/lowmach_solver.h`:`int muscl_order` 字段
- `src/main.cpp`:`--lm-muscl-order` flag

從階段 1 開始實施,用 `runs/lane_emden_perturbed --perturb 1e-3` 作快速驗證。
