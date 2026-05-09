# Derivation book template

克隆这个目录作为新求解器的 derivation book 起点:

```bash
cp -r docs/derivations/template docs/derivations/<new_solver_name>
cd docs/derivations/<new_solver_name>
# 编辑 README.md 列出 Part A/B/C/D 的 scope
# 写第一个 scripts/a1_*.py,对应 sections/a1_*.md
bash run_all.sh && bash build_manuscript.sh --pdf
```

## 已有的模板内容

- `scripts/_common.py` — 共享 sympy 符号表 + `LatexDump` helper +
  `assert_zero(expr, name)` 断言
- `run_all.sh` — 批跑所有 `scripts/*.py`,报告 pass/fail
- `build_manuscript.sh` — 拼接 `sections/*.md` → `manuscript.md` + 可选 PDF

## 新 solver README.md 应包含

1. **Scope** —— Part A/B/C/D 列表,每部分的章节
2. **Reproducibility protocol** —— 3 条强规则(见 `../README.md`)
3. **File structure** —— scripts/sections/output 布局
4. **How to regenerate** —— `bash run_all.sh && bash build_manuscript.sh`
5. **Derivation ↔ code 的链接** —— 对应 `src/gpu/.../<file>.cu` 注释里
   引用本 book 的章节锚点

参考完整示范:`docs/derivations/mhd/`。
