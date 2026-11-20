# -*- coding: utf-8 -*-
"""imprinting 設計実験(2026-09-14)の集計: 水準(目標値)× ネットワークごとの成功数 / セクター保持数、imprint_loss_eval の範囲、到達フラグ、
同水準での KAN 対 GELU の Fisher 両側検定。成功 = 統一プロトコルの検証(|B-2|<1e-3、|E-E_ref|/E_ref<1%、|E2/E4-1|<0.05、ΔE<1e-2、E_ref=2.3582)。
「pooled over networks」の Fisher 検定は、論文 6.4 節と同じ 3 つのネットワーク(各水準 15 本、成功 0 / 6 / 15、p = 0.017 と 7e-4)で計算する。
実行: python3 summarize_imprint_design.py"""
import os, glob, json, collections, statistics as st
from scipy.stats import fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); E_REF = 2.3582
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
rows = []
for f in sorted(glob.glob(os.path.join(RECORDS, "2026-09-14_imprint_design", "runs_target*", "*.json")) + glob.glob(os.path.join(RECORDS, "2026-09-14_imprint_design", "runs_default8000", "*.json"))):
    r = json.load(open(f)); L = r["lumps"]; cfg = r["config"]
    arch = "KAN[16]x4" if r["kind"] == "effkan" else f"{cfg.get('mlp_act','tanh')}-MLP[{r['width']}]x4"
    sector = abs(r["B"] - 2) < 1e-3; ok = sector and abs(r["E"] - E_REF) / E_REF < 0.01 and abs(r["virial"] - 1) < 0.05 and r["dE_grids"] < 1e-2
    tgt = L["pretrain_target_loss"] if L.get("pretrain_target_loss") is not None else 0.0   # 0.0 = 目標なし(既定 8000 ステップ)
    rows.append(dict(arch=arch, target=tgt, seed=r["seed"], used=L["pretrain_epochs_used"], reached=L["pretrain_reached"], loss=L["pretrain_loss"],
                     loss_eval=L["imprint_loss_eval"], E=r["E"], B=r["B"], vir=r["virial"], dE=r["dE_grids"], sector=sector, ok=ok, t=r["time_train_sec"], tag=r["tag"]))
print(f"{len(rows)} runs")
print("\n== per level x network ==")
print("| target | network | n | reached target | pretrain steps used (min-max) | imprint_loss_eval (min-max) | sector retained | success |")
print("|---|---|---|---|---|---|---|---|")
g = collections.defaultdict(list)
for r in rows: g[(r["target"], r["arch"])].append(r)
for (tgt, arch) in sorted(g, key=lambda k: (-k[0], k[1])):
    v = g[(tgt, arch)]; nre = sum(bool(r["reached"]) for r in v)
    print(f"| {('default (no target, 8000 steps)' if tgt == 0.0 else f'{tgt:g}')} | {arch} | {len(v)} | {('-' if tgt == 0.0 else f'{nre}/{len(v)}')} | {min(r['used'] for r in v)}-{max(r['used'] for r in v)} | {min(r['loss_eval'] for r in v):.1e}-{max(r['loss_eval'] for r in v):.1e} | "
          f"{sum(r['sector'] for r in v)}/{len(v)} | {sum(r['ok'] for r in v)}/{len(v)} |")
print("\n== KAN vs GELU at equal level (Fisher two-sided, success) ==")
for tgt in sorted({r["target"] for r in rows if r["target"] > 0}, reverse=True):
    a = g.get((tgt, "KAN[16]x4"), []); b = g.get((tgt, "gelu-MLP[64]x4"), []); c = g.get((tgt, "tanh-MLP[64]x4"), [])
    for key in ("ok",):
        line = f"  target {tgt:g} {key:6s}:"
        if a and b:
            x, y = sum(r[key] for r in a), sum(r[key] for r in b); line += f" KAN {x}/{len(a)} vs GELU {y}/{len(b)} p={fisher_exact([[x, len(a) - x], [y, len(b) - y]])[1]:.2f};"
        if b and c:
            y, z = sum(r[key] for r in b), sum(r[key] for r in c); line += f" GELU {y}/{len(b)} vs tanh {z}/{len(c)} p={fisher_exact([[y, len(b) - y], [z, len(c) - z]])[1]:.2f};"
        if a and c:
            x, z = sum(r[key] for r in a), sum(r[key] for r in c); line += f" KAN {x}/{len(a)} vs tanh {z}/{len(c)} p={fisher_exact([[x, len(a) - x], [z, len(c) - z]])[1]:.2f}"
        print(line)
print("\n== pooled over the three networks (KAN[16]x4, GELU-MLP[64]x4, tanh-MLP[64]x4; 15 runs per level): success by level (Fisher two-sided between adjacent levels) ==")
lv = sorted({r["target"] for r in rows if r["target"] > 0}, reverse=True)
for t1, t2 in zip(lv, lv[1:]):
    A = [r for r in rows if r["target"] == t1]; B = [r for r in rows if r["target"] == t2]
    x, y = sum(r["ok"] for r in A), sum(r["ok"] for r in B); print(f"  {t1:g}: {x}/{len(A)}  vs  {t2:g}: {y}/{len(B)}  p={fisher_exact([[x, len(A)-x], [y, len(B)-y]])[1]:.1e}")
def tg(r): return "default 8000" if r["target"] == 0.0 else f"{r['target']:g}"
print("\n== all runs ==")
print("| target | network | seed | steps used | reached | loss (single batch) | imprint_loss_eval | E | B | E2/E4 | dE | sector | success | time s |"); print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (-(r["target"] if r["target"] > 0 else 1e-9), r["arch"], r["seed"])):
    print(f"| {tg(r)} | {r['arch']} | {r['seed']} | {r['used']} | {r['reached']} | {r['loss']:.2e} | {r['loss_eval']:.2e} | {r['E']:.5f} | {r['B']:.5f} | {r['vir']:.4f} | {r['dE']:.1e} | {'o' if r['sector'] else 'x'} | {'o' if r['ok'] else 'x'} | {r['t']:.0f} |")
