# -*- coding: utf-8 -*-
"""表 1 の再実行(2026-09-11_table1_models)の集計: table1_models_eval.json(60 run × 48/80 格子)から
 (1) 行 × モデルごとの 48 格子シード平均(表 1 の値と比較)、80 格子シード平均、格子間シフトの最大 |ΔE| と B のずれ、
 (2) 既存 run(同構成・同シード)の記録との差(再現性)、
を出力する。実行: python3 summarize_table1_models.py [runs_reported.csv]"""
import os, sys, json, csv, statistics as st, collections
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
ev = json.load(open(os.path.join(DATA, "table1_models_eval.json")))
rows = list(csv.DictReader(open(CSV))); f = float
TABLE1 = {("shared", "effkan"): 2.35705, ("shared", "mlp_tanh"): 2.35794, ("shared", "mlp_gelu"): 2.35720,
          ("lr_tuned", "effkan"): 2.35700, ("lr_tuned", "mlp_tanh"): 2.35743, ("lr_tuned", "mlp_gelu"): 2.35705,
          ("tb30", "effkan"): 2.35722, ("tb30", "mlp_tanh"): 2.35781, ("tb30", "mlp_gelu"): 2.35712,
          ("tb120", "effkan"): 2.35703, ("tb120", "mlp_tanh"): 2.35711, ("tb120", "mlp_gelu"): 2.35699}

# 既存 run(同 tag、元フォルダ)の 48 格子値
def orig_E(tag, row):
    dirs = {"shared": ["2026-08-27_B2_arch_sweep/runs", "2026-09-11_refit_v2/runs_refit_v2", "2026-08-29_weekend/runs_act", "2026-09-08_gelu_sweep/runs"],
            "lr_tuned": ["2026-08-29_weekend/runs_lr", "2026-08-29_weekend/runs_act", "2026-09-10_table1_seeds/runs_lr"],
            "tb30": ["2026-08-28_B5_finalize/runs_timebudget", "2026-08-29_weekend/runs_gelu_tb", "2026-09-10_table1_seeds/runs_tb"],
            "tb120": ["2026-08-28_B5_finalize/runs_timebudget", "2026-08-29_weekend/runs_gelu_tb", "2026-09-10_table1_seeds/runs_tb"]}[row]
    for r in rows:
        if r["run_id"] == tag and f"{r['experiment']}/{r['runs_dir']}" in dirs:
            return f(r["E"]), int(r["epochs_run"] or 0)
    return None, None

g = collections.defaultdict(list)
for key, v in ev.items():
    g[(v["row"], v["arch"])].append(v)
print("| row | arch | n | mean E48 (re-run) | table 1 | mean E80 | mean(E80-E48) | max|E80-E48| | max|B80-2| | max|re-run - original| (48) |")
print("|---|---|---|---|---|---|---|---|---|---|")
worst_all = []
for row in ("shared", "lr_tuned", "tb30", "tb120"):
    for arch in ("effkan", "mlp_tanh", "mlp_gelu"):
        v = g[(row, arch)]
        E48 = [x["eval"]["48x28x36"]["E"] for x in v]; E80 = [x["eval"]["80x44x60"]["E"] for x in v]
        d = [b - a for a, b in zip(E48, E80)]; dB = [abs(x["eval"]["80x44x60"]["B"] - 2) for x in v]
        diffs = []
        for x in v:
            tag = key_tag = [k for k, vv in ev.items() if vv is x][0].split("/")[-1]
            oe, oep = orig_E(tag, row)
            if oe is not None: diffs.append((x["record_E48"] - oe, tag, oep))
        md = max(diffs, key=lambda t: abs(t[0])) if diffs else (float("nan"), "-", 0)
        worst_all.append((max(abs(t) for t in d), row, arch))
        print(f"| {row} | {arch} | {len(v)} | {st.mean(E48):.6f} | {TABLE1[(row, arch)]:.5f} | {st.mean(E80):.6f} | {st.mean(d):+.1e} | {max(abs(t) for t in d):.1e} | {max(dB):.1e} | {md[0]:+.1e} ({md[1].split('_ep')[0]} s{md[1].split('_s')[1][:1]}, n_orig={len(diffs)}) |")
print("\nmax |E80-E48| over all 60 runs:", f"{max(worst_all)[0]:.1e}", max(worst_all)[1:])
for arch in ("effkan", "mlp_tanh", "mlp_gelu"):
    m = max(w for w in worst_all if w[2] == arch); print(f"  {arch}: max |E80-E48| = {m[0]:.1e} ({m[1]})")
