# -*- coding: utf-8 -*-
"""表 7 の群 1・群 2(q_vac + g_soft·v の構成、5000 エポック)の 8 本の B 履歴: max|B|、|B|>1e-3 のエポック区間、20 エポック目以降の max|B|、最終 B。"""
import os, glob, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
def segments(mask):
    segs = []; start = None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if start is not None and (not m or i == len(mask) - 1): segs.append((start + 1, (i if not m else i + 1))); start = None
    return segs
print("| run | w_topo | epochs | max|B| (epoch) | epochs with |B|>1e-3 | intervals | max|B| after epoch 20 | B_final (training grid) | B (multi-grid eval) | E |")
print("|---|---|---|---|---|---|---|---|---|---|")
for d in [os.path.join(RECORDS, "2026-09-14_soft_5000", "runs_w200"), os.path.join(RECORDS, "2026-09-14_soft_5000", "runs_w1000")]:
    for f in sorted(glob.glob(os.path.join(d, "soft_*.json"))):
        r = json.load(open(f)); B = np.load(f.replace(".json", "_hist.npz"))["B"]; a = np.abs(B); i = int(a.argmax()); m = a > 1e-3; segs = segments(m)
        seg = ", ".join(f"{s}-{e}" if s != e else f"{s}" for s, e in segs) if segs else "none"
        print(f"| {os.path.relpath(f, SKY)} | {r['config']['w_topo']:.0f} | {len(B)} | {a[i]:.2e} (ep {i+1}) | {int(m.sum())} | {seg} | {a[20:].max():.2e} | {B[-1]:+.6f} | {r['B']:+.6f} | {r['E']:.4f} |")
