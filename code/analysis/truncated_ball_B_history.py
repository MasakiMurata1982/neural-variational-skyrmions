# -*- coding: utf-8 -*-
"""打ち切り球(r_max_topo=12)の 4 本(2026-08-28_B3_ansatz_free/runs_free2)について、訓練中の |B| の最大値と |B|>1e-3 の区間。"""
import os, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
D = os.path.join(RECORDS, "2026-08-28_B3_ansatz_free", "runs_free2")
TAGS = ["free_B1_effkan_w16_d4_ep5000_s0_g5", "free_B1_mlp_w64_d4_ep5000_s0", "free_B2_effkan_w16_d4_ep5000_s0_g5", "free_B2_mlp_w64_d4_ep5000_s0"]
def segments(mask):
    ep = np.arange(1, len(mask) + 1); segs = []; start = None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i - 1 if not m else i; segs.append((int(ep[start]), int(ep[end]))); start = None
    return segs
H = {}
print("| run | epochs | max|B| (epoch) | epochs with |B|>1e-3 | intervals (epoch ranges) | max|B| after epoch 20 | B_final (training grid) |")
print("|---|---|---|---|---|---|---|")
for t in TAGS:
    d = np.load(os.path.join(D, t + "_hist.npz")); B, E = d["B"], d["E"]; H[t] = (B, E)
    a = np.abs(B); i = int(a.argmax()); m = a > 1e-3; segs = segments(m)
    segtxt = ", ".join(f"{s}-{e}" if s != e else f"{s}" for s, e in segs) if segs else "none"
    print(f"| {t} | {len(B)} | {a[i]:.4f} (ep {i+1}) | {int(m.sum())} | {segtxt} | {a[20:].max():.2e} | {B[-1]:+.5f} |")
    if segs and len(segs) > 1:
        print(f"|   | details of intervals for {t}: " + "; ".join(f"ep {s}-{e}: max|B|={a[s-1:e].max():.4f}, max E={E[s-1:e].max():.1f}" for s, e in segs) + " |")
