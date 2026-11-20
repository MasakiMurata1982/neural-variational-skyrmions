# -*- coding: utf-8 -*-
"""local_minima_v2.json(モード (c) の 21 本)から、付録 A.4 のモード (c) の量を出す。
B = 2 の 1 本は (I2-I1)/I1 とエネルギーの超過、
B = 3 と 4 の 20 本は (I3-I1)/I1 の範囲、エネルギーの超過の範囲、この 2 つの Spearman の順位相関。
I1 <= I2 <= I3 は慣性テンソルの主値で、inertia_ratios = [I1/I3, I2/I3, 1] なので (I3-I1)/I1 = 1/(I1/I3) - 1、(I2-I1)/I1 = (I2/I3)/(I1/I3) - 1。"""
import os, sys, json, numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
d = json.load(open(os.path.join(DATA, "local_minima_v2.json")))
print("== appendix A.4, mode (c): the quantities stated in the paper ==")
b2 = [r for r in d if r["B"] == 2]; hi = [r for r in d if r["B"] in (3, 4)]
for r in b2:
    print(f"  B = 2, {len(b2)} run ({r['tag']}): (I2-I1)/I1 = {r['inertia_ratios'][1] / r['inertia_ratios'][0] - 1:.2f}; energy {r['excess_pct']:.1f}% above the reference energy")
dev = np.array([1 / r["inertia_ratios"][0] - 1 for r in hi]); exh = np.array([r["excess_pct"] for r in hi])
rho_p, p_p = spearmanr(exh, dev)
print(f"  B = 3 and B = 4, {len(hi)} runs ({sum(r['B'] == 3 for r in hi)} at B = 3, {sum(r['B'] == 4 for r in hi)} at B = 4): (I3-I1)/I1 between {dev.min():.3f} and {dev.max():.1f}; "
      f"energy {exh.min():.1f} to {exh.max():.1f}% above the reference energy; Spearman rank correlation of the energy excess with (I3-I1)/I1 = {rho_p:+.2f} (p = {p_p:.1e})")
print("  paper: B = 2: 0.29 and 1.9%; B = 3 and 4, twenty runs: between 0.095 and 4.1, 1.2 to 10.3%, Spearman +0.79")
print("  | tag | arch | B | E | excess % | I1/I3 | I2/I3 | (I3-I1)/I1 | (I2-I1)/I1 |"); print("  |---|---|---|---|---|---|---|---|---|")
for r in d:
    a, b = r["inertia_ratios"][0], r["inertia_ratios"][1]
    print(f"  | {r['tag']} | {r['arch']} | {r['B']} | {r['E']:.5f} | {r['excess_pct']:.2f} | {a:.3f} | {b:.3f} | {1 / a - 1:.3f} | {b / a - 1:.3f} |")
