# -*- coding: utf-8 -*-
"""図 2 と 3.1 節の E_0 = 2.41627: B=2 の有理写像アンザッツ(ネットワークの補正なし)のエネルギーを、
1 次元の値(E1d)と、評価格子 48×28×36、80×44×60 の値で出す。"""
import os, sys, torch, json
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
sys.path.insert(0, SRC)
from physics import train_profile_1d, q_base_field, eval_multi_grid
dev = torch.device("cuda")
torch.manual_seed(0)
pm, E1d, I4 = train_profile_1d(B=2, epochs=1500, device=dev, verbose=False)
for p in pm.parameters(): p.requires_grad_(False)
qb = lambda xi: q_base_field(xi, pm, B=2)
e48 = eval_multi_grid(qb, dev, quad=(48, 28, 36), r_max=60.0)
e80 = eval_multi_grid(qb, dev, quad=(80, 44, 60), r_max=60.0)
row = dict(E1d=E1d, E48=e48["E"], B48=e48["B"], dE48=e48["dE_grids"], E80=e80["E"], B80=e80["B"], dE80=e80["dE_grids"])
print(f"E1d={E1d:.6f}  48: E={e48['E']:.6f} B={e48['B']:.6f} dE={e48['dE_grids']:.1e} | 80: E={e80['E']:.6f} B={e80['B']:.6f} dE={e80['dE_grids']:.1e}", flush=True)
json.dump([row], open(os.path.join(DATA, "check_base_profile.json"), "w"), indent=1)
