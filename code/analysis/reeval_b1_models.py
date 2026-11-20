# -*- coding: utf-8 -*-
"""runs_b1_regrid の保存モデル(FastKAN 3 本 + 対照 effKAN/MLP 2 本、B=1 anchored)を学習し直さず、
48×28×36 と 80×44×60(3 動径写像 a=2.6/3.0/3.4、各 1 回転、seed 1234、r_max=60)で再評価して E, B, E/B を出す。
背景プロファイル(1D、B=1)は code/src/physics.py の train_profile_1d で学習(シード固定、E_1d=1.2314456)。"""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
sys.path.insert(0, SRC)
from train import get_base_profile
from physics import build_q_hard, eval_multi_grid
from models import build_net
dev = torch.device("cuda"); RUNS = os.path.join(RECORDS, "2026-09-07_retraining", "runs_b1_regrid")
torch.manual_seed(0); pmodel, E1d, _ = get_base_profile(1, dev); print(f"1D profile B=1 E_1d={E1d:.7f}", flush=True)
GRIDS = [(48, 28, 36), (80, 44, 60)]
res = {}
for f in sorted(os.listdir(RUNS)):
    if not f.endswith(".json"): continue
    r = json.load(open(os.path.join(RUNS, f))); tag = r["tag"]
    net = build_net(r["kind"], r["width"], r["depth"], dev, r.get("kan_grid", 5), zero_init=True, seed=r["seed"], mlp_act=r["config"].get("mlp_act", "tanh"))
    net.load_state_dict(torch.load(os.path.join(RUNS, tag + "_model.pt"), map_location=dev)); net.eval()
    qb = lambda xi: build_q_hard(xi, net, pmodel, B=1)
    out = {"record_E48": r["E"], "record_B48": r["B"], "record_dE48": r["dE_grids"]}
    for q in GRIDS:
        e = eval_multi_grid(qb, dev, quad=q, r_max=60.0)
        out["x".join(map(str, q))] = dict(E=e["E"], B=e["B"], E_over_B=e["E"] / e["B"], virial=e["virial"], dE=e["dE_grids"])
    res[tag] = out
    print(f"{tag:42s} rec48 E={r['E']:.6f} B={r['B']:.6f} | " + " | ".join(f"{k}: E={v['E']:.6f} B={v['B']:.6f} E/B={v['E_over_B']:.6f} dE={v['dE']:.1e}" for k, v in out.items() if isinstance(v, dict)), flush=True)
    del net; torch.cuda.empty_cache()
json.dump(res, open(os.path.join(DATA, "reeval_b1_models.json"), "w"), indent=1); print("DONE", flush=True)
