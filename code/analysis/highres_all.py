# -*- coding: utf-8 -*-
"""表 2(runs_uniform、runs_uniform_kan21)と表 6(runs_uniform_tb、runs_uniform_mlp_tb)の成功 run を、
評価格子 80×44×60(3 動径写像・ランダム回転)で再評価する。表 4・表 9・表 10 の値は、この格子の値である。"""
import os, sys, json, glob, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SRC)
from physics import build_q_free, eval_multi_grid
from models import build_net
device = torch.device("cuda")
E_TABLE = {1: 1.2322, 2: 2.3582, 3: 3.4386, 4: 4.4804}
def ok(r):
    Bt = r["B_target"]
    return (abs(r["B"] - Bt) < 1e-3 and abs(r["E"] - E_TABLE[Bt]) / E_TABLE[Bt] < 0.01
            and abs(r["virial"] - 1) < 0.05 and r["dE_grids"] < 1e-2)
OUTJ = os.path.join(DATA, "highres_all.json")
res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
t0 = time.time()
for sub in ("runs_uniform", "runs_uniform_tb", "runs_uniform_kan21", "runs_uniform_mlp_tb"):
    d = os.path.join(RECORDS, "2026-08-31_free_uniform", sub)
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if not ok(r) or r["tag"] in res:
            continue
        cfg = r.get("config", {})
        net = build_net(r["kind"], r["width"], r["depth"], device, r.get("kan_grid", 5),
                        zero_init=False, mlp_act=cfg.get("mlp_act", "tanh"))
        net.load_state_dict(torch.load(f.replace(".json", "_model.pt"), map_location=device))
        net.eval()
        ev = eval_multi_grid(lambda xi: build_q_free(xi, net), device, quad=(80, 44, 60), r_max=60.0)
        res[r["tag"]] = dict(sub=sub, B_target=r["B_target"], kind=r["kind"], width=r["width"],
                             seed=r["seed"], E48=r["E"], E80=ev["E"], B80=ev["B"],
                             virial80=ev["virial"], dE80=ev["dE_grids"])
        print(f"{r['tag']:52s} E48={r['E']:.6f} E80={ev['E']:.6f} (Δ={ev['E']-r['E']:+.1e}) "
              f"B={ev['B']:.5f} vir={ev['virial']:.4f} dE={ev['dE_grids']:.1e} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(res, open(OUTJ, "w"), indent=1)
        del net; torch.cuda.empty_cache()
print(f"HIGHRES_ALL_DONE n={len(res)} {time.time()-t0:.0f}s", flush=True)
