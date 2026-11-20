# -*- coding: utf-8 -*-
"""参照値を下回る anchored FastKAN B=1 run(上位 3 本)と対照(effKAN [16]x1 s3、MLP [64]x2 s3)を
同設定・同シードで再学習し、48×28×36 / 80×44×60 / 120×60×80 で再評価する。hedgehog 3D 値は 1.2314415。
記録の 5 本は、train_profile_1d の再スタート閾値が 1e-9 の版で作られた。bit 単位で再現するには、physics.py の `if E_s < E_best - 1e-6:` を 1e-9 にする(MANIFEST.md の Code versions)。"""
import os, sys, json, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, get_base_profile
from physics import build_q_hard, eval_multi_grid
from models import build_net
dev = torch.device("cuda"); OUT = os.path.join(HERE, "runs_b1_regrid"); res = {}
torch.manual_seed(0)
pmodel, E1d, _ = get_base_profile(1, dev); print(f"1D profile B=1 E0={E1d:.7f}", flush=True)
CANDS = [("fastkan", 32, 3, 1), ("fastkan", 32, 4, 1), ("fastkan", 8, 4, 2), ("effkan", 16, 1, 3), ("mlp", 64, 2, 3)]
SW = os.path.abspath(os.path.join(SKY, "..", "data", "run_records", "2026-08-27_B2_arch_sweep", "runs"))   # 元の run の記録
for kind, w, d, s in CANDS:
    tag_o = f"hard_B1_{kind}_w{w}_d{d}_ep1000_s{s}" + ("_g5" if kind != "mlp" else "")
    o = json.load(open(os.path.join(SW, tag_o + ".json")))
    t0 = time.time()
    r = train_one(kind, w, d, seed=s, device=dev, outdir=OUT, verbose=False, force=True,
                  track="hard", B=1, epochs=1000, kan_grid=5, save_model=True, log_every=100000)
    net = build_net(kind, w, d, dev, 5, zero_init=True, seed=s, mlp_act="tanh")
    net.load_state_dict(torch.load(os.path.join(OUT, r["tag"] + "_model.pt"), map_location=dev)); net.eval()
    qb = lambda xi: build_q_hard(xi, net, pmodel, B=1)
    ev = {}
    for q in [(48, 28, 36), (80, 44, 60), (120, 60, 80)]:
        e = eval_multi_grid(qb, dev, quad=q, r_max=60.0); ev[f"{q[0]}x{q[1]}x{q[2]}"] = dict(E=e["E"], B=e["B"], virial=e["virial"], dE=e["dE_grids"])
    res[tag_o] = dict(orig=dict(E=o["E"], B=o["B"], dE=o["dE_grids"], virial=o["virial"]), rerun=ev, time=time.time() - t0)
    print(f"{tag_o:42s} orig48 E={o['E']:.6f} dE={o['dE_grids']:.1e} | rerun " +
          " | ".join(f"{k}: E={v['E']:.6f} B={v['B']:.6f} dE={v['dE']:.1e}" for k, v in ev.items()) + f" [{time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(os.path.join(HERE, "retrain_b1_regrid.json"), "w"), indent=1)
    del net; torch.cuda.empty_cache()
print("DONE", flush=True)
