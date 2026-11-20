# -*- coding: utf-8 -*-
"""7.4 節: 同じコード・同じシードで走らせ直したときの再現幅(GPU の非決定性)を測る。
対象は runs_refit_v2 の 5 本(tanh-MLP [8]x4 s4、KAN [2]x4 s3、tanh-MLP [32]x2 s4、KAN [16]x1 s3、KAN [64]x3 s3)。それぞれ 2 回走らせる。
比べる相手は、配布物の記録(data/run_records/2026-09-11_refit_v2/runs_refit_v2/)。出力: このフォルダの runs_determinism_rep{1,2}/、determinism_check.json。"""
import os, sys, json, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, get_base_profile
dev = torch.device("cuda")
RECORDS = os.path.abspath(os.path.join(SKY, "..", "data", "run_records", "2026-09-11_refit_v2", "runs_refit_v2"))
torch.manual_seed(0); pmodel, E1d, _ = get_base_profile(2, dev); print(f"1D profile E0={E1d:.7f}", flush=True)
TARGETS = [("mlp", 8, 4, 4), ("effkan", 2, 4, 3), ("mlp", 32, 2, 4), ("effkan", 16, 1, 3), ("effkan", 64, 3, 3)]
out = []; t0 = time.time()
for rep in (1, 2):
    for kind, w, d, s in TARGETS:
        r = train_one(kind, w, d, seed=s, device=dev, outdir=os.path.join(HERE, f"runs_determinism_rep{rep}"), verbose=False,
                      track="hard", B=2, epochs=1000, lr=2e-3, kan_grid=5, log_every=100000)
        ref = json.load(open(os.path.join(RECORDS, r["tag"] + ".json")))
        out.append(dict(rep=rep, tag=r["tag"], E=r["E"], E_v2=ref["E"]))
        print(f"rep{rep} {r['tag']:44s} E={r['E']:.6f}  v2={ref['E']:.6f} (d={r['E']-ref['E']:+.1e}) [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(HERE, "determinism_check.json"), "w"), indent=1)
print(f"DETERMINISM_DONE {time.time()-t0:.0f}s", flush=True)
