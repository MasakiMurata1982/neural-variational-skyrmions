# -*- coding: utf-8 -*-
"""7.4 節: 統一プロトコルの 3 本を同じシードで走らせ直す(GPU の非決定性の実測)。
force=True で別のフォルダ(runs_uniform_repro/)に学習し直し、配布物の記録(data/run_records/2026-08-31_free_uniform/runs_uniform/)と
E, B, ビリアル比, 成功判定を比べる。出力: このフォルダの runs_uniform_repro/、repro_check.json。"""
import os, sys, json
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORIG = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "run_records", "2026-08-31_free_uniform", "runs_uniform"))
OUT = os.path.join(HERE, "runs_uniform_repro")
COMMON = dict(track="free", w_topo=1000.0, w_origin=0.0, w_far=50.0,
              epochs=20000, pretrain_epochs=8000, pretrain_lr=1e-3,
              lump_spread=1.5, lump_max_r=2.5, save_model=False, log_every=10000)
E_TABLE = {1: 1.2322, 2: 2.3582, 3: 3.4386, 4: 4.4804}
def ok(r):
    Bt = r["B_target"]
    return (abs(r["B"] - Bt) < 1e-3 and abs(r["E"] - E_TABLE[Bt]) / E_TABLE[Bt] < 0.01
            and abs(r["virial"] - 1) < 0.05 and r["dE_grids"] < 1e-2)

CASES = [("effkan", 16, 2, 3, {}), ("mlp", 64, 3, 4, dict(mlp_act="gelu")),
         ("effkan", 16, 4, 1, {})]
res = []
for kind, w, Bt, s, extra in CASES:
    r = train_one(kind, w, 4, seed=s, device=device, outdir=OUT, verbose=False,
                  force=True, B=Bt, n_lumps=Bt, **COMMON, **extra)
    o = json.load(open(os.path.join(ORIG, r["tag"] + ".json")))
    d = dict(tag=r["tag"], E_orig=o["E"], E_rerun=r["E"], dE=r["E"] - o["E"],
             B_orig=o["B"], B_rerun=r["B"], vir_orig=o["virial"], vir_rerun=r["virial"],
             ok_orig=ok(o), ok_rerun=ok(r),
             pretrain_orig=o["lumps"]["pretrain_loss"], pretrain_rerun=r["lumps"]["pretrain_loss"])
    res.append(d)
    print(f"{r['tag']:48s} E {o['E']:.6f} -> {r['E']:.6f} (dE={d['dE']:+.1e})  "
          f"B {o['B']:.5f} -> {r['B']:.5f}  virial {o['virial']:.4f} -> {r['virial']:.4f}  "
          f"success {d['ok_orig']} -> {d['ok_rerun']}  pretrain {d['pretrain_orig']:.1e} -> {d['pretrain_rerun']:.1e}",
          flush=True)
json.dump(res, open(os.path.join(HERE, "repro_check.json"), "w"), indent=1)
print("REPRO_DONE")
