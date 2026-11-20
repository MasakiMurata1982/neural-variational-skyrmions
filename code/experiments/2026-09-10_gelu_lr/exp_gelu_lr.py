# -*- coding: utf-8 -*-
"""GELU-MLP の学習率掃引を tanh-MLP と同じ形に揃える。
既存(2026-08-29_weekend/runs_act): (64,4) で lr ∈ {2e-3, 5e-3, 1e-2} × 3 シード、(32,4) は lr 2e-3 のみ。
追加: (64,4) lr ∈ {5e-4, 1e-3}、(32,4) lr ∈ {5e-4, 1e-3, 5e-3, 1e-2}、各 3 シード(seed 0-2)= 18 run。
他の設定は sweep と同一(B=2、1000 ep、コサイン減衰、w_topo 200)。プロファイルのウォームアップあり。"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, get_base_profile
dev = torch.device("cuda"); OUT = os.path.join(HERE, "runs")
torch.manual_seed(0); get_base_profile(2, dev)
todo = [(64, 4, lr, s) for lr in (5e-4, 1e-3) for s in (0, 1, 2)] + [(32, 4, lr, s) for lr in (5e-4, 1e-3, 5e-3, 1e-2) for s in (0, 1, 2)]
t0 = time.time()
for i, (w, d, lr, s) in enumerate(todo, 1):
    r = train_one("mlp", w, d, seed=s, device=dev, outdir=OUT, verbose=False, track="hard", B=2, epochs=1000, lr=lr, mlp_act="gelu", log_every=100000)
    print(f"[{i:2d}/{len(todo)}] {r['tag']:46s} E={r['E']:.6f} B={r['B']:.6f} vir={r['virial']:.4f} dE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s [{time.time()-t0:.0f}s]", flush=True)
print(f"GELU_LR_DONE {time.time()-t0:.0f}s", flush=True)
