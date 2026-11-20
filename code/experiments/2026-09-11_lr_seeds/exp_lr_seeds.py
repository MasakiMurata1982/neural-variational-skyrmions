# -*- coding: utf-8 -*-
"""学習率掃引のシード追加。
 (a) effKAN (32,3) / tanh (64,4) / GELU (64,4) の lr 5e-4, 1e-3, 5e-3 に seed 3, 4(18 run)
 (b) effKAN (32,3) の lr 5e-3, 1e-2 に seed 5-9(10 run)
設定は runs_lr / runs_act と同一(B=2, 1000 ep)。プロファイルはウォームアップ済み。"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, get_base_profile
dev = torch.device("cuda"); OUT = os.path.join(HERE, "runs")
torch.manual_seed(0); get_base_profile(2, dev)
CONF = [("effkan", 32, 3, {}), ("mlp", 64, 4, {}), ("mlp", 64, 4, dict(mlp_act="gelu"))]
todo = [(k, w, d, ex, lr, s) for (k, w, d, ex) in CONF for lr in (5e-4, 1e-3, 5e-3) for s in (3, 4)]
todo += [("effkan", 32, 3, {}, lr, s) for lr in (5e-3, 1e-2) for s in range(5, 10)]
t0 = time.time()
for i, (k, w, d, ex, lr, s) in enumerate(todo, 1):
    r = train_one(k, w, d, seed=s, device=dev, outdir=OUT, verbose=False, track="hard", B=2, epochs=1000, lr=lr, log_every=100000, **ex)
    print(f"[{i:2d}/{len(todo)}] {r['tag']:48s} E={r['E']:.6f} B={r['B']:.6f} vir={r['virial']:.4f} dE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s [{time.time()-t0:.0f}s]", flush=True)
print(f"LR_SEEDS_DONE {time.time()-t0:.0f}s", flush=True)
