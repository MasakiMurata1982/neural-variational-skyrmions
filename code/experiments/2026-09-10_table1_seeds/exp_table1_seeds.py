# -*- coding: utf-8 -*-
"""表 1 の 2〜4 行目を 5 シードに揃えるため、各条件に seed 3, 4 を追加(計 18 run)。
設定は既存 run(2026-08-29_weekend/runs_lr, runs_act, runs_gelu_tb; 2026-08-28_B5_finalize/runs_timebudget)と同一で、シードのみ 3, 4。
  2 行目(最良学習率、1000 ep): effKAN [32]x3 lr 1e-2 / tanh [64]x4 lr 1e-2 / GELU [64]x4 lr 1e-2  → runs_lr/
  3・4 行目(壁時計 30 s / 120 s、epochs 上限 1e5、lr 2e-3): effKAN [32]x3 / tanh [64]x4 / GELU [64]x4    → runs_tb/
プロファイルはウォームアップ済み。時間予算 run は壁時計を測るため、他ジョブと同時実行しないこと。"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, get_base_profile
dev = torch.device("cuda")
torch.manual_seed(0); get_base_profile(2, dev)
CONF = [("effkan", 32, 3, {}), ("mlp", 64, 4, {}), ("mlp", 64, 4, dict(mlp_act="gelu"))]
t0 = time.time(); i = 0
for kind, w, d, extra in CONF:                      # 2 行目
    for s in (3, 4):
        i += 1
        r = train_one(kind, w, d, seed=s, device=dev, outdir=os.path.join(HERE, "runs_lr"), verbose=False,
                      track="hard", B=2, epochs=1000, lr=1e-2, log_every=100000, **extra)
        print(f"[{i:2d}/18] {r['tag']:46s} E={r['E']:.6f} B={r['B']:.6f} vir={r['virial']:.4f} dE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s [{time.time()-t0:.0f}s]", flush=True)
for budget in (30, 120):                            # 3・4 行目
    for kind, w, d, extra in CONF:
        for s in (3, 4):
            i += 1
            r = train_one(kind, w, d, seed=s, device=dev, outdir=os.path.join(HERE, "runs_tb"), verbose=False,
                          track="hard", B=2, epochs=100000, time_budget_sec=budget, log_every=100000, **extra)
            print(f"[{i:2d}/18] {r['tag']:46s} E={r['E']:.6f} B={r['B']:.6f} vir={r['virial']:.4f} dE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s [{time.time()-t0:.0f}s]", flush=True)
print(f"TABLE1_SEEDS_DONE {time.time()-t0:.0f}s", flush=True)
