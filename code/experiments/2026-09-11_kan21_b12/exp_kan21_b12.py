# -*- coding: utf-8 -*-
"""統一プロトコルの KAN[21]x4 腕を B=1, 2 に拡張(各 10 placement、seed 0..9、計 20 run)。
設定は code/experiments/2026-08-31_free_uniform/exp_free_uniform_kan21.py(B=3,4)と完全同一で電荷のみ変更。"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one
device = torch.device("cuda"); OUT = os.path.join(HERE, "runs")
COMMON = dict(track="free", w_topo=1000.0, w_origin=0.0, w_far=50.0,
              epochs=20000, pretrain_epochs=8000, pretrain_lr=1e-3,
              lump_spread=1.5, lump_max_r=2.5, save_model=True, log_every=10000)
t0 = time.time(); done = 0
for Bt in (1, 2):
    for s in range(10):
        try:
            r = train_one("effkan", 21, 4, seed=s, device=device, outdir=OUT, verbose=False, B=Bt, n_lumps=Bt, **COMMON)
            done += 1
            print(f"[{done:2d}/20] {r['tag']:44s} params={r['n_params']} E={r['E']:.5f} B={r['B']:.5f} virial={r['virial']:.4f} "
                  f"ΔE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s (累計 {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[FAIL] effkan21 B={Bt} s{s}: {str(e)[:150]}", flush=True); torch.cuda.empty_cache()
print(f"KAN21_B12_DONE {time.time()-t0:.0f}s", flush=True)
