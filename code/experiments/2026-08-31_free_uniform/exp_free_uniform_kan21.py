# -*- coding: utf-8 -*-
"""統一プロトコル: KAN を GELU-MLP[64]x4(12,996 パラメータ)側に揃える 2×2 設計の KAN[21]x4 の腕。
effKAN のパラメータ数は Σ in·out·(grid+k+1) = 9·Σ in·out。
  [16]x4: 9·(3·16 + 3·16² + 16·4) = 7,920
  [21]x4: 9·(3·21 + 3·21² + 21·4) = 13,230  (MLP[64] の 12,996 に +1.8% で最近接)
B=3,4 × シード 0..19、他の設定は統一プロトコルと完全同一。計 40 run。"""
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(HERE, "runs_uniform_kan21")
COMMON = dict(track="free", w_topo=1000.0, w_origin=0.0, w_far=50.0,
              epochs=20000, pretrain_epochs=8000, pretrain_lr=1e-3,
              lump_spread=1.5, lump_max_r=2.5, save_model=True, log_every=10000)
t0 = time.time(); done = 0
for Bt in (3, 4):
    for s in range(20):
        try:
            r = train_one("effkan", 21, 4, seed=s, device=device, outdir=OUT,
                          verbose=False, B=Bt, n_lumps=Bt, **COMMON)
            done += 1
            print(f"[{done:3d}/40] {r['tag']:52s} params={r['n_params']} E={r['E']:.5f} "
                  f"B={r['B']:.5f} virial={r['virial']:.4f} ΔE={r['dE_grids']:.1e} "
                  f"{r['time_train_sec']:.0f}s (累計 {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[FAIL] effkan21 B={Bt} s{s}: {str(e)[:150]}", flush=True)
            torch.cuda.empty_cache()
print(f"KAN21_DONE {time.time()-t0:.0f}s", flush=True)
