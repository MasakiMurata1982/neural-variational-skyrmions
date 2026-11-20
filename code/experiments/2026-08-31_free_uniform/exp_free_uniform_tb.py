# -*- coding: utf-8 -*-
"""
free トラック統一プロトコルの「等・壁時計時間」比較(KAN 側)。

表2は等エポック(20000)比較で、KAN[16]x4 は 1 run ≈ 700 s、GELU-MLP は ≈ 176 s。
同じ計算時間での比較のため、KAN の本学習を
MLP の本学習時間(176 s − 事前回帰 14 s ≈ 160 s)に固定して B=3,4 × 20 シードを
再実行する。事前回帰(8000 ep)・損失・求積・評価はすべて統一プロトコルと同一。
time_budget_sec モード: 最初の 10 エポックで速度を実測し、残り時間に合わせて
cosine スケジュールを構成(train.py 参照)。

計 40 run。キャッシュ式。
"""
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(HERE, "runs_uniform_tb")

COMMON = dict(track="free", w_topo=1000.0, w_origin=0.0, w_far=50.0,
              epochs=20000, pretrain_epochs=8000, pretrain_lr=1e-3,
              lump_spread=1.5, lump_max_r=2.5,
              time_budget_sec=160, save_model=True, log_every=10000)

t0 = time.time()
done = 0
for Bt in (3, 4):
    for s in range(20):
        try:
            r = train_one("effkan", 16, 4, seed=s, device=device, outdir=OUT,
                          verbose=False, B=Bt, n_lumps=Bt, **COMMON)
            done += 1
            print(f"[{done:3d}/40] {r['tag']:52s} E={r['E']:.5f} "
                  f"B={r['B']:.5f} virial={r['virial']:.4f} "
                  f"ΔE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s "
                  f"(累計 {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[FAIL] effkan B={Bt} s{s}: {str(e)[:150]}", flush=True)
            torch.cuda.empty_cache()
print(f"FREE_UNIFORM_TB_DONE {time.time()-t0:.0f}s", flush=True)
