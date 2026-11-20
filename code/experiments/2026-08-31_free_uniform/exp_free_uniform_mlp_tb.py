# -*- coding: utf-8 -*-
"""等・壁時計比較の MLP 側: GELU-MLP[64]x4 に KAN[16]x4 と同じ本学習時間を与える。
KAN の 1 run ≈ 700 s のうち事前回帰 ≈ 40 s → 本学習 ≈ 660 s。
MLP[64] は 1 エポック ≈ 8 ms なので 660 s で ≈ 8 万エポック(epochs は上限として 100000)。
time_budget_sec モード(最初の 10 エポックで速度を実測 → cosine スケジュールを構成)。
B=3,4 × シード 0..19、事前回帰・損失・求積・評価は統一プロトコルと同一。計 40 run。"""
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(HERE, "runs_uniform_mlp_tb")
COMMON = dict(track="free", w_topo=1000.0, w_origin=0.0, w_far=50.0,
              epochs=100000, pretrain_epochs=8000, pretrain_lr=1e-3,
              lump_spread=1.5, lump_max_r=2.5, time_budget_sec=660,
              save_model=True, log_every=100000, mlp_act="gelu")
t0 = time.time(); done = 0
for Bt in (3, 4):
    for s in range(20):
        try:
            r = train_one("mlp", 64, 4, seed=s, device=device, outdir=OUT,
                          verbose=False, B=Bt, n_lumps=Bt, **COMMON)
            done += 1
            print(f"[{done:3d}/40] {r['tag']:56s} E={r['E']:.5f} B={r['B']:.5f} "
                  f"virial={r['virial']:.4f} ΔE={r['dE_grids']:.1e} "
                  f"{r['time_train_sec']:.0f}s (累計 {time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[FAIL] mlp64 tb B={Bt} s{s}: {str(e)[:150]}", flush=True)
            torch.cuda.empty_cache()
print(f"MLP_TB_DONE {time.time()-t0:.0f}s", flush=True)
