# -*- coding: utf-8 -*-
"""
統一プロトコルのシード拡張(B=3,4 のみ、シード 10..19)。

成功率の差の検出力を上げるため、B=3,4 のセルを 20 シードに拡張する。
設定は exp_free_uniform.py と完全同一(プロトコル変更なし・シード追加のみ)。

計 60 run(B∈{3,4} × 3アーム × シード10..19)。キャッシュ式。
"""
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(HERE, "runs_uniform")

COMMON = dict(track="free", w_topo=1000.0, w_origin=0.0, w_far=50.0,
              epochs=20000, pretrain_epochs=8000, pretrain_lr=1e-3,
              lump_spread=1.5, lump_max_r=2.5,
              save_model=True, log_every=10000)

ARMS = [("effkan", 16, 4, {}),
        ("mlp", 64, 4, dict(mlp_act="gelu")),
        ("mlp", 48, 4, dict(mlp_act="gelu"))]

t0 = time.time()
done = 0
for Bt in (3, 4):
    for (kind, w, d, extra) in ARMS:
        for s in range(10, 20):
            try:
                r = train_one(kind, w, d, seed=s, device=device, outdir=OUT,
                              verbose=False, B=Bt, n_lumps=Bt,
                              **COMMON, **extra)
                done += 1
                print(f"[{done:3d}/60] {r['tag']:52s} E={r['E']:.5f} "
                      f"B={r['B']:.5f} virial={r['virial']:.4f} "
                      f"ΔE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s "
                      f"(累計 {time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[FAIL] {kind} w{w} B={Bt} s{s}: {str(e)[:150]}", flush=True)
                torch.cuda.empty_cache()
print(f"FREE_UNIFORM_EXT_DONE {time.time()-t0:.0f}s", flush=True)
