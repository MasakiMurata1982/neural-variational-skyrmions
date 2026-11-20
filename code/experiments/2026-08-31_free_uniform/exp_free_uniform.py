# -*- coding: utf-8 -*-
"""
free トラック(ansatz-free)の統一プロトコル網羅比較。
論文の表 2 の全セルを同一条件で走らせる。

統一設計:
  B ∈ {1,2,3,4} × モデル {effKAN[16]x4, GELU-MLP[64]x4, GELU-MLP[48]x4(パラメータ整合)}
  × シード {0..9}
  エポック 20000 / 事前回帰 8000ep lr1e-3 / 塊配置 spread=1.5, clamp=2.5
  損失 E + 1000(B−n)² + 50·遠方真空 / GL 求積 / 多重格子評価
計 120 run。キャッシュ式(中断・再実行安全)。
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
for Bt in (1, 2, 3, 4):
    for (kind, w, d, extra) in ARMS:
        for s in range(10):
            try:
                r = train_one(kind, w, d, seed=s, device=device, outdir=OUT,
                              verbose=False, B=Bt, n_lumps=Bt,
                              **COMMON, **extra)
                done += 1
                print(f"[{done:3d}/120] {r['tag']:52s} E={r['E']:.5f} "
                      f"B={r['B']:.5f} virial={r['virial']:.4f} "
                      f"ΔE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s "
                      f"(累計 {time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[FAIL] {kind} w{w} B={Bt} s{s}: {str(e)[:150]}", flush=True)
                torch.cuda.empty_cache()
print(f"FREE_UNIFORM_DONE {time.time()-t0:.0f}s", flush=True)
