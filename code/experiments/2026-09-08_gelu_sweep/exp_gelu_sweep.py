# -*- coding: utf-8 -*-
"""GELU-MLP を effKAN・tanh-MLP と同じ土俵にする。
幅 {2,4,8,16,32,64} × 深さ {1,2,3,4} = 24 構成 × 5 シード、B=2、lr 2e-3、コサイン減衰、1000 ep、w_topo=200、
評価 48×28×36 × 3 動径写像(= 2026-08-27_B2_arch_sweep と同一設定、train.py の DEFAULTS)。

深さ 4・seed 0-2 の 18 本は 2026-08-29_weekend/runs_act にあるので作らない。作るのは 102 run(18 構成 × 5 + 深さ 4 の 6 構成 × seed 3,4)。
出力: runs/(既存と同じ JSON 形式、キャッシュ式)。"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one
dev = torch.device("cuda"); OUT = os.path.join(HERE, "runs")
WIDTHS, DEPTHS, SEEDS = (2, 4, 8, 16, 32, 64), (1, 2, 3, 4), (0, 1, 2, 3, 4)
# プロファイルを先に学習してキャッシュする(ウォームアップ)。train_one は manual_seed(seed) の後に get_base_profile を呼ぶが、
# プロセス最初の run ではその学習が乱数を消費し、以降の run(キャッシュ済み)と初期化の条件が変わる。
# 元の実験(sweep、runs_act)でも大半の run はキャッシュ済みプロファイルで走っているので、その条件に揃える。
from train import get_base_profile
torch.manual_seed(0); get_base_profile(2, dev)
todo = []
for d in DEPTHS:
    for w in WIDTHS:
        for s in SEEDS:
            if d == 4 and s in (0, 1, 2):
                continue   # 2026-08-29_weekend/runs_act の既存 run を流用
            todo.append((w, d, s))
print(f"runs={len(todo)}", flush=True)
t0 = time.time()
for i, (w, d, s) in enumerate(todo, 1):
    try:
        r = train_one("mlp", w, d, seed=s, device=dev, outdir=OUT, verbose=False,
                      track="hard", B=2, epochs=1000, mlp_act="gelu", log_every=100000)
        print(f"[{i:3d}/{len(todo)}] {r['tag']:40s} E={r['E']:.6f} B={r['B']:.6f} vir={r['virial']:.4f} dE={r['dE_grids']:.1e} "
              f"{r['time_train_sec']:.0f}s [{time.time()-t0:.0f}s]", flush=True)
    except Exception as e:
        print(f"[{i:3d}/{len(todo)}] FAIL mlp [{w}]x{d} s{s}: {str(e)[:120]}", flush=True); torch.cuda.empty_cache()
print(f"GELU_SWEEP_DONE {time.time()-t0:.0f}s", flush=True)
