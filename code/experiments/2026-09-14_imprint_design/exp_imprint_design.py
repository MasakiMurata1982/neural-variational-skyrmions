# -*- coding: utf-8 -*-
"""imprinting の設計実験(35 本)。操作変数は刷り込み損失の目標値(直近 50 ステップ平均で早期停止、上限 30,000 ステップ)。
  要因計画 30 本: 目標 {1e-3, 1e-4, 1e-5} × {KAN[16]x4 (effkan), GELU-MLP[64]x4} × seed 0-4
  到達可能性 5 本: tanh-MLP[64]x4、目標 1e-5、seed 0-4(上限まで回して未到達ならそれを記録)
共通設定 = 統一プロトコル: free、B_target=2、epochs 20000、lr 2e-3(cosine、eta_min 0.02 lr)、w_topo 1000、w_far 50(r_far_lo 30)、r_max 60、
  quad (32,20,24)、quad_eval (48,28,36)、n_lumps 2、lump_spread 1.5、lump_max_r 2.5、kan_grid 5、pretrain_lr 1e-3、pretrain_epochs(上限)30000。
記録: <tag>.json の lumps に pretrain_target_loss / pretrain_epochs_used / pretrain_reached / pretrain_loss / imprint_loss_eval(固定大バッチ、横軸)。
出力先: tag は幅・深さ・seed で決まるので、目標値ごとにフォルダ runs_target0.001 / runs_target0.0001 / runs_target1e-05 に分ける。
実行順: tanh 5 本 → GELU 15 本 → KAN 15 本(短いものから)。
--tanh-extra: tanh-MLP[64]x4 の 15 本(目標 1e-3 × 5、1e-4 × 5、既定 8000 ステップ × 5 → runs_default8000/)。"""
import os, sys, time, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one
dev = torch.device("cuda")
COMMON = dict(track="free", B=2, n_lumps=2, w_topo=1000.0, w_origin=0.0, w_far=50.0, r_far_lo=30.0, r_max=60.0,
              epochs=20000, lr=2e-3, quad=(32, 20, 24), quad_eval=(48, 28, 36),
              lump_spread=1.5, lump_max_r=2.5, kan_grid=5, pretrain_lr=1e-3, pretrain_epochs=30000, log_every=100000)
if "--tanh-extra" in sys.argv:
    # tanh-MLP[64]x4 の 15 本: 目標 1e-3、目標 1e-4(上限 30000)、既定(目標なし、8000 ステップ、lr 1e-3)。
    PLAN = [("mlp", 64, 4, dict(mlp_act="tanh"), 1e-3, "tanh 1e-3"), ("mlp", 64, 4, dict(mlp_act="tanh"), 1e-4, "tanh 1e-4"),
            ("mlp", 64, 4, dict(mlp_act="tanh", pretrain_epochs=8000), None, "tanh default")]
else:
    PLAN = [("mlp", 64, 4, dict(mlp_act="tanh"), 1e-5, "tanh reachability")]
    PLAN += [("mlp", 64, 4, dict(mlp_act="gelu"), t, "factorial") for t in (1e-3, 1e-4, 1e-5)]
    PLAN += [("effkan", 16, 4, {}, t, "factorial") for t in (1e-3, 1e-4, 1e-5)]
SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(HERE, "runs"); t0 = time.time(); i = 0; n_total = len(PLAN) * len(SEEDS)
for kind, w, d, extra, tgt, role in PLAN:
    sub = OUT + (f"_target{tgt:g}" if tgt is not None else "_default8000")   # 目標なし(8000 ステップ)は runs_default8000/
    for s in SEEDS:
        i += 1; tag_guess = f"free_B2_{kind}_w{w}_d{d}"
        if any(f.endswith(f"_s{s}" + ("_g5" if kind == "effkan" else "_" + extra.get("mlp_act", "tanh")) + ".json") and f.startswith(tag_guess) for f in os.listdir(sub) if os.path.isdir(sub)) if os.path.isdir(sub) else False:
            print(f"[{i}/{n_total}] skip existing {sub} seed {s}", flush=True); continue
        kw = dict(COMMON); kw.update(extra)   # extra の pretrain_epochs(既定の 8000)が COMMON の上限 30000 を上書きする
        r = train_one(kind, w, d, seed=s, device=dev, outdir=sub, verbose=False, pretrain_target_loss=tgt, **kw)
        L = r["lumps"]
        print(f"[{i}/{n_total}] {role:18s} target={tgt if tgt is None else f'{tgt:g}'} {r['tag']:40s} pre_used={L['pretrain_epochs_used']:5d} reached={L['pretrain_reached']} "
              f"loss_single={L['pretrain_loss']:.2e} loss_eval={L['imprint_loss_eval']:.2e} | E={r['E']:.5f} B={r['B']:.5f} vir={r['virial']:.4f} dE={r['dE_grids']:.1e} {r['time_train_sec']:.0f}s [{time.time()-t0:.0f}s]", flush=True)
print(f"IMPRINT_DESIGN_DONE n={i} {time.time()-t0:.0f}s", flush=True)
