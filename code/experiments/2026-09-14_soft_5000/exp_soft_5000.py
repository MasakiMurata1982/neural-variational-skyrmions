# -*- coding: utf-8 -*-
"""表 7 の群 1・群 2: soft トラック(遠方を構成で真空に固定、乱数初期化、seed 0)の 8 本を 5000 エポック走らせる。
runs_w200/ は w_topo=200、runs_w1000/ は w_topo=1000 + 原点アンカー w_origin=10。
設定は、配布物の記録(data/run_records/2026-09-14_soft_5000/)の config から読む。出力: このフォルダの runs_w200/、runs_w1000/。
目的: 群 4(打ち切り球)で B が動き始めた 3,665 エポックを超えても、soft 表現では B が 0 のままであることを確認する。"""
import os, sys, json, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, DEFAULTS
dev = torch.device("cuda"); SRC = os.path.abspath(os.path.join(SKY, "..", "data", "run_records", "2026-09-14_soft_5000"))
TAGS = ["soft_B1_effkan_w16_d4_ep5000_s0_g5", "soft_B2_effkan_w16_d4_ep5000_s0_g5", "soft_B1_mlp_w64_d4_ep5000_s0", "soft_B2_mlp_w64_d4_ep5000_s0"]
t0 = time.time(); i = 0
for out in ("runs_w200", "runs_w1000"):
    for tag in TAGS:
        i += 1; r = json.load(open(os.path.join(SRC, out, tag + ".json"))); cfg = dict(r["config"])
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in DEFAULTS or k == "save_model"}
        kw["epochs"] = 5000; kw["log_every"] = 100000
        n = train_one(r["kind"], r["width"], r["depth"], seed=r["seed"], device=dev, outdir=os.path.join(HERE, out), verbose=False, **kw)
        print(f"[{i}/8] {out:10s} w_topo={cfg['w_topo']:6.0f} w_origin={cfg.get('w_origin', 0)} {n['tag']:40s} E={n['E']:.5f} B={n['B']:+.6f} {n['time_train_sec']:.0f}s [{time.time()-t0:.0f}s]", flush=True)
print(f"SOFT5000_DONE {time.time()-t0:.0f}s", flush=True)
