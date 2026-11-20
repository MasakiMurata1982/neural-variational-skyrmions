# -*- coding: utf-8 -*-
"""表 1 の 12 構成(4 行 × 3 モデル)を seed 0-4 でモデル保存つきで再実行し、48×28×36 と 80×44×60 で E, B を評価する。
  行 1 shared(lr 2e-3, 1000 ep): KAN (64,3) / tanh (64,4) / GELU (64,4)           → runs_shared/
  行 2 lr tuned(lr 1e-2, 1000 ep): KAN (32,3) / tanh (64,4) / GELU (64,4)          → runs_lr/
  行 3,4 時間予算 30 s / 120 s(epochs 上限 1e5, lr 2e-3): KAN (32,3) / tanh (64,4) / GELU (64,4) → runs_tb/
既存 run と同じ設定(プロファイルはウォームアップ済み)。時間予算 run は他ジョブと重ねない。
評価は学習直後に保存モデルから eval_multi_grid(3 動径写像 a=2.6/3.0/3.4、各 1 回転、seed 1234、r_max 60)。"""
import os, sys, json, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, get_base_profile
from physics import build_q_hard, eval_multi_grid
from models import build_net
dev = torch.device("cuda")
torch.manual_seed(0); pmodel, E1d, _ = get_base_profile(2, dev); print(f"1D profile E0={E1d:.7f}", flush=True)
ROWS = [("shared", "runs_shared", dict(epochs=1000), [("effkan", 64, 3, {}), ("mlp", 64, 4, {}), ("mlp", 64, 4, dict(mlp_act="gelu"))]),
        ("lr_tuned", "runs_lr", dict(epochs=1000, lr=1e-2), [("effkan", 32, 3, {}), ("mlp", 64, 4, {}), ("mlp", 64, 4, dict(mlp_act="gelu"))]),
        ("tb30", "runs_tb", dict(epochs=100000, time_budget_sec=30), [("effkan", 32, 3, {}), ("mlp", 64, 4, {}), ("mlp", 64, 4, dict(mlp_act="gelu"))]),
        ("tb120", "runs_tb", dict(epochs=100000, time_budget_sec=120), [("effkan", 32, 3, {}), ("mlp", 64, 4, {}), ("mlp", 64, 4, dict(mlp_act="gelu"))])]
OUTJ = os.path.join(HERE, "table1_models_eval.json"); res = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
t0 = time.time(); i = 0
for row, sub, kw, confs in ROWS:
    for kind, w, d, extra in confs:
        for s in range(5):
            i += 1
            r = train_one(kind, w, d, seed=s, device=dev, outdir=os.path.join(HERE, sub), verbose=False, track="hard", B=2,
                          save_model=True, log_every=100000, **kw, **extra)
            net = build_net(kind, w, d, dev, 5, zero_init=True, seed=s, mlp_act=extra.get("mlp_act", "tanh"))
            net.load_state_dict(torch.load(os.path.join(HERE, sub, r["tag"] + "_model.pt"), map_location=dev)); net.eval()
            qb = lambda xi: build_q_hard(xi, net, pmodel, B=2)
            ev = {}
            for q in [(48, 28, 36), (80, 44, 60)]:
                e = eval_multi_grid(qb, dev, quad=q, r_max=60.0); ev["x".join(map(str, q))] = dict(E=e["E"], B=e["B"], virial=e["virial"], dE=e["dE_grids"])
            res[f"{sub}/{r['tag']}"] = dict(row=row, kind=kind, width=w, depth=d, seed=s, arch=("mlp_" + extra.get("mlp_act", "tanh")) if kind == "mlp" else kind,
                                            record_E48=r["E"], record_B48=r["B"], eval=ev, time=r["time_train_sec"])
            print(f"[{i:2d}/60] {row:8s} {r['tag']:46s} rec48 E={r['E']:.6f} | 48: E={ev['48x28x36']['E']:.6f} B={ev['48x28x36']['B']:.6f} | 80: E={ev['80x44x60']['E']:.6f} B={ev['80x44x60']['B']:.6f} (Δ={ev['80x44x60']['E']-ev['48x28x36']['E']:+.1e}) [{time.time()-t0:.0f}s]", flush=True)
            json.dump(res, open(OUTJ, "w"), indent=1)
            del net; torch.cuda.empty_cache()
print(f"TABLE1_MODELS_DONE {time.time()-t0:.0f}s", flush=True)
