# -*- coding: utf-8 -*-
"""anchored 掃引の B=2、シード 3・4 の 96 本(effkan 48 + tanh-MLP 48)を、現行コード(train_profile_1d の再スタート閾値 1e-6)で走らせる。
論文の図 2 と表 1 が使うのは、この 96 本である。
走らせる構成とシードは、配布物の記録(data/run_records/2026-09-11_refit_v2/runs_refit_v2/)から読む。出力: このフォルダの runs_refit_v2/、refit_v2.json。
プロファイルはウォームアップ(manual_seed(0) → get_base_profile)してから学習する。"""
import os, sys, json, glob, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(SKY, "src"))
from train import train_one, get_base_profile
dev = torch.device("cuda"); OUT = os.path.join(HERE, "runs_refit_v2")
RECORDS = os.path.abspath(os.path.join(SKY, "..", "data", "run_records", "2026-09-11_refit_v2", "runs_refit_v2"))
todo = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RECORDS, "*.json")))]
print(f"refit_v2 targets: {len(todo)}", flush=True)
torch.manual_seed(0); pmodel, E1d, _ = get_base_profile(2, dev); print(f"1D profile E0={E1d:.7f}", flush=True)
OUTJ = os.path.join(HERE, "refit_v2.json"); log = json.load(open(OUTJ)) if os.path.exists(OUTJ) else []
done = {d["tag"] for d in log}
t0 = time.time()
for i, r in enumerate(todo, 1):
    cfg = r["config"]
    if r["tag"] in done: continue
    try:
        n = train_one(r["kind"], r["width"], r["depth"], seed=r["seed"], device=dev, outdir=OUT, verbose=False,
                      track="hard", B=2, epochs=cfg["epochs"], lr=cfg.get("lr", 2e-3), kan_grid=cfg.get("kan_grid", 5), log_every=100000)
        log.append(dict(tag=n["tag"], kind=r["kind"], width=r["width"], depth=r["depth"], seed=r["seed"], n_params=n["n_params"],
                        E_refit_v2=n["E"], B_refit_v2=n["B"], dE_refit_v2=n["dE_grids"], time_train_sec=n["time_train_sec"]))
        print(f"[{i:3d}/{len(todo)}] {n['tag']:48s} E {n['E']:.6f} (record {r['E']:.6f}, diff {n['E']-r['E']:+.1e})  B {n['B']:.6f}  [{time.time()-t0:.0f}s]", flush=True)
    except Exception as e:
        print(f"[{i:3d}/{len(todo)}] FAIL {r['tag']}: {str(e)[:120]}", flush=True); torch.cuda.empty_cache()
    json.dump(log, open(OUTJ, "w"), indent=1)
print(f"REFIT_V2_DONE n={len(log)} {time.time()-t0:.0f}s", flush=True)
