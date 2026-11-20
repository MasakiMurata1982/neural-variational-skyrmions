# -*- coding: utf-8 -*-
"""
壁時計時間を揃えた KAN と MLP の比較(anchored、B=2)。
  stage=timebudget : 予算 30 s / 120 s × 4 つの構成 × 3 シード。エポック数ではなく時間で打ち切る。→ runs_timebudget/
"""
import os, sys, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["timebudget"])
    args = ap.parse_args()
    t0 = time.time()

    if args.stage == "timebudget":
        outdir = os.path.join(HERE, "runs_timebudget")
        # 代表構成: 各モデルの「時間パレート上」の構成(anchored 掃引 2026-08-27_B2_arch_sweep より)
        CONFIGS = [("mlp", 64, 4), ("mlp", 32, 4),
                   ("effkan", 16, 2), ("effkan", 32, 3)]
        for budget in (30, 120):
            for (m, w, d) in CONFIGS:
                for s in (0, 1, 2):
                    r = train_one(m, w, d, seed=s, device=device, outdir=outdir,
                                  verbose=False, track="hard", B=2,
                                  epochs=100000, time_budget_sec=budget)
                    print(f"[tb{budget:3d}s] {m:7s} [{w}]x{d} s{s}: "
                          f"E={r['E']:.5f} B={r['B']:.5f} "
                          f"virial={r['virial']:.4f} ΔE={r['dE_grids']:.1e} "
                          f"実測{r['time_train_sec']:.0f}s "
                          f"({time.time()-t0:.0f}s 経過)")
    print(f"STAGE_{args.stage.upper()}_DONE  {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
