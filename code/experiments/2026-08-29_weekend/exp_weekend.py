# -*- coding: utf-8 -*-
"""
anchored トラックの追加の run。各ステージは独立・結果は run 単位でキャッシュ。

ステージ(どれも anchored トラック):
  fairness_lr  : 4 つのネットワーク × 学習率 5 点 × 3 シード(B=2)→ runs_lr/
  fairness_act : MLP の活性化関数 silu / gelu、幅 3 つ × 3 シード(B=2)→ runs_act/
  fairness_act2: GELU-MLP [64]x4 の学習率 5e-3 と 1e-2 → runs_act/
  fairness_act3: GELU-MLP の幅 2〜32(深さ 4)→ runs_act/
  gelu_tb      : GELU-MLP [64]x4 を 30 s と 120 s で止める → runs_gelu_tb/
  pykan_b1     : pykan の B=1 → runs_pykan_b1/
  seeds5       : anchored 掃引の全構成にシード 3, 4 を足す(B=1, 2)→ 2026-08-27_B2_arch_sweep/runs/
                 (論文が使う B=2 のシード 3, 4 の 96 本は、2026-09-11_refit_v2 の run)
  kan_grid     : KAN の格子 G = 3 と 10 → runs_grid/
"""
import os, sys, json, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B2DIR = os.path.join(HERE, "..", "2026-08-27_B2_arch_sweep", "runs")


def run(kind, w, d, s, outdir, label="", **kw):
    try:
        r = train_one(kind, w, d, seed=s, device=device, outdir=outdir,
                      verbose=False, **kw)
        print(f"[{label}] {r['tag']:52s} E={r['E']:.5f} B={r['B']:.5f} "
              f"virial={r['virial']:.4f} ΔE={r['dE_grids']:.1e} "
              f"{r['time_train_sec']:.0f}s", flush=True)
        return r
    except Exception as e:
        print(f"[{label}] {kind} [{w}]x{d} s{s}: 失敗 {str(e)[:150]}", flush=True)
        torch.cuda.empty_cache()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    args = ap.parse_args()
    t0 = time.time()

    if args.stage == "fairness_lr":
        # 代表構成 × LR 5 点 × 3 シード(hard, B=2)。2e-3 は対照として再掲。
        out = os.path.join(HERE, "runs_lr")
        for (m, w, d) in [("mlp", 64, 4), ("mlp", 32, 4),
                          ("effkan", 16, 4), ("effkan", 32, 3)]:
            for lr in (5e-4, 1e-3, 2e-3, 5e-3, 1e-2):
                for s in (0, 1, 2):
                    run(m, w, d, s, out, "lr", track="hard", B=2, lr=lr)

    elif args.stage == "fairness_act":
        out = os.path.join(HERE, "runs_act")
        for act in ("silu", "gelu"):
            for (w, d) in [(64, 4), (32, 4), (16, 4)]:
                for s in (0, 1, 2):
                    run("mlp", w, d, s, out, "act", track="hard", B=2,
                        mlp_act=act)

    elif args.stage == "fairness_act2":
        # GELU-MLP [64]x4 の学習率 5e-3 と 1e-2
        out = os.path.join(HERE, "runs_act")
        for lr in (5e-3, 1e-2):
            for s in (0, 1, 2):
                run("mlp", 64, 4, s, out, "act2", track="hard", B=2,
                    mlp_act="gelu", lr=lr)

    elif args.stage == "fairness_act3":
        # GELU-MLP のパラメータ効率(幅掃引、d=4)
        out = os.path.join(HERE, "runs_act")
        for w in (2, 4, 8, 16, 32):
            for s in (0, 1, 2):
                run("mlp", w, 4, s, out, "geluPar", track="hard", B=2,
                    mlp_act="gelu")

    elif args.stage == "gelu_tb":
        # 時間予算(30 s、120 s)の GELU-MLP
        out = os.path.join(HERE, "runs_gelu_tb")
        for budget in (30, 120):
            for s in (0, 1, 2):
                run("mlp", 64, 4, s, out, "geluTB", track="hard", B=2,
                    mlp_act="gelu", epochs=100000, time_budget_sec=budget)

    elif args.stage == "pykan_b1":
        out = os.path.join(HERE, "runs_pykan_b1")
        for (w, d) in [(8, 2), (4, 3)]:
            for s in (0, 1, 2):
                run("pykan", w, d, s, out, "pykanB1", track="hard", B=1)

    elif args.stage == "seeds5":
        # B2 スイープの共通集合(幅6×深さ4)を mlp/effkan とも 5 シードに拡張
        for Bv in (2, 1):
            for m in ("mlp", "effkan"):
                for d in (1, 2, 3, 4):
                    for w in (2, 4, 8, 16, 32, 64):
                        for s in (3, 4):
                            run(m, w, d, s, B2DIR, f"s5B{Bv}",
                                track="hard", B=Bv)

    elif args.stage == "kan_grid":
        out = os.path.join(HERE, "runs_grid")
        for g in (3, 10):
            for (w, d) in [(16, 4), (8, 2)]:
                for s in (0, 1, 2):
                    run("effkan", w, d, s, out, "grid", track="hard", B=2,
                        kan_grid=g)

    else:
        raise SystemExit(f"unknown stage {args.stage}")

    print(f"STAGE_{args.stage.upper()}_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
