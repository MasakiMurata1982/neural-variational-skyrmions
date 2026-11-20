# -*- coding: utf-8 -*-
"""
表 7 の群 3・群 4 と、図 5 の刷り込みありの run(free トラック)。
stage=pilot_free   : effkan [16]x4 / mlp [64]x4 × B∈{1,2} × seed 0(4 run、5000 ep、原点アンカー w_origin=10)→ runs_v2/
stage=pilot_free2  : 同じ 4 run で、電荷ペナルティを打ち切り球 r<=12 で測る → runs_free2/
stage=pilot_seeded : B=1 の塊 2 個の積に事前回帰してから緩和(B=2、seed 0, 1、10000 ep)→ runs_seeded2/
                     論文が使うのは effkan [16]x4 seed 1 の 1 本(図 5)。
"""
import os, sys, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# runs_v2/: w_topo=1000 + 原点アンカー U(0)=-1 (w_origin=10)
OUTDIR = os.path.join(HERE, "runs_v2")
W_TOPO, W_ORIGIN = 1000.0, 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["pilot_free", "pilot_free2", "pilot_seeded"])
    args = ap.parse_args()

    # free2 : B ペナルティを打ち切り球 r<=12 で測定(電荷注入チャネルを開ける)
    # seeded: ランダム配置・ランダム iso 回転の粗い B=1 塊 ×n の積アンザッツに
    #         事前回帰してから緩和(トポロジーのみ持ち込み、対称性は含まない)
    track = "free"
    extra = {}
    global OUTDIR
    if args.stage.endswith("free2"):
        extra = dict(r_max_topo=12.0)
        OUTDIR = os.path.join(HERE, "runs_free2")
    if args.stage.endswith("seeded"):
        OUTDIR = os.path.join(HERE, "runs_seeded2")   # 生出力回帰+10000ep
        combos = [(m, w, d, s, Bt)
                  for (m, w, d) in [("effkan", 16, 4), ("mlp", 64, 4)]
                  for Bt in (2,) for s in (0, 1)]
    else:
        combos = [(m, w, d, s, Bt)
                  for (m, w, d) in [("effkan", 16, 4), ("mlp", 64, 4)]
                  for Bt in (1, 2) for s in (0,)]
    epochs = 5000
    if "seeded" in args.stage:
        extra = dict(extra)
        extra["n_lumps"] = -1  # 実行時に B_target で置換(下のループ内)
        epochs = 10000         # 塊の合体・緩和は遅いため長めに

    print(f"stage={args.stage}  組み合わせ数={len(combos)}  epochs={epochs}")
    t0 = time.time()
    for i, (m, w, d, s, Bt) in enumerate(combos):
        try:
            kw = dict(track=track, B=Bt, epochs=epochs, w_topo=W_TOPO,
                      w_origin=W_ORIGIN, soft_scale=3.0, log_every=500,
                      save_model=True, **extra)
            if kw.get("n_lumps") == -1:
                kw["n_lumps"] = Bt
                kw["w_origin"] = 0.0   # 塊の積が既に位相を持つのでアンカー不要
                if m == "mlp":
                    # MLP は事前回帰の適合が浅いと巻きがほどける
                    kw["pretrain_epochs"] = 8000
                    kw["pretrain_lr"] = 1e-3
            if track == "free":
                kw["w_far"] = 50.0
            r = train_one(m, w, d, seed=s, device=device, outdir=OUTDIR,
                          verbose=True, **kw)
            print(f"[{i+1}/{len(combos)}] {r['tag']}: E={r['E']:.5f} B={r['B']:.5f} "
                  f"virial={r['virial']:.4f} ΔE={r['dE_grids']:.1e} "
                  f"({time.time()-t0:.0f}s 経過)")
        except Exception as e:
            print(f"[{i+1}/{len(combos)}] {m} [{w}]x{d} s{s} B={Bt}: 失敗 {e}")
            torch.cuda.empty_cache()
    print(f"\n完了 {time.time()-t0:.0f} 秒")


if __name__ == "__main__":
    main()
