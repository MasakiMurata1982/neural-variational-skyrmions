# -*- coding: utf-8 -*-
"""
anchored(hard)トラックでのアーキテクチャ比較(anchored 掃引)。
モデル: MLP / effKAN / FastKAN(全構成) + pykan(参照点のみ)
グリッド: 幅 (2,4,8,16,32,64) × 深さ (1,2,3,4) × シード (0,1,2) × B∈{1,2}
記録: E, E2/E4, |B-n|, ΔE(多重格子), 学習時間, ピークメモリ
→ 図 2 と表 1 の run。

実行例:
  python3 exp_b2_sweep.py --stage main --B 2
  python3 exp_b2_sweep.py --stage pykan_ref --B 2
"""
import os, sys, json, time, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
from train import train_one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = os.path.join(HERE, "runs")

WIDTHS = (2, 4, 8, 16, 32, 64)
DEPTHS = (1, 2, 3, 4)
SEEDS = (0, 1, 2)
PYKAN_REF = [(8, 2), (4, 3), (16, 4)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main", choices=["main", "pykan_ref"])
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--models", default="mlp,effkan,fastkan")
    args = ap.parse_args()

    t0 = time.time()
    rows, failed = [], []
    if args.stage == "main":
        models = args.models.split(",")
        combos = [(m, w, d, s) for m in models for d in DEPTHS
                  for w in WIDTHS for s in SEEDS]
    else:
        combos = [("pykan", w, d, s) for (w, d) in PYKAN_REF for s in SEEDS]

    print(f"stage={args.stage}  B={args.B}  組み合わせ数={len(combos)}")
    for i, (m, w, d, s) in enumerate(combos):
        try:
            r = train_one(m, w, d, seed=s, device=device, outdir=OUTDIR,
                          verbose=False, track="hard", B=args.B)
            rows.append(r)
            print(f"[{i+1:3d}/{len(combos)}] {r['tag']:44s} E={r['E']:.5f} "
                  f"E2/E4={r['virial']:.4f} ΔE={r['dE_grids']:.1e} "
                  f"{r['time_train_sec']:.0f}s  (累計 {time.time()-t0:.0f}s)")
        except torch.cuda.OutOfMemoryError:
            failed.append(dict(model=m, width=w, depth=d, seed=s, error="OOM"))
            print(f"[{i+1:3d}/{len(combos)}] {m} [{w}]x{d} s{s}: OOM — スキップ")
            torch.cuda.empty_cache()
        except Exception as e:
            failed.append(dict(model=m, width=w, depth=d, seed=s, error=str(e)[:200]))
            print(f"[{i+1:3d}/{len(combos)}] {m} [{w}]x{d} s{s}: 失敗 {e}")
            torch.cuda.empty_cache()

    out = dict(stage=args.stage, B=args.B, n_done=len(rows), failed=failed,
               total_sec=time.time() - t0)
    json.dump(out, open(os.path.join(
        HERE, f"sweep_{args.stage}_B{args.B}_meta.json"), "w"),
        indent=1, ensure_ascii=False)
    print(f"\n完了: {len(rows)} 構成 / 失敗 {len(failed)} / {time.time()-t0:.0f} 秒")

if __name__ == "__main__":
    main()
