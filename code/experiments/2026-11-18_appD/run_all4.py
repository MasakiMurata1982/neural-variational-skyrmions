# -*- coding: utf-8 -*-
"""付録 D(図 6): 表 2 の 3 本(GELU-MLP [64]x4、B=2 seed 7、B=3 seed 13、B=4 seed 16)を統一プロトコルで走らせ直し、途中の場を書き出す。
最初の B=1(GELU-MLP [48]x4 seed 4)は、プロセスの 1 本目のずれを避けるためのウォームアップで、論文では使わない。
学習ループは code/src/train.py のまま（train_dump.py は、場を書き出す関数 1 つとその呼び出しを足したもの）。
出力: このフォルダの dump_B{1,2,3,4}/(run の記録と場の npz)。
"""
import os, sys, time

SP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SP, '..', '..', 'src')

sys.path.insert(0, SRC)
sys.path.insert(0, SP)
import torch
import train_dump as T

CASES = [(1, 48, 4, 1.231451556762),
         (2, 64, 7, 2.356966040244),
         (3, 64, 13, 3.437086811959),
         (4, 64, 16, 4.480050170227)]

frames = (list(range(0, 11)) + list(range(12, 101, 4)) + list(range(108, 301, 8))
          + list(range(320, 1201, 20)) + [1500, 2000, 3000, 5000, 10000, 20000])
frames = sorted(set(frames))

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device %s / 1 本あたり %d フレーム' % (dev, len(frames)), flush=True)

T.DUMP_EPOCHS = set(frames)
T.DUMP_L, T.DUMP_N = 4.0, 80        # 刻み 0.113。B=4 の立方も入る大きさ

for B, w, s, E_rec in CASES:
    out = os.path.join(SP, 'dump_B%d' % B)
    os.makedirs(out, exist_ok=True)
    T.DUMP_DIR = out
    t0 = time.time()
    print('=== B=%d  w%d  seed %d ===' % (B, w, s), flush=True)
    r = T.train_one('mlp', w, 4, seed=s, device=dev, outdir=out, verbose=False,
                    track='free', B=B, n_lumps=B,
                    w_topo=1000.0, w_origin=0.0, w_far=50.0,
                    epochs=20000, pretrain_epochs=8000, pretrain_lr=1e-3,
                    lump_spread=1.5, lump_max_r=2.5,
                    save_model=True, mlp_act='gelu', log_every=100000)
    print('   E %.12f   記録 %.12f   差 %+.3e' % (r['E'], E_rec, r['E'] - E_rec),
          flush=True)
    print('   B %.12f   E2/E4 %.9f   塊の間隔 %s'
          % (r['B'], r['virial'], ['%.2f' % x for x in r['lumps']['sep']]),
          flush=True)
    print('   %.0f 秒' % (time.time() - t0), flush=True)
