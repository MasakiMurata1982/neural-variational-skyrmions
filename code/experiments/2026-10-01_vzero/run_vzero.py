# Re-run selected mode-(a) runs of the uniform protocol with a probe of |v|.
# Settings are those of code/experiments/2026-08-31_free_uniform (COMMON), unchanged.
# Usage: python3 run_vzero.py g64_B3_s3,g64_B4_s9   (the two re-runs of appendix A.4; output in runs_g64_B3_s3,g64_B4_s9/ of this folder)
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'src')
sys.path.insert(0, HERE)          # train_vzero first
sys.path.insert(1, os.path.abspath(SRC))
import torch
from train_vzero import train_one

device = torch.device('cuda')
COMMON = dict(track="free", w_topo=1000.0, w_origin=0.0, w_far=50.0,
              epochs=20000, pretrain_epochs=8000, pretrain_lr=1e-3,
              lump_spread=1.5, lump_max_r=2.5,
              save_model=False, log_every=1000)
PROBE = dict(probe_L=6.0, probe_n=121, probe_every=(2000, 20))
RUNS = {
    'g64_B3_s3': ('mlp', 64, 4, 3, 3, dict(mlp_act='gelu')),
    'g64_B4_s9': ('mlp', 64, 4, 4, 9, dict(mlp_act='gelu')),
}
mode = sys.argv[1]
OUT = os.path.join(HERE, 'runs_' + mode)
# warm-up: the first run of a process can differ from later runs (section 7.4 of the paper)
train_one('mlp', 8, 1, seed=0, device=device, outdir=None, verbose=False,
          **dict(COMMON, epochs=20, pretrain_epochs=20), B=1, n_lumps=1, mlp_act='gelu')
for name in mode.split(','):
    kind, w, d, Bt, s, extra = RUNS[name]
    t0 = time.time()
    r = train_one(kind, w, d, seed=s, device=device, outdir=OUT, force=True,
                  verbose=True, B=Bt, n_lumps=Bt, **COMMON, **PROBE, **extra)
    print('RUN_DONE', name, r['tag'], 'B=%.5f E=%.5f %.0fs' % (r['B'], r['E'], time.time() - t0), flush=True)
