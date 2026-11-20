# Figure 6 (appendix D): baryon-density isosurfaces during training.
# Data: field dumps written by re-running the three GELU-MLP[64]x4 runs of table 2
# (B=2 seed 7, B=3 seed 13, B=4 seed 16) on an RTX 6000 Ada with PyTorch 2.10.0+cu128;
# the re-runs reproduce the recorded E and B histories bit for bit.
# The dumps that the figure uses are in data/derived/appD_frames (dump_B2, dump_B3, dump_B4).
# usage: python make_fig_relaxation.py DATA_DIR fig_relaxation.png
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure

HERE = sys.argv[1]
SEED = {2: 7, 3: 13, 4: 16}
VIEWS = {2: (32.0, -145.0), 3: (44.2, -131.2), 4: (40.0, 112.0)}


def load(B, ep):
    d = np.load(os.path.join(HERE, 'dump_B%d' % B, 'dump_ep%05d.npz' % ep))
    return d['vol'], d['lin']


out = sys.argv[2]
rows = []
for a in ['2:0,52,68,84,100,164,20000', '3:0,100,340,400,460,580,20000',
          '4:0,100,600,1000,1200,1500,20000']:
    B, eps = a.split(':')
    rows.append((int(B), [int(e) for e in eps.split(',')]))
ncol = max(len(e) for _, e in rows)
fig = plt.figure(figsize=(1.75 * ncol, 1.9 * len(rows)), dpi=200)
for i, (B, eps) in enumerate(rows):
    lev = 0.5 * load(B, 20000)[0].max()
    rec = json.load(open(os.path.join(HERE, 'dump_B%d' % B,
                                      'free_B%d_mlp_w64_d4_ep20000_s%d_gelu.json' % (B, SEED[B]))))
    cen = np.array(rec['lumps']['centers'])
    for j, ep in enumerate(eps):
        vol, lin = load(B, ep)
        h = float(lin[1] - lin[0])
        ax = fig.add_subplot(len(rows), ncol, i * ncol + j + 1, projection='3d')
        v, f, _, _ = measure.marching_cubes(vol, lev, spacing=(h, h, h))
        v = v + lin[0]
        ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], cmap='viridis', lw=0,
                        edgecolor='none', antialiased=True, alpha=0.55 if ep == 0 else 1.0)
        if ep == 0:
            ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], s=22, c='red', edgecolors='white', linewidths=0.6, depthshade=False, zorder=10)
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_zlim(-2.6, 2.6)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(*VIEWS[B]); ax.set_axis_off()
        ax.set_title('epoch %d' % ep, fontsize=8, pad=-4)
        if j == 0:
            ax.text2D(-0.12, 0.5, '$B=%d$' % B, transform=ax.transAxes, rotation=90,
                      va='center', fontsize=10)
fig.subplots_adjust(left=0.03, right=1, bottom=0, top=0.96, wspace=-0.15, hspace=0.0)
fig.savefig(out)
print('saved', out)
