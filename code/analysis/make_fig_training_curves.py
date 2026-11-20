# -*- coding: utf-8 -*-
"""Figure 4 (fig_training_curves.png): three training curves per charge, chosen by arrival epoch.

Runs: the successful 20,000-epoch uniform-protocol runs of the four networks of table 2 (KAN [16]x4, KAN [21]x4, GELU-MLP [64]x4,
GELU-MLP [48]x4; no wall-clock budget; the reproducibility re-runs `runs_uniform_repro` excluded): 65 runs for B=3, 39 for B=4.
Arrival epoch of a run = the first epoch after which the training-grid energy E stays within +-1 % of the reference value for the rest
of training (REF = {3: 3.4386, 4: 4.4804}; E is the raw per-epoch history, not smoothed). The runs with the smallest, median and largest
arrival epoch are drawn, irrespective of network; a dot marks each arrival epoch; the dotted line is the reference energy.

Input: data/runs_reported.csv (selection) and data/run_records/<experiment>/<runs_dir>/<run_id>_hist.npz (supplementary archive 1,
unpacked in place). Output: fig_training_curves.png next to this script. stdout: the three runs per charge with rank and arrival epoch."""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
OUT = os.path.join(HERE, 'fig_training_curves.png')

REF = {3: 3.4386, 4: 4.4804}
YTOP = {3: 10.0, 4: 8.0}
ARMS = {('effkan', 16), ('effkan', 21), ('mlp_gelu', 64), ('mlp_gelu', 48)}
LABS = ['earliest', 'median', 'latest']
COLS = ['#1b7837', '#2c4f9e', '#E69F00']   # Okabe-Ito orange for the latest run (colour-blind safe)

rows = list(csv.DictReader(open(CSV, encoding='utf-8')))

def hist_path(r):
    return os.path.join(RECORDS, r['experiment'], r['runs_dir'], r['run_id'] + '_hist.npz')

def arrival_epoch(E, ref):
    """1-based epoch after the last epoch with |E - ref|/ref >= 1 %; 1 if the run is inside the band from the start."""
    out = np.where(np.abs(E - ref) / ref >= 0.01)[0]
    return int(out[-1]) + 2 if len(out) else 1

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, B in zip(axes, (3, 4)):
    # successful runs only; four arms, 20,000 epochs, no wall-clock budget, the three reproducibility re-runs excluded
    sel = [r for r in rows if r['uniform_protocol'] == 'True' and r['epochs_config'] == '20000'
           and (r['wallclock_budget_s'] or '') == '' and r['runs_dir'] != 'runs_uniform_repro'
           and (r['arch'], int(r['width'])) in ARMS and r['B_target'] == str(B)
           and r['success_uniform'] == 'True']
    runs = []
    for r in sel:
        p = hist_path(r)
        if not os.path.exists(p):
            sys.exit(f"missing history {p}: unpack supplementary archive 1 into data/run_records/")
        E = np.asarray(np.load(p)['E'], float)
        runs.append((arrival_epoch(E, REF[B]), E, r['run_id']))
    runs.sort(key=lambda t: t[0])
    n = len(runs)
    idx = [0, (n - 1) // 2, n - 1]
    ep = np.arange(1, 20001)
    for lab, c, i in zip(LABS, COLS, idx):
        t, E, rid = runs[i]
        ax.plot(ep, E, color=c, lw=1.0, label='%s (%d)' % (lab, t))
        ax.plot([t], [E[t - 1]], 'o', ms=5, color=c, mec='k', mew=0.5, zorder=5)
        print('B=%d %-9s rank %2d/%d arrival %5d  %s' % (B, lab, i + 1, n, t, rid))
    ax.axhline(REF[B], color='k', ls=':', lw=1)
    ax.text(1.3, REF[B] - 0.012 * (YTOP[B] - REF[B]), 'reference energy', fontsize=8, va='top')
    ax.set_xscale('log')
    ax.set_xlim(1, 20000)
    ax.set_ylim(REF[B] - 0.09 * (YTOP[B] - REF[B]), YTOP[B])
    ax.set_title('$B=%d$' % B)
    ax.set_xlabel('Epoch (energy minimisation)')
    ax.set_ylabel('$E$ on training grid')
    ax.grid(alpha=0.3, which='major')
    ax.legend(fontsize=8, loc='upper right', title='arrival epoch', title_fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(OUT)
