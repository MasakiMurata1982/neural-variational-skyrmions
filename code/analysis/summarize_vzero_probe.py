# -*- coding: utf-8 -*-
"""Appendix A.4, mode (a): the minimum of |v| at the charge changes of the two re-runs.

Input: the histories of the two re-runs, data/run_records/2026-10-01_vzero/runs/<tag>_hist.npz (archive 1, unpacked in place).
  B      the charge on the training grid at every epoch
  probe  one row per recorded epoch: epoch, minimum of |v| on a grid of spacing 0.1, median of |v| on that grid, position of the minimum.
         The probe was recorded at every epoch for the first 2,000 epochs and at every 20th epoch after that.

Rule: the charge of an epoch is the median of B over 21 epochs, rounded to an integer. A change is an epoch after which this integer
differs. For every change in the first 2,000 epochs the script takes the smallest recorded minimum of |v| within 8 epochs of the change
and compares it with the median of the minimum of |v| over the first 2,000 epochs. A change counts as a dip when the smallest value is
below 0.3 times that median.

The paper states the range of the dips rounded outwards: the minimum of |v| fell to between 0.01 and 0.09, 4 to 17 times below its
median over the first 2,000 epochs.

Run: python3 summarize_vzero_probe.py"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))   # release layout
RUNS = os.path.join(RELEASE, "data", "run_records", "2026-10-01_vzero", "runs")
TAGS = ["free_B3_mlp_w64_d4_ep20000_s3_gelu", "free_B4_mlp_w64_d4_ep20000_s9_gelu"]
FIRST, WIN, HALF, FRAC = 2000, 21, 8, 0.3


def rounded_charge(B):
    """Median of B over WIN epochs (the ends repeat the first and the last value), rounded to an integer."""
    h = WIN // 2
    p = np.concatenate([np.full(h, B[0]), B, np.full(h, B[-1])])
    return np.rint(np.median(np.lib.stride_tricks.sliding_window_view(p, WIN), axis=1)).astype(int)


dips = []; n_changes = 0
for tag in TAGS:
    f = os.path.join(RUNS, tag + "_hist.npz")
    if not os.path.exists(f):
        sys.exit("missing %s (unpack archive 1 into data/run_records/)" % f)
    d = np.load(f)
    B = d["B"].astype(float); P = d["probe"]; ep = P[:, 0].astype(int); vmin = P[:, 1]
    n = rounded_charge(B)
    changes = [int(i) + 1 for i in np.where(np.diff(n) != 0)[0] if i + 1 <= FIRST]      # last epoch (1-based) with the old integer
    med = float(np.median(vmin[ep <= FIRST]))
    print("%s: %d changes of the charge in the first %d epochs; median of min|v| over these epochs %.4f"
          % (tag, len(changes), FIRST, med))
    k = 0
    for e in changes:
        w = (ep >= e - HALF) & (ep <= e + HALF)
        m = float(vmin[w].min()); dip = m < FRAC * med; k += dip
        print("   epoch %5d  %d -> %d   smallest min|v| %.5f = %.3f x median (median / value = %6.3f)  %s"
              % (e, n[e - 1], n[e], m, m / med, med / m, "dip" if dip else "no dip"))
        if dip:
            dips.append((m, med / m))
    print("   dips: %d of %d" % (k, len(changes)))
    n_changes += len(changes)
a = np.array(dips)
print("both re-runs: %d dips in %d changes; smallest min|v| %.5f to %.5f; median / value %.3f to %.3f"
      % (len(a), n_changes, a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max()))
print("paper (appendix A.4): between 0.01 and 0.09, 4 to 17 times below the median over the first 2,000 epochs")
