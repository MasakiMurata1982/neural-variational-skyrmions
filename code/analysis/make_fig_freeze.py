# -*- coding: utf-8 -*-
"""図 5(fig_freezing.png、凍結/注入/刷り込みの B 軌跡)を描く。
入力: 3 本の run の *_hist.npz(毎エポックの E, B。B は訓練格子上の値)。
出力: fig_freezing.png、stdout に群 4(B を r≤12 で測る run)の B の逸脱統計。"""
import os, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
B3D = os.path.join(RECORDS, "2026-08-28_B3_ansatz_free")
CASES = [("frozen", os.path.join(B3D, "runs_v2", "free_B2_effkan_w16_d4_ep5000_s0_g5_hist.npz"), r"random initial data, $B$ on $r\leq60$", "#0072B2", "--", 2.4),
         ("truncated", os.path.join(B3D, "runs_free2", "free_B2_effkan_w16_d4_ep5000_s0_g5_hist.npz"), r"random initial data, $B$ on $r\leq12$", "#E69F00", ":", 1.6),
         ("imprinted", os.path.join(B3D, "runs_seeded2", "free_B2_effkan_w16_d4_ep10000_s1_g5_hist.npz"), r"imprinted initial data, $B$ on $r\leq60$", "#009E73", "-", 1.6)]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5})
fig, ax = plt.subplots(figsize=(7.6, 4.1)); data = {}
for key, path, label, c, ls, lw in CASES:
    d = np.load(path); Bh, Eh = d["B"], d["E"]; data[key] = (Bh, Eh, os.path.relpath(path, RECORDS))
    ax.plot(np.arange(1, len(Bh) + 1), Bh, ls, color=c, lw=lw, label=label, zorder=3 if ls == "--" else 2)
ax.axhline(2.0, color="k", lw=0.8, ls=":"); ax.text(150, 2.04, "target $B=2$", fontsize=8)
ax.set_xlabel("Epoch (energy minimisation)"); ax.set_ylabel("Topological charge $B$ (training grid)"); ax.set_ylim(-0.15, 2.25); ax.set_xlim(-100, 5100); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_freezing.png"), dpi=300); print("fig_freezing.png")
print("\n== excursions of B in the frozen and the truncated-ball run ==")
for key in ("frozen", "truncated"):
    B, E, src = data[key]; ep = np.arange(1, len(B) + 1)
    for thr in (0.01, 0.1, 1.0):
        m = np.abs(B) > thr; idx = ep[m]
        print(f"  {key:9s} |B|>{thr:<4}: {m.sum():4d} of {len(B)} epochs" + (f"; first at epoch {idx[0]}, last at {idx[-1]}; epochs (first 40): {idx[:40].tolist()}" if m.sum() else ""))
    i = int(np.argmax(np.abs(B))); print(f"  {key:9s} max|B|={abs(B[i]):.4f} at epoch {ep[i]} (E there = {E[i]:.3f}); B_final={B[-1]:+.5f}, E_final={E[-1]:.4f}; max B over last 1000 epochs = {B[-1000:].max():.4f}, mean = {B[-1000:].mean():+.5f}")
    if key == "truncated":
        # 逸脱区間(|B|>0.05 の連続区間)を列挙
        m = np.abs(B) > 0.05; segs = []; start = None
        for j, flag in enumerate(m):
            if flag and start is None: start = j
            if (not flag or j == len(m) - 1) and start is not None: end = j if not flag else j + 1; segs.append((ep[start], ep[end - 1], float(np.abs(B[start:end]).max()), float(E[start:end].max()))); start = None
        print(f"  {key:9s} excursion segments with |B|>0.05 (start, end, max|B|, max E): {segs}")
        late = ep >= 3000; print(f"  {key:9s} epochs >= 3000: mean B = {B[late].mean():+.4f}, median B = {np.median(B[late]):+.5f}, fraction |B|>0.01 = {np.mean(np.abs(B[late])>0.01):.3f}")
print("\n== all random-initialisation runs (runs_v2/, runs_free2/): |B| after epoch 20 ==")
import glob
for f in sorted(glob.glob(os.path.join(B3D, "runs_v2", "*_hist.npz")) + glob.glob(os.path.join(B3D, "runs_free2", "*_hist.npz"))):
    B = np.load(f)["B"]; print(f"  {os.path.relpath(f, B3D):58s} n={len(B)} max|B| epochs 1-20: {np.abs(B[:20]).max():.4f}; max|B| epochs >20: {np.abs(B[20:]).max():.4f}; B_final={B[-1]:+.5f}")
