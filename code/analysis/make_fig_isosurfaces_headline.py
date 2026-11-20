# -*- coding: utf-8 -*-
"""図 3(fig_isosurfaces.png)を headline 解(表 3・表 4・付録 A.1、headline_symmetry.py の TAGS と同じ 4 本)から描く。
データは compute_iso_data.py が保存した iso_B{1..4}.npz(96^3 一様格子のバリオン密度 vol、格子 lin、等値面 level=0.5*vol.max()、
marching cubes の verts/faces)。このスクリプトは torch を使わない(numpy / matplotlib のみ。npz が無ければ compute_iso_data.py を先に実行)。
描画条件: 2×2、各軸 -3..3、box aspect 1:1:1、viridis(z 座標)、パネルタイトルは形状名のみ、suptitle なし。
視点(B=1 は matplotlib 既定):
  B=1: elev 30, azim -60(既定)
  B=2: 慣性テンソル最大固有値の固有ベクトル(トーラス対称軸、elev 7 / azim -145)から 25 度傾けた方向 → elev 32, azim -145
  B=3: 穴の 1 つを正面に見る方向(C3 軸)から 20 度傾けた向き → elev 44.2, azim -131.2
  B=4: 立方体の体対角線方向(3 つの面の穴が見える)→ elev 28, azim -125"""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5})
VIEWS = {1: (30.0, -60.0), 2: (32.0, -145.0), 3: (44.2, -131.2), 4: (28.0, -125.0)}   # (elev, azim)
TITLES = {1: r"$B = 1$ (spherical)", 2: r"$B = 2$ (toroidal)", 3: r"$B = 3$ (tetrahedral)", 4: r"$B = 4$ (cubic)"}
fig = plt.figure(figsize=(10.5, 10.8)); log = []
for k, B in enumerate((1, 2, 3, 4)):
    d = np.load(os.path.join(DATA, f"iso_B{B}.npz")); v, f = d["verts"], d["faces"]; elev, azim = VIEWS[B]
    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], cmap="viridis", lw=0.05, edgecolor="none", alpha=0.95)
    ax.set_title(TITLES[B], fontsize=12)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(-3, 3); ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    line = f"B={B} {str(d['tag']):46s} view_init(elev={elev:.0f}, azim={azim:.0f})  level={float(d['level']):.4f} verts={len(v)} faces={len(f)}  record E={float(d['E_record']):.6f} B={float(d['B_record']):.6f} E2/E4={float(d['virial_record']):.5f}"
    print(line); log.append(line)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.95, hspace=0.26, wspace=0.05)
fig.savefig(os.path.join(HERE, "fig_isosurfaces.png"), dpi=300)
print("saved fig_isosurfaces.png")
