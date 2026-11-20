# -*- coding: utf-8 -*-
"""図 3 の再描画用データを解ごとに 1 つの npz に保存する(torch が必要なのはここだけ)。
内容: vol(96^3 float32、バリオン密度、各軸 lin=np.linspace(-4,4,96)、indexing="ij")、lin、level(=0.5*vol.max())、
verts(N×3、原点中心の座標)、faces(M×3)、tag/runs_dir/B_target/n_params などのメタデータ。
描画は make_fig_isosurfaces_headline.py が npz から行う(numpy / scikit-image / matplotlib のみ)。"""
import os, sys, json
import numpy as np, torch
from skimage import measure
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
sys.path.insert(0, SRC)
from physics import build_q_free, densities_split
from models import build_net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNI = os.path.join(RECORDS, "2026-08-31_free_uniform", "runs_uniform")
TBD = os.path.join(RECORDS, "2026-08-31_free_uniform", "runs_uniform_mlp_tb")
CASES = [(1, UNI, "free_B1_mlp_w48_d4_ep20000_s4_gelu"), (2, UNI, "free_B2_mlp_w64_d4_ep20000_s7_gelu"),
         (3, TBD, "free_B3_mlp_w64_d4_ep100000_s13_tb660_gelu"), (4, TBD, "free_B4_mlp_w64_d4_ep100000_s16_tb660_gelu")]
L, n = 4.0, 96
lin = np.linspace(-L, L, n)
for B, rdir, tag in CASES:
    meta = json.load(open(os.path.join(rdir, tag + ".json"))); cfg = meta.get("config", {})
    net = build_net(meta["kind"], meta["width"], meta["depth"], device, meta.get("kan_grid", 5), zero_init=False, mlp_act=cfg.get("mlp_act", "tanh"))
    net.load_state_dict(torch.load(os.path.join(rdir, tag + "_model.pt"), map_location=device)); net.eval()
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1); out = np.zeros(pts.shape[0], dtype=np.float32)
    for i in range(0, pts.shape[0], 16384):
        xi = torch.tensor(pts[i:i + 16384], dtype=torch.float32, device=device).requires_grad_(True)
        b, e2, e4 = densities_split(xi, build_q_free(xi, net), create_graph=False); out[i:i + 16384] = b.detach().cpu().numpy()
    vol = out.reshape(n, n, n); level = 0.5 * float(vol.max())
    verts, faces, _, _ = measure.marching_cubes(vol, level, spacing=(lin[1] - lin[0],) * 3); verts = (verts - L).astype(np.float32)
    np.savez_compressed(os.path.join(DATA, f"iso_B{B}.npz"), vol=vol, lin=lin.astype(np.float32), level=np.float32(level), verts=verts, faces=faces.astype(np.int32),
                        tag=tag, runs_dir=os.path.relpath(rdir, RECORDS), B_target=B, n_params=meta["n_params"], E_record=meta["E"], B_record=meta["B"], virial_record=meta["virial"],
                        grid_note="vol[i,j,k] = baryon density at (lin[i], lin[j], lin[k]); verts are in the same coordinates (origin-centred); level = 0.5*vol.max()")
    print(f"B={B} {tag}: vol max {vol.max():.4f} level {level:.4f} verts {len(verts)} faces {len(faces)} -> iso_B{B}.npz", flush=True)
    del net; torch.cuda.empty_cache()
