# -*- coding: utf-8 -*-
"""最細格子(80×44×60)で最良となった代表解の対称性指標(慣性主値の縮退・多重極 S_l)。
表 3 と表 9(付録 A.1)の対称性の値を出す。B=1: MLP48 s4, B=2: MLP64 s7, B=3: MLP64 s13(660 s), B=4: MLP64 s16(660 s)。"""
import os, sys, json
import numpy as np, torch
from multipole import multipole_powers   # S_l via scipy.special.sph_harm_y (scipy >= 1.15); self-test: python3 multipole.py
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
sys.path.insert(0, SRC)
from physics import build_q_free, densities_split, make_quadrature
from models import build_net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNI = os.path.join(RECORDS, "2026-08-31_free_uniform", "runs_uniform")
TBD = os.path.join(RECORDS, "2026-08-31_free_uniform", "runs_uniform_mlp_tb")
TAGS = [(UNI, "free_B1_mlp_w48_d4_ep20000_s4_gelu"), (UNI, "free_B2_mlp_w64_d4_ep20000_s7_gelu"),
        (TBD, "free_B3_mlp_w64_d4_ep100000_s13_tb660_gelu"), (TBD, "free_B4_mlp_w64_d4_ep100000_s16_tb660_gelu")]
out = {}
for d, tag in TAGS:
    r = json.load(open(os.path.join(d, tag + ".json"))); cfg = r.get("config", {})
    net = build_net(r["kind"], r["width"], r["depth"], device, r.get("kan_grid", 5), zero_init=False, mlp_act=cfg.get("mlp_act", "tanh"))
    net.load_state_dict(torch.load(os.path.join(d, tag + "_model.pt"), map_location=device)); net.eval()
    x, w = make_quadrature(48, 28, 36, a_map=3.0, r_max=30.0, device=device)
    bs, es = [], []
    for i in range(0, x.shape[0], 8192):
        xi = x[i:i+8192].detach().clone().requires_grad_(True)
        b, e2, e4 = densities_split(xi, build_q_free(xi, net), create_graph=False)
        bs.append(b.detach().double().cpu()); es.append((e2+e4).detach().double().cpu())
    b = torch.cat(bs).numpy(); e = torch.cat(es).numpy(); xd = x.detach().double().cpu().numpy(); wd = w.double().cpu().numpy()
    M = np.sum(e*wd); com = np.sum(xd*(e*wd)[:, None], axis=0)/M; xc = xd-com; r2 = np.sum(xc**2, axis=1)
    T = np.einsum("n,nij->ij", e*wd, r2[:, None, None]*np.eye(3)[None]-xc[:, :, None]*xc[:, None, :]); ev = np.sort(np.linalg.eigvalsh(T))
    B = r["B_target"]
    degs = dict(pair01=abs(ev[1]-ev[0])/ev[0], all=(ev[2]-ev[0])/ev[0])   # pair01 = (I2-I1)/I1 (torus), all = (I3-I1)/I1
    S = multipole_powers(b, wd, xd, ls=(2, 3, 4))   # about the baryon centroid; S_l = sqrt(sum_m |Q_lm|^2)
    out[tag] = dict(B=B, inertia=ev.tolist(), degeneracy=degs, S=S)
    print(f"{tag:44s} I={ev[0]:.4f},{ev[1]:.4f},{ev[2]:.4f}  deg01={degs['pair01']:.1e} degAll={degs['all']:.1e}  S2={S[2]:.2e} S3={S[3]:.2e} S4={S[4]:.2e}", flush=True)
    del net
    if device.type == "cuda": torch.cuda.empty_cache()
json.dump(out, open(os.path.join(DATA, "headline_symmetry.json"), "w"), indent=1)
