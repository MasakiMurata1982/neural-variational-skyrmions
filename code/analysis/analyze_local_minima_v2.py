# -*- coding: utf-8 -*-
"""付録 A.4 のモード (c): 統一プロトコルの run(runs_uniform、runs_uniform_kan21、2026-09-11_kan21_b12/runs)のうち、mode_c() の条件を満たす 21 本の形の解析。
各 run の保存モデルはその run の JSON と同じフォルダから読む。出力: local_minima_v2.json(慣性主値の比、エネルギーの超過、arch, runs_dir)。"""
import os, sys, json, glob
import numpy as np, torch
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
DIRS = [os.path.join(RECORDS, "2026-08-31_free_uniform", "runs_uniform"),
        os.path.join(RECORDS, "2026-08-31_free_uniform", "runs_uniform_kan21"),
        os.path.join(RECORDS, "2026-09-11_kan21_b12", "runs")]
E_TABLE = {1: 1.2322, 2: 2.3582, 3: 3.4386, 4: 4.4804}
def mode_c(r):
    Bt = r["B_target"]
    return (abs(r["B"] - Bt) < 1e-3 and abs(r["virial"] - 1) < 0.05 and r["dE_grids"] < 1e-2 and abs(r["E"] - E_TABLE[Bt]) / E_TABLE[Bt] >= 0.01)
def analyse(net, R=30.0):
    x, w = make_quadrature(48, 28, 36, a_map=3.0, r_max=R, device=device)
    bs, es = [], []
    for i in range(0, x.shape[0], 8192):
        xi = x[i:i + 8192].detach().clone().requires_grad_(True)
        b, e2, e4 = densities_split(xi, build_q_free(xi, net), create_graph=False)
        bs.append(b.detach().double().cpu()); es.append((e2 + e4).detach().double().cpu())
    b = torch.cat(bs).numpy(); e = torch.cat(es).numpy(); xd = x.detach().double().cpu().numpy(); wd = w.double().cpu().numpy()
    M = np.sum(e * wd); com = np.sum(xd * (e * wd)[:, None], axis=0) / M; xc = xd - com; r2 = np.sum(xc ** 2, axis=1)
    T = np.einsum("n,nij->ij", e * wd, r2[:, None, None] * np.eye(3)[None] - xc[:, :, None] * xc[:, None, :]); ev = np.sort(np.linalg.eigvalsh(T)) / M
    return ev
rows = []
for d in DIRS:
    for f in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(f))
        if isinstance(r, dict) and r.get("track") == "free" and "epochs" in r and r["epochs"] == 20000: r["_dir"] = d; rows.append(r)
sel = sorted([r for r in rows if mode_c(r)], key=lambda r: (r["B_target"], r["E"]))
out = []
print(f"{'tag':48s} {'arch':9s} {'E':>8s} {'+%':>5s} {'vir':>6s}  inertia ratios (I1:I2:I3)")
for r in sel:
    tag = r["tag"]; cfg = r.get("config", {}); arch = f"{r['kind']}{r['width']}" if r["kind"] != "mlp" else f"{cfg.get('mlp_act','tanh')}{r['width']}"
    net = build_net(r["kind"], r["width"], r["depth"], device, r.get("kan_grid", 5), zero_init=False, mlp_act=cfg.get("mlp_act", "tanh"))
    net.load_state_dict(torch.load(os.path.join(r["_dir"], tag + "_model.pt"), map_location=device)); net.eval()
    ev = analyse(net); ratios = ev / ev[-1]; exc = (r["E"] - E_TABLE[r["B_target"]]) / E_TABLE[r["B_target"]] * 100
    print(f"{tag:48s} {arch:9s} {r['E']:8.4f} {exc:5.1f} {r['virial']:6.3f}  {ratios[0]:.3f}:{ratios[1]:.3f}:1", flush=True)
    out.append(dict(tag=tag, B=r["B_target"], E=r["E"], excess_pct=exc, virial=r["virial"], inertia_ratios=[float(v) for v in ratios],
                    arch=arch, runs_dir=os.path.relpath(r["_dir"], SKY)))
    del net
    if device.type == "cuda": torch.cuda.empty_cache()
json.dump(out, open(os.path.join(DATA, "local_minima_v2.json"), "w"), indent=1)
print(f"\n{len(out)} mode-(c) runs analysed -> local_minima_v2.json")
