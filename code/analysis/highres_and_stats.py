# -*- coding: utf-8 -*-
"""2026-09-11_kan21_b12 の KAN[21]x4、B=1,2 の 20 run について
 (1) 統一プロトコルの成功判定(表 2 のセル)、
 (2) 成功 run の保存モデルを 80×44×60(3 動径写像、各 1 回転、seed 1234)で再評価(表 10・付録 A.3 の条件、highres_all.py と同一)、
 (3) 表 10(付録 A.3)の形式のセル別統計(n, mean E80, std, min E80, median time)を出力する。
実行: python3 highres_and_stats.py"""
import os, sys, json, glob, statistics as st, torch
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
sys.path.insert(0, SRC)
from physics import build_q_free, eval_multi_grid
from models import build_net
dev = torch.device("cuda")
E_TABLE = {1: 1.2322, 2: 2.3582, 3: 3.4386, 4: 4.4804}


def ok(r):
    Bt = r["B_target"]
    return (abs(r["B"] - Bt) < 1e-3 and abs(r["E"] - E_TABLE[Bt]) / E_TABLE[Bt] < 0.01
            and abs(r["virial"] - 1) < 0.05 and r["dE_grids"] < 1e-2)


recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RECORDS, "2026-09-11_kan21_b12", "runs", "*.json")))]
OUTJ = os.path.join(DATA, "highres_kan21_b12.json")
hi = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
for r in recs:
    if not ok(r) or r["tag"] in hi:
        continue
    net = build_net(r["kind"], r["width"], r["depth"], dev, r.get("kan_grid", 5), zero_init=False)
    net.load_state_dict(torch.load(os.path.join(RECORDS, "2026-09-11_kan21_b12", "runs", r["tag"] + "_model.pt"), map_location=dev)); net.eval()
    ev = eval_multi_grid(lambda xi: build_q_free(xi, net), dev, quad=(80, 44, 60), r_max=60.0)
    hi[r["tag"]] = dict(B_target=r["B_target"], seed=r["seed"], E48=r["E"], E80=ev["E"], B80=ev["B"], virial80=ev["virial"], dE80=ev["dE_grids"],
                        time=r["time_train_sec"])
    print(f"{r['tag']:44s} E48={r['E']:.6f} E80={ev['E']:.6f} B80={ev['B']:.6f} dE80={ev['dE_grids']:.1e}", flush=True)
    json.dump(hi, open(OUTJ, "w"), indent=1)
    del net; torch.cuda.empty_cache()

print("\n== table 2 cells (uniform success) ==")
for Bt in (1, 2):
    cell = [r for r in recs if r["B_target"] == Bt]
    print(f"B={Bt} KAN[21]x4: {sum(ok(r) for r in cell)}/{len(cell)}   (fail tags: {[r['tag'] for r in cell if not ok(r)]})")
print("\n== table 10 rows (appendix A.3; successful runs, 80x44x60) ==")
print("B & arm & n & mean E & std & min E & time [s]")
for Bt in (1, 2):
    v = [hi[r["tag"]] for r in recs if r["B_target"] == Bt and ok(r)]
    E = [x["E80"] for x in v]; T = [r["time_train_sec"] for r in recs if r["B_target"] == Bt]
    print(f"{Bt} & KAN [21]x4 & {len(E)} & {st.mean(E):.5f} & {st.stdev(E):.1e} & {min(E):.5f} & {st.median(T):.0f}")
    print(f"   (E48 mean {st.mean(x['E48'] for x in v):.6f}; E80-E48 max |diff| {max(abs(x['E80']-x['E48']) for x in v):.1e}; B80 min/max {min(x['B80'] for x in v):.6f}/{max(x['B80'] for x in v):.6f})")
print("STATS_DONE")
