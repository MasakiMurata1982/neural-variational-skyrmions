# -*- coding: utf-8 -*-
"""anchored 掃引(B=2、学習率 2e-3)の集計。csv から
 (1) 表 1 の 1 行目(shared lr 2e-3)・2 行目(lr tuned)の 5 シード平均・標本標準偏差・標準誤差、
 (2) 図 2 の front(シード平均・best、パラメータ数)、
 (3) 図 2 の説明文の中央値(24 組の標本標準偏差の中央値)、
を出力する。実行: python3 summarize_refit_v2.py [runs_reported.csv]"""
import os, sys, csv, json, math, statistics as st, collections
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
rows = list(csv.DictReader(open(CSV))); f = float
print(f"{CSV}: {len(rows)} rows; verification_rerun=True: {sum(r['verification_rerun']=='True' for r in rows)}")
NAME = {"effkan": "KAN", "mlp_tanh": "tanh-MLP", "mlp_gelu": "GELU-MLP", "pykan": "pykan"}

# 行の選択
def sweep_rows(refit_dir):
    return [r for r in rows if r["track"] == "hard" and r["B_target"] == "2" and r["epochs_config"] == "1000" and r["wallclock_budget_s"] == ""
            and r["arch"] in ("effkan", "mlp_tanh", "mlp_gelu")
            and ((r["experiment"] == "2026-08-27_B2_arch_sweep" and r["runs_dir"] == "runs") or r["runs_dir"] == refit_dir
                 or (r["experiment"] == "2026-09-08_gelu_sweep" and r["runs_dir"] == "runs") or (r["experiment"] == "2026-08-29_weekend" and r["runs_dir"] == "runs_act"))]

def stats(E):
    n = len(E); m = st.mean(E); sd = st.stdev(E) if n > 1 else float("nan"); psd = st.pstdev(E)
    return n, m, sd, psd, sd / math.sqrt(n)

print("\n== (1) table 1 rows 1-2: 5-seed statistics ==")
print("  row 1 shared (lr 2e-3, 1000 ep): KAN (64,4) / tanh (64,4) / GELU (64,4)")
for label, rd in (("sweep", "runs_refit_v2"),):
    sel = sweep_rows(rd)
    for arch, w, d in (("effkan", 64, 4), ("mlp_tanh", 64, 4), ("mlp_gelu", 64, 4)):
        E = sorted((int(r["seed"]), f(r["E"])) for r in sel if r["arch"] == arch and r["width"] == str(w) and r["depth"] == str(d) and f(r["lr"]) == 2e-3)
        n, m, sd, psd, se = stats([e for _, e in E])
        print(f"    {label:24s} {NAME[arch]:9s} ({w},{d}) n={n} seeds={[s for s,_ in E]} mean={m:.6f} ({m:.5f}) sd={sd:.1e} se={se:.1e} min={min(e for _,e in E):.6f} max={max(e for _,e in E):.6f}")
    # 行 1 の「最良構成」(シード平均最小)も表示
    g = collections.defaultdict(list)
    for r in sel:
        if f(r["lr"]) == 2e-3: g[(r["arch"], int(r["width"]), int(r["depth"]))].append(f(r["E"]))
    for arch in ("effkan", "mlp_tanh", "mlp_gelu"):
        ks = [k for k in g if k[0] == arch]; kb = min(ks, key=lambda k: st.mean(g[k]))
        print(f"    {label:24s} {NAME[arch]:9s} best seed-mean config {kb[1:]} n={len(g[kb])} mean={st.mean(g[kb]):.6f}; min single run={min(min(g[k]) for k in ks):.6f}")
print("  row 2 lr tuned (1000 ep, lr 1e-2): KAN (32,3) / tanh (64,4) / GELU (64,4)")
SRC2 = {"2026-08-29_weekend/runs_lr", "2026-08-29_weekend/runs_act", "2026-09-10_table1_seeds/runs_lr", "2026-09-10_gelu_lr/runs", "2026-09-11_lr_seeds/runs"}
for arch, w, d in (("effkan", 32, 3), ("mlp_tanh", 64, 4), ("mlp_gelu", 64, 4)):
    E = sorted((int(r["seed"]), f(r["E"])) for r in rows if r["track"] == "hard" and r["B_target"] == "2" and r["arch"] == arch and r["width"] == str(w) and r["depth"] == str(d)
               and r["epochs_config"] == "1000" and r["wallclock_budget_s"] == "" and f(r["lr"]) == 1e-2 and f"{r['experiment']}/{r['runs_dir']}" in SRC2 and int(r["seed"]) <= 4)
    n, m, sd, psd, se = stats([e for _, e in E])
    print(f"    {NAME[arch]:9s} ({w},{d}) lr=1e-2 n={n} seeds={[s for s,_ in E]} mean={m:.6f} ({m:.5f}) sd={sd:.1e} se={se:.1e}")

# (2) fronts
def fronts(sel, arch):
    g = collections.defaultdict(list)
    for r in sel:
        if r["arch"] == arch and f(r["lr"]) == 2e-3: g[int(r["n_params"])].append(f(r["E"]))
    pts_mean = sorted((p, st.mean(v), len(v)) for p, v in g.items()); pts_best = sorted((p, min(v), len(v)) for p, v in g.items())
    def front(pts):
        best = float("inf"); out = []
        for p, e, n in pts:
            if e < best - 1e-12: best = e; out.append((p, e, n))
        return out
    return front(pts_mean), front(pts_best)
print("\n== (2) figure 2 fronts (params: E; seed mean / best) ==")
for arch in ("effkan", "mlp_tanh", "mlp_gelu"):
    fn = fronts(sweep_rows("runs_refit_v2"), arch)
    for lab, i in (("seed-mean", 0), ("best", 1)):
        print(f"  {NAME[arch]:9s} {lab:9s} " + " ".join(f"{p}:{e:.6f}" for p, e, n in fn[i]))

# (3) seed std medians
print("\n== (3) figure 2 caption: median over 24 configs of the seed std (sample, N-1) ==")
for label, rd in (("sweep", "runs_refit_v2"),):
    sel = sweep_rows(rd)
    for arch in ("effkan", "mlp_tanh", "mlp_gelu"):
        g = collections.defaultdict(list)
        for r in sel:
            if r["arch"] == arch and f(r["lr"]) == 2e-3: g[(int(r["width"]), int(r["depth"]))].append(f(r["E"]))
        sds = [st.stdev(v) for v in g.values() if len(v) >= 3]
        print(f"  {label} {NAME[arch]:9s} configs={len(sds)} median stdev={st.median(sds):.2e}; n per config={sorted(set(len(v) for v in g.values()))}")
