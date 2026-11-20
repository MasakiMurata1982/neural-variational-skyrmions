# -*- coding: utf-8 -*-
"""csv から論文の表を再生成する。対象: 表 2、表 4 の "This work" の行、表 6、表 10。
表 5・11 は classify_failures.py、表 12 は make_inventory_from_reported.py、表 3 は headline_symmetry.py、表 1 は table1_recompute_v2.py を参照。
選択条件(全表共通): uniform_protocol == True、success_uniform == True(成功)。
  表 2 / 表 6 の成功数: 各 (runs_dir, arch, width, B_target) の success_uniform の True の数 / 行数
  表 4: 成功 run の E_80x44x60 の最小値(B ごと、表 2 と表 6 の全 run)、その run の E/B、|B80-n|、E2/E4(48 格子記録値)
  表 10: 成功 run の E_80x44x60 の n / 平均 / 標本標準偏差 / 最小、wallclock_s(train_one の所要時間 = 刷り込み + 本学習)の中央値
  アーム: runs_uniform (KAN[16], GELU[64], GELU[48]) / runs_uniform_kan21 + 2026-09-11_kan21_b12/runs (KAN[21]) / runs_uniform_tb (KAN[16] 160 s) / runs_uniform_mlp_tb (GELU[64] 660 s)
表 10 は論文と同じ順で出す(4 ネットワークの 16 行のあとに GELU-MLP[64]x4 660 s の 2 行)。
実行: python3 make_tables_from_csv.py [runs_reported.csv]"""
import os, sys, csv, statistics as st, collections
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
rows = [r for r in csv.DictReader(open(CSV)) if r["uniform_protocol"] == "True" and r["runs_dir"] != "runs_uniform_repro"]; f = float
def arm(r):
    d = r["runs_dir"]; a = f"{'KAN' if r['arch']=='effkan' else 'GELU-MLP'}[{r['width']}]x{r['depth']}"
    if d == "runs_uniform_tb": return a + " 160 s"
    if d == "runs_uniform_mlp_tb": return a + " 660 s"
    return a
ARMS = ["KAN[16]x4", "KAN[21]x4", "GELU-MLP[64]x4", "GELU-MLP[48]x4", "KAN[16]x4 160 s", "GELU-MLP[64]x4 660 s"]
g = collections.defaultdict(list)
for r in rows: g[(arm(r), int(r["B_target"]))].append(r)
print(f"{CSV}: uniform-protocol rows (excluding repro) = {len(rows)}\n")
print("== table 2 / table 6: successes / runs ==")
print("| B | " + " | ".join(ARMS) + " |"); print("|---" * (len(ARMS) + 1) + "|")
for B in (1, 2, 3, 4):
    cells = []
    for a in ARMS:
        v = g.get((a, B), []); cells.append(f"{sum(r['success_uniform']=='True' for r in v)}/{len(v)}" if v else "--")
    print(f"| {B} | " + " | ".join(cells) + " |")
print("\n== table 4, rows \"This work\": lowest-energy successful run per B over all runs of tables 2 and 6, 80x44x60 grid ==")
print("| B | E80 min | E80/B | |B80 - n| | E2/E4 (record, 48 grid) | run |"); print("|---|---|---|---|---|---|")
for B in (1, 2, 3, 4):
    allsucc = [r for a in ARMS for r in g.get((a, B), []) if r["success_uniform"] == "True" and r["E_80x44x60"]]
    best = min(allsucc, key=lambda r: f(r["E_80x44x60"]))
    print(f"| {B} | {f(best['E_80x44x60']):.5f} | {f(best['E_80x44x60'])/B:.5f} | {abs(f(best['B_80x44x60'])-B):.1e} | {f(best['E2_over_E4']):.4f} | {best['run_id']} |")
STATS_HEAD = "| B | arm | successes / runs | mean E80 | std (sample, N-1) | min E80 | median wallclock_s (all runs of the cell) |"
def stats_rows(cells):
    print(STATS_HEAD); print("|---|---|---|---|---|---|---|")
    for B, a in cells:
        allv = g.get((a, B), []); v = [r for r in allv if r["success_uniform"] == "True" and r["E_80x44x60"]]
        if not v: continue
        E = [f(r["E_80x44x60"]) for r in v]; ta = [f(r["wallclock_s"]) for r in allv if r["wallclock_s"]]
        print(f"| {B} | {a} | {len(v)}/{len(allv)} | {st.mean(E):.5f} | {st.stdev(E) if len(E)>1 else float('nan'):.1e} | {min(E):.5f} | {st.median(ta):.0f} |")
print("\n== table 10: successful runs per cell, E on 80x44x60; the 18 rows in the order of the table ==")
print("(columns of table 10: successes / runs, mean E80, std (sample, N-1), min E80, median wallclock_s (all runs of the cell))")
stats_rows([(B, a) for B in (1, 2, 3, 4) for a in ARMS[:4]] + [(B, "GELU-MLP[64]x4 660 s") for B in (3, 4)])
print("\nnote: the std column is the sample standard deviation (statistics.stdev, N-1). wallclock_s = time_train_sec of the run record (train_one total: imprinting + energy minimisation).")
