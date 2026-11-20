# -*- coding: utf-8 -*-
"""表 5 と表 11 の失敗モードの分類を runs_reported.csv から再現する。
モード (c) の抽出条件は analyze_local_minima_v2.py の mode_c() と同じ:
  (c) |B-Bt| < 1e-3 and |E2/E4 - 1| < 0.05 and dE_grids < 1e-2 and (E - E_ref)/E_ref >= 0.01   (E_ref = 成功判定と同じ格子参照値 E_TABLE)
ここではそれを固定し、残りを (a) |B-Bt| >= 1e-3、(b) B は正確だが (c) でない(未緩和)とする。
表 5 は 4 アーム(KAN[16], KAN[21], GELU[64], GELU[48])240 run 全体の分類(a 0/6/21、b 0/2/7、c 1/7/13)。
B=1 の列、表 5 の successful の行、表 11 の配置(ネットワークごとに (a)/(b)/(c))も出す。
実行: python3 classify_failures.py [runs_reported.csv]"""
import os, sys, csv, collections
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows(cp932)でも ₂ ₄ Δ を出力できるように
HERE = os.path.dirname(os.path.abspath(__file__)); SKY = os.path.abspath(os.path.join(HERE, "..", ".."))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
E_TABLE = {1: 1.2322, 2: 2.3582, 3: 3.4386, 4: 4.4804}   # 成功判定・mode_c() と同じ格子参照値(付録の E_ref)
rows = list(csv.DictReader(open(CSV))); f = float
ARMS = {("effkan", "16"): "KAN[16]", ("mlp_gelu", "64"): "GELU[64]", ("mlp_gelu", "48"): "GELU[48]", ("effkan", "21"): "KAN[21]"}
uni_all = [r for r in rows if r["uniform_protocol"] == "True" and r["runs_dir"] in ("runs_uniform", "runs_uniform_kan21", "runs") and r["epochs_config"] == "20000"
           and (r["arch"], r["width"]) in ARMS]                                  # 表 2 の 4 ネットワークの 240 本
uni = [r for r in uni_all if r["success_uniform"] == "False"]

def classify(r):
    Bt = int(r["B_target"]); B = f(r["B"]); E = f(r["E"]); vir = f(r["E2_over_E4"]); dE = f(r["delta_E"]); ex = (E - E_TABLE[Bt]) / E_TABLE[Bt]
    dB = abs(B - Bt); dv = abs(vir - 1)
    if dB >= 1e-3: mode = "a"
    elif dv < 0.05 and dE < 1e-2 and ex >= 0.01: mode = "c"
    else: mode = "b"
    return mode

res = collections.defaultdict(lambda: collections.Counter())
for r in uni:
    arm = ARMS[(r["arch"], r["width"])]; mode = classify(r); Bt = int(r["B_target"])
    res[(arm, Bt)][mode] += 1
print(f"{CSV}: unsuccessful uniform-protocol runs = {len(uni)}\n")
print("== Table 5 reproduction: all four arms (KAN[16], KAN[21], GELU[64], GELU[48]), B = 1 to 4 ==")
print("| mode | B=1 | B=2 | B=3 | B=4 | Table 5 |\n|---|---|---|---|---|---|")
T5 = {"a": (0, 0, 6, 21), "b": (0, 0, 2, 7), "c": (0, 1, 7, 13), "successful": (40, 39, 65, 39)}
T11 = ("KAN[16]", "KAN[21]", "GELU[64]", "GELU[48]")   # 表 11 と表 2 のネットワークの順
ok = True
for m in "abc":
    c = [sum(res[(a, B)][m] for a in T11) for B in (1, 2, 3, 4)]
    ok &= tuple(c) == T5[m]; print(f"| ({m}) | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {' / '.join(map(str, T5[m]))} {'OK' if tuple(c) == T5[m] else 'MISMATCH'} |")
n_all = [sum(int(r["B_target"]) == B for r in uni_all) for B in (1, 2, 3, 4)]
c = [sum(int(r["B_target"]) == B and r["success_uniform"] == "True" for r in uni_all) for B in (1, 2, 3, 4)]
ok &= tuple(c) == T5["successful"]; print(f"| successful | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {' / '.join(map(str, T5['successful']))} {'OK' if tuple(c) == T5['successful'] else 'MISMATCH'} |")
print(f"| runs | {n_all[0]} | {n_all[1]} | {n_all[2]} | {n_all[3]} | 40 / 40 / 80 / 80 {'OK' if n_all == [40, 40, 80, 80] else 'MISMATCH'} |"); ok &= n_all == [40, 40, 80, 80]
print("reproduced (4 arms):", ok)
print("\n== Table 11 layout: unsuccessful runs per network and charge, as (a) / (b) / (c) ==")
print("| network | B=1 | B=2 | B=3 | B=4 |\n|---|---|---|---|---|")
for arm in T11:
    print(f"| {arm} | " + " | ".join(f"{res[(arm, B)]['a']} / {res[(arm, B)]['b']} / {res[(arm, B)]['c']}" for B in (1, 2, 3, 4)) + " |")
print("| all four networks | " + " | ".join(" / ".join(str(sum(res[(a, B)][m] for a in T11)) for m in "abc") for B in (1, 2, 3, 4)) + " |")
