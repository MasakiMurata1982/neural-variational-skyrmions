#!/usr/bin/env python3
"""
Recompute the numbers of table 1,
directly from the csv of the runs.

usage:  python3 table1_recompute_v2.py [runs_reported.csv]
default: data/runs_reported.csv
"""
import os, csv, sys, collections, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__)); RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))   # release layout
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT

# run set used for figure 2 and for row 1 of table 1
SRC = {("2026-08-27_B2_arch_sweep",      "runs"),
       ("2026-09-11_refit_v2",           "runs_refit_v2"),
       ("2026-09-08_gelu_sweep",         "runs"),
       ("2026-08-29_weekend",            "runs_act")}

ARCHS = ("effkan", "mlp_tanh", "mlp_gelu")
NAME  = {"effkan": "KAN", "mlp_tanh": "tanh-MLP", "mlp_gelu": "GELU-MLP"}

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
# 同シードの検証再実行(runs_b1_regrid, table1_models, determinism)は統計から除く。
# 表 1 の 2〜4 行目は seed 0-4 の 5 シード平均(学習率掃引の seed 5-9 は 4 節の脚注の検定にだけ使う)。
rows = [x for x in rows if x.get("verification_rerun", "") != "True"]
SEEDS_TAB1 = {"0", "1", "2", "3", "4"}
print(f"{CSV}: {len(rows)} rows (verification re-runs excluded)\n")


def group(sel, key):
    d = collections.defaultdict(list)
    for x in sel:
        d[key(x)].append(float(x["E"]))
    return d


def hdr(t):
    print("=" * 68)
    print(t)
    print("=" * 68)


# ---------------------------------------------------------------- row 1
hdr("row 1  untuned defaults  (lr 2e-3, 1000 epochs, figure-2 run set)")
r1 = [x for x in rows
      if (x["experiment"], x["runs_dir"]) in SRC
      and x["track"] == "hard" and x["B_target"] == "2"
      and x["lr"] == "0.002" and not x["wallclock_budget_s"]
      and x["E"]]
d = group(r1, lambda x: (x["arch"], int(x["width"]), int(x["depth"])))
TAB1 = {"effkan": "2.357053", "mlp_tanh": "2.357939", "mlp_gelu": "2.357197"}   # 表 1 の 1 行目 = 5 シード平均
print(f"{'':<10}{'min run':>11}{'min seed-mean':>16}{'  at':>8}{'  n':>4}"
      f"{'   table 1':>11}")
for a in ARCHS:
    ks = [k for k in d if k[0] == a]
    mn = min(min(d[k]) for k in ks)
    kb = min(ks, key=lambda k: st.mean(d[k]))
    print(f"{NAME[a]:<10}{mn:>11.6f}{st.mean(d[kb]):>16.6f}"
          f"  (w{kb[1]},d{kb[2]}){len(d[kb]):>4}{TAB1[a]:>11}")

# ---------------------------------------------------------------- row 2
hdr("row 2  learning rate tuned per model  (all lr, 1000 epochs)")
r2 = [x for x in rows
      if x["track"] == "hard" and x["B_target"] == "2"
      and not x["wallclock_budget_s"]
      and x["E"] and x["epochs_config"] == "1000" and x["arch"] in ARCHS and x["seed"] in SEEDS_TAB1]
d2 = group(r2, lambda x: (x["arch"], int(x["width"]), int(x["depth"]), x["lr"]))
for a in ARCHS:
    ks = [k for k in d2 if k[0] == a and len(d2[k]) >= 3]
    ks.sort(key=lambda k: st.mean(d2[k]))
    print(f"\n{NAME[a]}  (three best by seed mean)")
    for k in ks[:3]:
        v = sorted(d2[k])
        print(f"   (w{k[1]},d{k[2]}) lr={k[3]:<7} n={len(v)}  "
              f"mean={st.mean(v):.6f}  min={v[0]:.6f}  max={v[-1]:.6f}")

# ------------------------------------------------------- rows 3 and 4
hdr("rows 3 and 4  wall-clock budgets")
TAB34 = {("30", "effkan"): "2.357219", ("30", "mlp_tanh"): "2.357811",
         ("30", "mlp_gelu"): "2.357120", ("120", "effkan"): "2.357028",
         ("120", "mlp_tanh"): "2.357113", ("120", "mlp_gelu"): "2.356989"}   # 5 シード平均
r34 = [x for x in rows if x["wallclock_budget_s"] in ("30", "120") and x["E"] and x["seed"] in SEEDS_TAB1]
d34 = group(r34, lambda x: (x["wallclock_budget_s"], x["arch"],
                            int(x["width"]), int(x["depth"])))
print(f"{'budget':<8}{'':<10}{'min run':>11}{'min seed-mean':>16}"
      f"{'  at':>8}{'  n':>4}{'   table 1':>11}")
for b in ("30", "120"):
    for a in ARCHS:
        ks = [k for k in d34 if k[0] == b and k[1] == a]
        if not ks:
            continue
        mn = min(min(d34[k]) for k in ks)
        kb = min(ks, key=lambda k: st.mean(d34[k]))
        print(f"{b + ' s':<8}{NAME[a]:<10}{mn:>11.6f}{st.mean(d34[kb]):>16.6f}"
              f"  (w{kb[2]},d{kb[3]}){len(d34[kb]):>4}{TAB34[(b, a)]:>11}")

# --------------------------------------------------- figure 2 caption
hdr("figure 2 caption: seed standard deviation, median over the 24 (width, depth) pairs")
d5 = group(r1, lambda x: (x["arch"], int(x["width"]), int(x["depth"])))
for a in ARCHS:
    sds = [st.stdev(d5[k]) for k in d5 if k[0] == a and len(d5[k]) >= 3]   # sample std (N-1), as in the paper
    print(f"{NAME[a]:<10} pairs={len(sds):<3} median={st.median(sds):.3e}")

# ------------------------------------------------------ pykan seed check
hdr("pykan: the three seeds of a configuration are one run (README, section 4)")
pk = [x for x in rows if x["arch"] == "pykan" and x["E"]]
dp = collections.defaultdict(list)
for x in pk:
    dp[(x["runs_dir"], x["B_target"], int(x["width"]), int(x["depth"]))].append(
        (int(x["seed"]), float(x["E"])))
for k in sorted(dp):
    v = sorted(dp[k])
    if len(v) < 2:
        continue
    spread = max(e for _, e in v) - min(e for _, e in v)
    print(f"{k[0]:<22} B={k[1]} (w{k[2]},d{k[3]})  spread={spread:.2e}  "
          + " ".join(f"s{s}:{e:.10f}" for s, e in v))
