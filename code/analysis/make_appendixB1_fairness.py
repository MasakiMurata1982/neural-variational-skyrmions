# -*- coding: utf-8 -*-
"""Appendix B.1 (fairness checklist, three tests) from data/runs_reported.csv.
Appendix B.1 has three tests: (i) learning rate, 36 runs; (ii) MLP activation
(SiLU), 9 runs; (iii) wall-clock budget, 12 runs. They are the rows b1_lr, b1_silu and b1_time of table 12 (column
cohort_inventory). For each test the script prints the quantity that
the appendix states, including the comparison of KAN[16]x4 with the MLPs[64]x4 of table 1 and the parameter counts of
(i) and (iii).
The ten runs at seeds 5-9 of KAN[32]x3 (row t1_seeds59 of table 12) are not part of appendix B.1; they
belong to the footnote of section 4 on the best learning rate, whose numbers are printed separately at the end.
Run: python3 make_appendixB1_fairness.py [runs_reported.csv]"""
import os, sys, csv, statistics as st, collections, math
HERE = os.path.dirname(os.path.abspath(__file__))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = next((a for a in sys.argv[1:] if a.endswith(".csv")), CSV_DEFAULT)
rows = list(csv.DictReader(open(CSV))); f = float
def fold(r): return f"{r['experiment']}/{r['runs_dir']}"
def cfg(r): return (r["arch"], int(r["width"]), int(r["depth"]))
def lr(r): return f(r["lr"])
def seed(r): return int(r["seed"])
def tb(r): return r["wallclock_budget_s"]
def coh(k): return [r for r in rows if r["cohort_inventory"] == k]
def mean_se(E):
    m = st.mean(E); sd = st.stdev(E) if len(E) > 1 else float("nan"); return m, sd, sd / math.sqrt(len(E))
def pm(E):   # mean(standard error in units of the last digit), as in table 1
    m, _, se = mean_se(E); return f"{m:.6f}({round(se * 1e6):d})"
def nparams(arch, w, d):
    s = {r["n_params"] for r in rows if r["track"] == "hard" and cfg(r) == (arch, w, d) and (arch != "effkan" or r["kan_grid"] in ("5", "5.0"))}
    assert len(s) == 1, (arch, w, d, s); return int(s.pop())
NAME = {"effkan": "KAN", "mlp_tanh": "tanh-MLP", "mlp_gelu": "GELU-MLP"}
B1 = [("b1_lr", "B.1 (i) learning rate: KAN[16]x4, tanh-MLP[32]x4, GELU-MLP[32]x4 at four rates, three seeds", 36),
      ("b1_silu", "B.1 (ii) MLP activation: SiLU at depth 4 and widths 16, 32, 64, three seeds", 9),
      ("b1_time", "B.1 (iii) wall-clock budget: KAN[16]x2 and tanh-MLP[32]x4 after 30 s and 120 s, three seeds", 12)]
print(f"{os.path.relpath(CSV, RELEASE)}: {len(rows)} rows\n| table12_id | test of appendix B.1 | runs | folders (n) |\n|---|---|---|---|")
tot = 0
for k, lab, n_paper in B1:
    v = coh(k); fo = collections.Counter(fold(r) for r in v); tot += len(v)
    print(f"| {k} | {lab} | {len(v)} | {'; '.join(f'{a} ({n})' for a, n in sorted(fo.items()))} |")
    assert len(v) == n_paper, (k, len(v), n_paper)
print(f"| | **Total of appendix B.1** | **{tot}** | |")
# ---- (i) learning rate ----
print("\n== (i) Learning rate: the 36 added runs, KAN[16]x4 and both MLPs at [32]x4, four rates, three seeds (seed-mean E) ==")
g = collections.defaultdict(list)
for r in coh("b1_lr"):
    assert cfg(r) in {("effkan", 16, 4), ("mlp_tanh", 32, 4), ("mlp_gelu", 32, 4)}; g[(lr(r), r["arch"])].append(f(r["E"]))
for l in sorted(set(k[0] for k in g)):
    m = {a: st.mean(g[(l, a)]) for a in ("effkan", "mlp_tanh", "mlp_gelu")}
    order = sorted(m, key=m.get)
    print(f"  lr={l:<7g} KAN {m['effkan']:.6f} (n={len(g[(l,'effkan')])})  tanh-MLP {m['mlp_tanh']:.6f} (n={len(g[(l,'mlp_tanh')])})  GELU-MLP {m['mlp_gelu']:.6f} (n={len(g[(l,'mlp_gelu')])})"
          f"  -> lowest {NAME[order[0]]}, highest {NAME[order[-1]]}")
print("  paper: at each of these four rates the KAN has the lowest mean energy, and the tanh-MLP has the highest")
print("\n== (i) KAN[16]x4 against the MLPs[64]x4 of table 1, at each of the five rates (mean E with its standard error in units of the last digit) ==")
pk, pt, pg = nparams("effkan", 16, 4), nparams("mlp_tanh", 64, 4), nparams("mlp_gelu", 64, 4)
print(f"  parameters: KAN[16]x4 {pk}; tanh-MLP[64]x4 {pt}; GELU-MLP[64]x4 {pg}   (paper: 7 920 against 12 996)")
def anch(arch, w, d, l, cohorts):   # anchored track, B=2, 1000 epochs, no time limit
    return [f(r["E"]) for r in rows if r["cohort_inventory"] in cohorts and r["track"] == "hard" and r["B_target"] == "2" and cfg(r) == (arch, w, d)
            and lr(r) == l and tb(r) == "" and r["epochs_config"] == "1000"]
all_lower = True; all_within = True
for l in (5e-4, 1e-3, 2e-3, 5e-3, 1e-2):
    K = anch("effkan", 16, 4, l, {"sweep", "b1_lr"}); T = anch("mlp_tanh", 64, 4, l, {"sweep", "t1_lr"}); G = anch("mlp_gelu", 64, 4, l, {"sweep", "t1_lr"})
    (mk, _, sk), (mt, _, s_t), (mg, _, sg) = mean_se(K), mean_se(T), mean_se(G)
    dt = (mk - mt) / math.sqrt(sk ** 2 + s_t ** 2); dg = (mk - mg) / math.sqrt(sk ** 2 + sg ** 2)
    all_lower &= mk < mt; all_within &= abs(dg) < 3
    print(f"  lr={l:<7g} KAN[16]x4 {pm(K)} (n={len(K)})  tanh-MLP[64]x4 {pm(T)} (n={len(T)})  GELU-MLP[64]x4 {pm(G)} (n={len(G)})"
          f"  KAN - tanh = {mk-mt:+.6f} ({dt:+.1f} s.e. of the difference)  KAN - GELU = {mk-mg:+.6f} ({dg:+.2f} s.e. of the difference)")
print(f"  KAN[16]x4 lower than tanh-MLP[64]x4 at each of the five rates: {all_lower}")
print(f"  KAN[16]x4 and GELU-MLP[64]x4 differ by less than three standard errors of the difference at each of the five rates: {all_within}")
print("  paper: at each of the five rates KAN[16]x4 is lower than the tanh-MLP[64]x4; at each of these rates the mean energies of KAN[16]x4 and the GELU-MLP[64]x4 differ by less than three standard errors of the difference")
print("  runs: KAN[16]x4 at 2e-3 and the MLPs[64]x4 at 2e-3 are sweep runs (five seeds); the MLPs[64]x4 at the other four rates are the runs of table 1, row 2 (t1_lr, five seeds); KAN[16]x4 at the other four rates are the added runs (b1_lr, three seeds)")
# ---- (ii) activation ----
print("\n== (ii) MLP activation: SiLU at depth 4, widths 16, 32, 64, three seeds, against tanh and GELU at the same width (B=2, anchored, lr 2e-3, 1000 epochs) ==")
def act_runs(arch, w, seeds):
    return [f(r["E"]) for r in rows if r["track"] == "hard" and r["B_target"] == "2" and cfg(r) == (arch, w, 4) and lr(r) == 2e-3 and tb(r) == "" and r["epochs_config"] == "1000"
            and r["verification_rerun"] != "True" and r["kan_grid"] == "" and seed(r) in seeds]
for w in (16, 32, 64):
    silu = [f(r["E"]) for r in coh("b1_silu") if int(r["width"]) == w]
    for label, seeds in (("seeds 0-2", {0, 1, 2}), ("all seeds", set(range(100)))):
        t = act_runs("mlp_tanh", w, seeds); ge = act_runs("mlp_gelu", w, seeds)
        between = min(st.mean(t), st.mean(ge)) < st.mean(silu) < max(st.mean(t), st.mean(ge))
        print(f"  width {w:2d} ({label}): tanh {st.mean(t):.6f} (n={len(t)})  SiLU {st.mean(silu):.6f} (n={len(silu)})  GELU {st.mean(ge):.6f} (n={len(ge)})  -> SiLU between: {between}")
print("  paper: at each of the three widths the mean energy of SiLU lies between those of tanh and GELU")
# ---- (iii) wall-clock budget ----
print("\n== (iii) Wall-clock budget, anchored track: KAN[16]x2 vs tanh-MLP[32]x4 at 30 s and 120 s, three seeds (seed-mean E) ==")
g = collections.defaultdict(list)
for r in coh("b1_time"):
    assert cfg(r) in {("effkan", 16, 2), ("mlp_tanh", 32, 4)}; g[(int(tb(r)), r["arch"])].append(f(r["E"]))
for t in sorted(set(k[0] for k in g)):
    print(f"  {t:3d} s: KAN[16]x2 {st.mean(g[(t,'effkan')]):.6f} (n={len(g[(t,'effkan')])})  tanh-MLP[32]x4 {st.mean(g[(t,'mlp_tanh')]):.6f} (n={len(g[(t,'mlp_tanh')])})  -> KAN lower: {st.mean(g[(t,'effkan')]) < st.mean(g[(t,'mlp_tanh')])}")
print("  paper: the KAN is again lower than the tanh-MLP after both times")
print(f"  parameters: KAN[16]x2 {nparams('effkan', 16, 2)}; tanh-MLP[32]x4 {nparams('mlp_tanh', 32, 4)}   (paper: 3 312 and 3 428)")
# ---- section 4, footnote on the best learning rate (not part of appendix B.1) ----
print("\n== Not appendix B.1 -- section 4, footnote on the best learning rate of table 1, row 2: 1e-2 against 5e-3 ==")
def at(arch, w, d, l, smax):
    return [f(r["E"]) for r in rows if r["cohort_inventory"] in {"t1_lr", "t1_seeds59"} and cfg(r) == (arch, w, d) and lr(r) == l and seed(r) <= smax]
def cmp_rates(label, arch, w, d, smax):
    A, C = at(arch, w, d, 1e-2, smax), at(arch, w, d, 5e-3, smax); (m1, _, se1), (m2, _, se2) = mean_se(A), mean_se(C)
    print(f"  {label}: lr 1e-2 mean={m1:.6f} se={se1:.1e} (n={len(A)}); lr 5e-3 mean={m2:.6f} se={se2:.1e} (n={len(C)});"
          f" 5e-3 higher by {m2-m1:.1e}, {(m2-m1)/math.sqrt(se1**2+se2**2):.1f} standard errors of the difference")
cmp_rates("GELU-MLP[64]x4, seeds 0-4 (t1_lr)", "mlp_gelu", 64, 4, 4)
cmp_rates("KAN[32]x3, seeds 0-4 (t1_lr)", "effkan", 32, 3, 4)
cmp_rates("KAN[32]x3, seeds 0-9 (t1_lr and the ten runs of t1_seeds59)", "effkan", 32, 3, 9)
print(f"  runs of t1_seeds59: {len(coh('t1_seeds59'))} (seeds {sorted({seed(r) for r in coh('t1_seeds59')})}, rates {sorted({lr(r) for r in coh('t1_seeds59')})})")
print("  paper: GELU-MLP higher by 4e-5, 1.9 standard errors; KAN higher by 3e-5, 3.0 standard errors with five seeds and 4.8 with ten")
