# -*- coding: utf-8 -*-
"""表 12(run inventory)を data/runs_reported.csv の cohort_inventory 列から数える。
cohort_inventory の値は表 12 の 20 行を表す。
表 12 の行の順に、行ごとの本数と合計を出す。run が 1 本も無い行は出さない。
実行: python3 make_inventory_from_reported.py [runs_reported.csv]"""
import os, sys, csv, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = next((a for a in sys.argv[1:] if a.endswith(".csv")), os.path.join(RELEASE, "data", "runs_reported.csv"))
rows = list(csv.DictReader(open(CSV)))
# (table12_id, 表 12 の見出し, 表 12 の行)。見出しと行の順は論文の表 12 と同じ。
LABEL = [
    ("sweep", "Section 4", "Anchored sweep (section 4, figure 2, table 1, appendices A.2 and A.5)"),
    ("t1_lr", "Section 4", "Table 1, row 2: learning rates 5e-4, 1e-3, 5e-3 and 1e-2, three networks, five seeds (section 4)"),
    ("t1_time", "Section 4", "Table 1, rows 3 and 4: stopped after 30 s and after 120 s, three networks, five seeds (section 4)"),
    ("t1_seeds59", "Section 4", "Seeds 5 to 9 of KAN[32]x3 at learning rates 5e-3 and 1e-2 (section 4; appendix C (iv))"),
    ("t1_regrid", "Section 4", "Re-runs of the 60 runs behind table 1, evaluated also on the 80x44x60 grid (section 4)"),
    ("uniform", "Section 5", "Uniform protocol: KAN[16]x4, KAN[21]x4, GELU-MLP[64]x4 and GELU-MLP[48]x4 (table 2)"),
    ("wallclock", "Section 5", "Equal wall-clock budget: KAN[16]x4 for 160 s and GELU-MLP[64]x4 for 660 s (section 5.5, table 6)"),
    ("t7", "Section 6", "The sixteen runs of table 7 (subsection 6.3)"),
    ("fig5_imprinted", "Section 6", "The imprinted run of figure 5 (subsection 6.3)"),
    ("design", "Section 6", "Runs that vary how far the imprinting regression is taken, at B_t = 2 (subsection 6.4, table 8)"),
    ("kan_grid", "Section 7", "KAN[16]x4 and KAN[8]x2 at G = 3 and G = 10, three seeds (section 7.3)"),
    ("repro13", "Section 7", "Thirteen re-runs to measure GPU nondeterminism (section 7.4)"),
    ("fastkan", "Appendix A", "Anchored-track FastKAN runs (appendices A.2 and A.5)"),
    ("a4_vzero", "Appendix A", "Re-runs of the two runs that unwind to the vacuum, recording the minimum of the norm of v (appendix A.4)"),
    ("a5_regrid", "Appendix A", "Re-runs of five anchored runs at B = 1, evaluated also on the 80x44x60 grid (appendix A.5)"),
    ("b1_lr", "Appendix B", "KAN[16]x4, tanh-MLP[32]x4, GELU-MLP[32]x4: four learning rates, three seeds (appendix B.1 (i))"),
    ("b1_silu", "Appendix B", "MLP with SiLU activation at depth 4 and widths 16, 32 and 64, three seeds (appendix B.1 (ii))"),
    ("b1_time", "Appendix B", "KAN[16]x2 and tanh-MLP[32]x4 stopped after 30 s and after 120 s, three seeds (appendix B.1 (iii))"),
    ("pykan", "Appendix B", "Anchored-track pykan runs (appendix B.2)"),
    ("appD", "Appendix D", "Re-runs of three runs of table 2, saving the baryon density during training (appendix D)"),
]
c = collections.Counter(r["cohort_inventory"] for r in rows)
shown = [(k, h, lab) for k, h, lab in LABEL if c[k]]
print(f"{CSV}: {len(rows)} rows, {len(shown)} rows of table 12\n| row | table12_id | heading | experiment | runs |\n|---|---|---|---|---|")
for i, (k, h, lab) in enumerate(LABEL, 1):
    if c[k]: print(f"| {i} | {k} | {h} | {lab} | {c[k]} |")
print(f"| | | | **Total** | **{sum(c.values())}** |")
unknown = set(c) - {k for k, _, _ in LABEL}
if unknown: print("WARNING unknown table12_id:", unknown)
