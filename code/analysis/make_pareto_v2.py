"""図 2(fig_pareto.png): anchored B=2 の Pareto 前線を描く。
系列は tanh-MLP、GELU-MLP、KAN (our impl.) の 3 つ。"""
import csv, collections, statistics, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
# --- release layout (skyrmion_release/): this file in code/analysis; code/src; data/runs_reported.csv; data/derived (JSON/npz/csv);
#     data/run_records/<experiment>/<runs_dir>/<tag>.json (+ _hist.npz from supplementary archive 1, _model.pt from archive 2, unpacked in place) ---
RELEASE = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(RELEASE, "code", "src"); DATA = os.path.join(RELEASE, "data", "derived"); RECORDS = os.path.join(RELEASE, "data", "run_records")
CSV_DEFAULT = os.path.join(RELEASE, "data", "runs_reported.csv")
CSV = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
SRC = {('2026-08-27_B2_arch_sweep','runs'),
       ('2026-09-11_refit_v2','runs_refit_v2'),
       ('2026-09-08_gelu_sweep','runs'),
       ('2026-08-29_weekend','runs_act')}
E0 = 2.41627   # rational-map ansatz energy, where every run starts

rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
def F(x):
    try: return float(x)
    except: return None

def agg(arch):
    g = collections.defaultdict(list)
    for r in rows:
        if (r['track']=='hard' and r['B_target']=='2' and r['lr']=='0.002'
            and (r.get('wallclock_budget_s') or '')=='' 
            and r['arch']==arch and (r['experiment'],r['runs_dir']) in SRC):
            g[int(r['n_params'])].append((F(r['E']), F(r['wallclock_s'])))
    out=[]
    for p,v in sorted(g.items()):
        Es=[e for e,_ in v]; Ws=[t for _,t in v]
        out.append(dict(params=p, n=len(v), E=statistics.mean(Es),
                        sd=(statistics.stdev(Es) if len(Es) > 1 else 0.0), wall=statistics.mean(Ws)))   # sample std (N-1), as in the paper
    return out

def front(pts, xk):
    best=float('inf'); f=[]
    for r in sorted(pts, key=lambda r:r[xk]):
        if r['E'] < best - 1e-12:
            best=r['E']; f.append(r)
    return f

SERIES = [("mlp_tanh","MLP (tanh)","tab:blue","-","o"),
          ("mlp_gelu","MLP (GELU)","tab:green","--","s"),
          ("effkan","KAN (our impl.)","tab:orange","-","^")]
D = {a: agg(a) for a,_,_,_,_ in SERIES}

def draw(path):
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6))
    for i,(xk,xlab,title) in enumerate([("params","Trainable parameters","(a) Accuracy vs. parameters"),
                     ("wall","Training wall-clock time [s] (1000 epochs)","(b) Accuracy vs. wall-clock time")]):
        a = ax[i]
        a.axhline(E0, color="0.45", ls="-.", lw=1.2)
        for arch,lab,col,ls,mk in SERIES:
            pts=D[arch]
            a.errorbar([r[xk] for r in pts],[r['E'] for r in pts],
                       yerr=[r['sd'] for r in pts], fmt=mk, ms=5, color=col,
                       ecolor=col, elinewidth=0.9, alpha=0.75, ls="none", zorder=2)
            f=front(pts,xk)
            a.step([r[xk] for r in f],[r['E'] for r in f], where="post",
                   color=col, ls=ls, lw=2.0, label=lab, zorder=3)
        a.set_xscale("log"); a.set_xlabel(xlab)
        a.set_ylim(2.3545, 2.4205); a.set_title(title)
        a.grid(alpha=0.3, lw=0.6)
        a.set_ylabel("$E$ (multi-grid evaluation, seed mean)")
    ax[0].text(0.02,0.975,"rational-map ansatz $E_0$ (init.)",transform=ax[0].transAxes,
               va="top",color="0.35",fontsize=9)
    ax[0].legend(loc="upper right", fontsize=9)
    ax[1].legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    print("wrote", path)

draw(os.path.join(HERE, "fig_pareto.png"))
for a,_,_,_,_ in SERIES:
    print(a, "configs", len(D[a]), "runs", sum(r['n'] for r in D[a]))
