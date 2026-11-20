# 3.2 節と付録 A.5 の値: 有理写像の 1 次元プロファイルのエネルギー。B=1 は 1.231446(動径積分を r = 1000 まで)と 1.231442(r ≤ 60)。
# B=2 では、r ≤ 60 の外に落ちる分が 4e-7 未満(3.2 節)。CPU で動き、ファイルは読まない。
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))  # release layout
from physics import train_profile_1d, energy_1d, angular_integrals
for B in (1, 2):
    torch.manual_seed(0)
    pm, E1d, I = train_profile_1d(B=B, epochs=1500, device="cpu", verbose=False)
    pmd = pm.double(); I_val = angular_integrals(B)[1]; E = {}
    for n_r, a, rmax in [(400, 2.0, 200.0), (1000, 2.0, 400.0), (4000, 2.0, 1000.0), (4000, 3.0, 60.0)]:
        E[rmax] = energy_1d(pmd, B=B, I_val=I_val, n_r=n_r, a_map=a, r_max=rmax, dev="cpu").item()
        print(f"B={B} 1D double: n_r={n_r} a={a} r_max={rmax}: E={E[rmax]:.8f}")
    print(f"B={B}: energy beyond r = 60 (r_max 1000 minus r_max 60): {E[1000.0] - E[60.0]:.1e}")
