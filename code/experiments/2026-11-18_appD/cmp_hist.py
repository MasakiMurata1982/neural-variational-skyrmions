# Compare the E and B histories of the three re-runs of appendix D with the recorded histories of the
# same runs of table 2 (2026-08-31_free_uniform/runs_uniform/ in archive 1).
#
# Usage: python cmp_hist.py <zip of archive 1> <folder of the re-run records>
#   <zip of archive 1>             the archive of the histories (skyrmion_histories.zip)
#   <folder of the re-run records>  the folder that holds free_B{2,3,4}_mlp_w64_d4_ep20000_s{7,13,16}_gelu_hist.npz
#                                   of the re-runs (data/run_records/2026-11-18_appD/runs/ after archive 1 is unpacked)
import os, zipfile, io, sys
import numpy as np

ZIP = sys.argv[1]
RUNS = sys.argv[2]
z = zipfile.ZipFile(ZIP)
for B, s in ((2, 7), (3, 13), (4, 16)):
    n = "free_B%d_mlp_w64_d4_ep20000_s%d_gelu_hist.npz" % (B, s)
    local = os.path.join(RUNS, n)
    a = np.load(local)
    r = np.load(io.BytesIO(z.read("2026-08-31_free_uniform/runs_uniform/" + n)))
    Ea, Er = a["E"].astype(float), r["E"].astype(float)
    Ba, Br = a["B"].astype(float), r["B"].astype(float)
    m = min(len(Ea), len(Er))
    print("B=%d len %d/%d  max|dE| %.3e  final dE %.3e  max|dB| %.3e  E bit-identical: %s  B bit-identical: %s"
          % (B, len(Ea), len(Er), np.abs(Ea[:m] - Er[:m]).max(), Ea[-1] - Er[-1],
             np.abs(Ba[:m] - Br[:m]).max(), np.array_equal(a["E"], r["E"]), np.array_equal(a["B"], r["B"])))
