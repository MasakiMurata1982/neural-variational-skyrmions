# -*- coding: utf-8 -*-
"""Rotation-invariant multipole powers S_l of a density given on a quadrature grid.

    S_l = sqrt( sum_m |Q_lm|^2 ),   Q_lm = int b(x) r^l conj(Y_lm(theta, phi)) d^3x / int b(x) r^l d^3x,

about the centroid of b (default).  Selection rules used in the paper (section 5.2, table 3):
spherical S_l ~ 0 for all l >= 1; torus D_inf-h: even l only (S_2 != 0); tetrahedral T_d: S_3 != 0, S_2 ~ 0;
cubic O_h: S_4 != 0, S_2 ~ S_3 ~ 0.

Spherical harmonics: scipy.special.sph_harm_y(l, m, theta, phi) with theta = POLAR angle, phi = AZIMUTH (scipy >= 1.15).
The older sph_harm(m, l, phi, theta) (argument order (m, l, azimuth, polar)!) was deprecated in scipy 1.15 and removed
afterwards; on scipy < 1.15 it is used as a fallback with the arguments re-ordered.  Both return the same Y_l^m
(Condon-Shortley phase, orthonormal on the sphere).

Run this file directly for a self-test (explicit Y_l^m formula, a test density with known S_3 = |a|/(4 pi),
rotation invariance, and the theta/phi swap that would break S_3 while leaving S_2 untouched)."""
import numpy as np
try:
    from scipy.special import sph_harm_y
except ImportError:  # scipy < 1.15
    from scipy.special import sph_harm as _sph_harm_old

    def sph_harm_y(l, m, theta, phi):
        return _sph_harm_old(m, l, phi, theta)


def multipole_powers(b, w, x, ls=(2, 3, 4), center=True):
    """Return {l: S_l} for the density values b (N,), quadrature weights w (N,), points x (N, 3).

    center=True: measure angles about the centroid of b (as in the paper); center=False: about the origin."""
    b = np.asarray(b, dtype=np.float64); w = np.asarray(w, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
    bw = b * w
    if center:
        x = x - np.sum(x * bw[:, None], axis=0) / np.sum(bw)
    r = np.linalg.norm(x, axis=1)
    theta = np.arccos(np.clip(x[:, 2] / (r + 1e-300), -1.0, 1.0))   # polar angle
    phi = np.arctan2(x[:, 1], x[:, 0])                                # azimuth
    S = {}
    for l in ls:
        norm = np.sum(bw * r ** l)
        s = 0.0
        for m in range(-l, l + 1):
            Q = np.sum(bw * r ** l * np.conj(sph_harm_y(l, m, theta, phi))) / norm
            s += abs(Q) ** 2
        S[int(l)] = float(np.sqrt(s))
    return S


# ----------------------------------------------------------------------------------------------------------- self-test
def _Y_explicit(l, m, theta, phi):
    """Y_l^m from the associated Legendre function (Condon-Shortley phase), independent of sph_harm/sph_harm_y."""
    from scipy.special import lpmv, gammaln
    am = abs(m)
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * np.exp(gammaln(l - am + 1) - gammaln(l + am + 1)))
    Y = norm * lpmv(am, l, np.cos(theta)) * np.exp(1j * am * phi)      # lpmv includes (-1)^m
    return Y if m >= 0 else (-1) ** am * np.conj(Y)


def _cartesian_grid(L=7.0, n=141):
    g = np.linspace(-L, L, n); h = g[1] - g[0]
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return x, np.full(x.shape[0], h ** 3)


def _test_density(x, a=1.0):
    """b = exp(-r^2) (1 + a * (1/4) sqrt(105/pi) (x^2 - y^2) z / r^3): S_3 = |a|/(4 pi), S_2 = S_4 = 0 (Cartesian form only)."""
    r = np.linalg.norm(x, axis=1)
    Y32 = 0.25 * np.sqrt(105 / np.pi) * (x[:, 0] ** 2 - x[:, 1] ** 2) * x[:, 2] / (r ** 3 + 1e-300)
    return np.exp(-r ** 2) * (1 + a * Y32)


if __name__ == "__main__":
    import scipy
    print(f"numpy {np.__version__}, scipy {scipy.__version__}")
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi, 2000); ph = rng.uniform(-np.pi, np.pi, 2000)
    # (a) sph_harm_y(l, m, theta, phi) equals the explicit Y_l^m
    worst = 0.0
    for l in range(0, 7):
        for m in range(-l, l + 1):
            d = np.max(np.abs(sph_harm_y(l, m, th, ph) - _Y_explicit(l, m, th, ph)))
            worst = max(worst, d)
    print(f"(a) max |sph_harm_y - explicit Y_l^m| over l<=6, all m: {worst:.2e}")
    assert worst < 1e-11
    # (b) test density with a known S_3
    x, w = _cartesian_grid()
    for a in (1.0, 0.3):
        S = multipole_powers(_test_density(x, a), w, x, ls=(1, 2, 3, 4), center=False)
        print(f"(b) a={a}: S_3={S[3]:.9f} (expected {abs(a)/(4*np.pi):.9f}), S_1={S[1]:.1e}, S_2={S[2]:.1e}, S_4={S[4]:.1e}")
        assert abs(S[3] - abs(a) / (4 * np.pi)) < 1e-6 and S[2] < 1e-6 and S[4] < 1e-6
    # (c) rotation invariance: rotate the points, keep the density values
    q = rng.normal(size=4); q /= np.linalg.norm(q); a0, b0, c0, d0 = q
    R = np.array([[a0*a0+b0*b0-c0*c0-d0*d0, 2*(b0*c0-a0*d0), 2*(b0*d0+a0*c0)],
                  [2*(b0*c0+a0*d0), a0*a0-b0*b0+c0*c0-d0*d0, 2*(c0*d0-a0*b0)],
                  [2*(b0*d0-a0*c0), 2*(c0*d0+a0*b0), a0*a0-b0*b0-c0*c0+d0*d0]])
    bt = _test_density(x, 1.0)
    S0 = multipole_powers(bt, w, x, center=False); S1 = multipole_powers(bt, w, x @ R.T, center=False)
    print(f"(c) rotation: S_3 {S0[3]:.9f} -> {S1[3]:.9f}, S_2 {S0[2]:.1e} -> {S1[2]:.1e}, S_4 {S0[4]:.1e} -> {S1[4]:.1e}")
    assert abs(S0[3] - S1[3]) < 1e-9
    # (d) the trap: swapping theta and phi
    r = np.linalg.norm(x, axis=1); theta = np.arccos(np.clip(x[:, 2] / (r + 1e-300), -1, 1)); phi = np.arctan2(x[:, 1], x[:, 0])
    bw = bt * w
    def S_swapped(l):
        norm = np.sum(bw * r ** l)
        return float(np.sqrt(sum(abs(np.sum(bw * r ** l * np.conj(sph_harm_y(l, m, phi, theta))) / norm) ** 2 for m in range(-l, l + 1))))
    print(f"(d) theta/phi swapped: S_2={S_swapped(2):.3e}, S_3={S_swapped(3):.3e} (correct: 0 and {1/(4*np.pi):.6f})")
    # (e) the older formula sph_harm(m, l, phi, theta), if still available
    try:
        import warnings
        from scipy.special import sph_harm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dd = max(np.max(np.abs(sph_harm(m, l, ph, th) - sph_harm_y(l, m, th, ph))) for l in range(7) for m in range(-l, l + 1))
        print(f"(e) max |sph_harm(m,l,phi,theta) - sph_harm_y(l,m,theta,phi)| = {dd:.2e}")
    except ImportError:
        print("(e) sph_harm not available in this scipy (removed); skipped")
    print("SELFTEST_OK")
