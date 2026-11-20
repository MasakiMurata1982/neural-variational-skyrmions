# -*- coding: utf-8 -*-
"""
Skyrme 模型の物理コア。
内容:
  - エネルギー密度を E2(2次項)と E4(Skyrme項)に分離して返す(Derrick ビリアル検証用)
  - 任意の B(有理写像 R=z^B)の基底場・1次元プロファイル
  - soft トラック(アンザッツ基底なし・トポロジー自由)の場の構成
単位系: エネルギーは 12π² 単位(Faddeev-Bogomolny 下限が E >= |B|)。
"""
import math
import numpy as np
import torch
import torch.nn as nn

PI = math.pi

# 参照値(単位 12π²)。
# E_TABLE: Manton & Sutcliffe, Table 9.1(3次元格子計算、離散化で ~0.06% 高い)
# E_STAR : 推定真値。B=2 は 1次元厳密変分値 × (B=1 の table/1D 比 1.000612) による補正。
#          B=1 は hedgehog が厳密なので 1D 変分値そのもの。
#          B=3,4 は table 値をそのまま置く。
E_TABLE = {1: 1.2322, 2: 2.3582, 3: 3.4386, 4: 4.4804}
E_STAR = {1: 1.23145, 2: 2.35676, 3: 3.4386, 4: 4.4804}


# ----------------------------------------------------------------------
# 求積 (Gauss-Legendre + 半無限区間写像 r = a s/(1-s))
# ----------------------------------------------------------------------
def make_quadrature(n_r=40, n_theta=24, n_phi=32, a_map=3.0, r_max=60.0,
                    device="cpu", dtype=torch.float32):
    """求積点 x (N,3) と重み w (N,) を返す。∫F d³x ≈ Σ w F"""
    s_max = r_max / (a_map + r_max)
    t_s, w_s = np.polynomial.legendre.leggauss(n_r)
    s = 0.5 * s_max * (t_s + 1.0)
    w_s = 0.5 * s_max * w_s
    r = a_map * s / (1.0 - s)
    dr_ds = a_map / (1.0 - s) ** 2
    w_r = w_s * dr_ds * r ** 2
    u, w_u = np.polynomial.legendre.leggauss(n_theta)
    sin_th = np.sqrt(np.clip(1.0 - u ** 2, 0.0, None))
    phi = 2.0 * PI * (np.arange(n_phi) + 0.5) / n_phi
    w_phi = np.full(n_phi, 2.0 * PI / n_phi)
    R, ST = r[:, None, None], sin_th[None, :, None]
    U, PH = u[None, :, None], phi[None, None, :]
    X = R * ST * np.cos(PH)
    Y = R * ST * np.sin(PH)
    Z = R * U + 0.0 * PH
    W = w_r[:, None, None] * w_u[None, :, None] * w_phi[None, None, :]
    x = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
    x = torch.tensor(x, dtype=dtype, device=device).requires_grad_(True)
    w = torch.tensor(W.reshape(-1), dtype=dtype, device=device)
    return x, w


def random_rotation(device, dtype=torch.float32):
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q = Q.clone(); Q[:, 0] = -Q[:, 0]
    return Q


class RandomizedQuadrature:
    """毎ステップ格子をランダム回転+動径写像 a を無作為選択(求積の悪用対策)。"""
    def __init__(self, n_r=32, n_theta=20, n_phi=24,
                 a_list=(2.4, 2.7, 3.0, 3.3, 3.6), r_max=60.0,
                 device="cpu", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.grids = []
        for a in a_list:
            x, w = make_quadrature(n_r, n_theta, n_phi, a_map=a, r_max=r_max,
                                   device=device, dtype=dtype)
            self.grids.append((x.detach(), w))
        self.n = len(self.grids)

    def sample(self, rotate=True):
        k = int(torch.randint(self.n, (1,)).item())
        x0, w = self.grids[k]
        if rotate:
            Q = random_rotation(self.device, self.dtype)
            x = (x0 @ Q.T).detach().requires_grad_(True)
        else:
            x = x0.clone().detach().requires_grad_(True)
        return x, w


# ----------------------------------------------------------------------
# 有理写像 R(z)=z^B と 1 次元プロファイル
# ----------------------------------------------------------------------
def features(x, eps=1e-12):
    """u = x/(1+r): 単位球内に有界化(KAN のスプライン定義域 [-1,1] と整合)"""
    r = x.norm(dim=1, keepdim=True)
    return x / (1.0 + r + eps)


def rational_map_direction(x, B=2, eps=1e-12, log_clamp=60.0):
    """R(z)=z^B に対する単位ベクトル n_R(tan(θ/2) を対数で扱い安定化)"""
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    r = x.norm(dim=1)
    rho = torch.sqrt(X * X + Y * Y + eps)
    log_t = torch.log(rho + eps) - torch.log(r + Z + eps)
    psi = 2.0 * torch.atan(torch.exp(torch.clamp(B * log_t, -log_clamp, log_clamp)))
    ang = B * torch.atan2(Y, X)
    sp = torch.sin(psi)
    return torch.stack([sp * torch.cos(ang), sp * torch.sin(ang), torch.cos(psi)], dim=1)


def angular_integrals(B=2, n=400):
    """I2 (=B のはず、実装検証用) と I4 (Manton (9.62) 用)"""
    u, w = np.polynomial.legendre.leggauss(n)
    t = np.sqrt((1.0 - u) / (1.0 + u))
    Lam = B * t ** (B - 1) * (1.0 + t ** 2) / (1.0 + t ** (2 * B))
    return 0.5 * float(np.sum(w * Lam ** 2)), 0.5 * float(np.sum(w * Lam ** 4))


class ProfileNet(nn.Module):
    """f(r) = (1-s)^p (π + s·h(s)), s = r/(a+r)。f(0)=π と遠方漸近形 r^-p を内蔵。"""
    def __init__(self, B=2, a=2.0, hidden=32, n_layers=2):
        super().__init__()
        self.a = a
        self.p = 0.5 * (1.0 + math.sqrt(1.0 + 8.0 * B))
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1); self.net[-1].bias.zero_()

    def forward(self, r):
        s = r / (self.a + r)
        h = self.net(s.reshape(-1, 1)).reshape(r.shape)
        return (1.0 - s) ** self.p * (PI + s * h)


_QUAD1D_CACHE = {}

def _quad_1d(n_r, a_map, r_max, dev):
    key = (n_r, a_map, r_max, str(dev))
    if key not in _QUAD1D_CACHE:
        t, w = np.polynomial.legendre.leggauss(n_r)
        s_max = r_max / (a_map + r_max)
        s = 0.5 * s_max * (t + 1.0); w = 0.5 * s_max * w
        r_np = a_map * s / (1.0 - s)
        jac = a_map / (1.0 - s) ** 2
        r = torch.tensor(r_np, dtype=torch.float64, device=dev).requires_grad_(True)
        _QUAD1D_CACHE[key] = (r, torch.tensor(w * jac, dtype=torch.float64, device=dev))
    return _QUAD1D_CACHE[key]


def energy_1d(pmodel, B=2, I_val=None, n_r=400, a_map=2.0, r_max=200.0, dev="cpu"):
    """Manton (9.62) の 1 次元換算エネルギー(有理写像 R=z^B 用)"""
    if I_val is None:
        I_val = angular_integrals(B)[1]
    r, wq = _quad_1d(n_r, a_map, r_max, dev)
    f = pmodel(r)
    fp = torch.autograd.grad(f, r, torch.ones_like(f), create_graph=True)[0]
    sf2 = torch.sin(f) ** 2
    integ = r ** 2 * fp ** 2 + 2.0 * B * sf2 * (fp ** 2 + 1.0) + I_val * sf2 ** 2 / r ** 2
    return (wq * integ).sum() / (3.0 * PI)


def _fit_profile_1d(pm, B, I_val, epochs, lbfgs_iter, lr, n_r, dev1d, verbose):
    """train_profile_1d の内側: Adam 粗最適化 → L-BFGS 仕上げ。E_1d を返す。"""
    opt = torch.optim.Adam(pm.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        E = energy_1d(pm, B=B, I_val=I_val, n_r=n_r, dev=dev1d)
        E.backward(); opt.step()
        if verbose and (ep % 500 == 0 or ep == 1):
            print(f"    [profile-1d Adam ep{ep:5d}] E_1d = {E.item():.7f}")
    optl = torch.optim.LBFGS(pm.parameters(), max_iter=lbfgs_iter, history_size=50,
                             tolerance_grad=1e-14, tolerance_change=1e-16,
                             line_search_fn="strong_wolfe")
    def closure():
        optl.zero_grad()
        E = energy_1d(pm, B=B, I_val=I_val, n_r=n_r, dev=dev1d)
        E.backward(); return E
    optl.step(closure)
    return energy_1d(pm, B=B, I_val=I_val, n_r=n_r, dev=dev1d).item()


def train_profile_1d(B=2, epochs=1500, lbfgs_iter=400, lr=5e-3, a=2.0,
                     n_r=400, device="cpu", verbose=True, restart_seeds=(0, 1)):
    """Adam 粗最適化 → L-BFGS 仕上げ。戻り値 (pmodel(float,on device), E_1d, I4)

    L-BFGS は初期値によって劣った局所解に落ちることがある(B=2 で E_1d=2.416352、正しくは 2.416270)。
    そこで (1) 呼び出し時点の乱数状態で 1 本学習し、(2) 続けて fork_rng の中で固定シード restart_seeds から
    再スタートし、(3) E_1d が最小のプロファイルを採用する。(2) は呼び出し側の乱数列を消費しない。"""
    I_val = angular_integrals(B)[1]
    dev1d = "cpu"
    pm = ProfileNet(B=B, a=a).to(dev1d).double()
    E_best = _fit_profile_1d(pm, B, I_val, epochs, lbfgs_iter, lr, n_r, dev1d, verbose)
    for s in restart_seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(s)
            pm_s = ProfileNet(B=B, a=a).to(dev1d).double()
            E_s = _fit_profile_1d(pm_s, B, I_val, epochs, lbfgs_iter, lr, n_r, dev1d, verbose=False)
        if verbose:
            print(f"    [profile-1d restart seed={s}] E_1d = {E_s:.7f}")
        # 採用閾値 1e-6: 正常に収束した場合(E_1d の差 ~1e-8)は最初の解を保ち、劣った局所解(差 8e-5)だけを置き換える。
        # 閾値 1e-9 では、E_1d が実質同じでも別のプロファイルに切り替わり、anchored の E が最大 1e-4 変わる
        # (2026-09-07_retraining の 5 本は 1e-9 の版で作られた。MANIFEST.md の Code versions)。
        if E_s < E_best - 1e-6:
            if verbose:
                print(f"    [profile-1d] 再スタート解を採用 ({E_best:.7f} -> {E_s:.7f})")
            pm, E_best = pm_s, E_s
    if verbose:
        print(f"    [profile-1d L-BFGS 後] E_1d = {E_best:.7f}")
    return pm.float().to(device), E_best, I_val


# ----------------------------------------------------------------------
# 場の構成
# ----------------------------------------------------------------------
def q_base_field(x, pmodel, B=2):
    """有理写像アンザッツ q_base = (cos f, sin f · n_R)"""
    f = pmodel(x.norm(dim=1))
    n = rational_map_direction(x, B=B)
    return torch.cat([torch.cos(f).unsqueeze(1), torch.sin(f).unsqueeze(1) * n], dim=1)


def envelope(x, a=3.0, power=3.0):
    """hard トラック用: g(0)=g(∞)=0(境界条件とトポロジーを壊さない補正窓)"""
    s = x.norm(dim=1) / (a + x.norm(dim=1))
    return 4.0 * s * (1.0 - s) ** power


def envelope_soft(x, a=3.0, power=2.0):
    """soft トラック用: g(0)=1, g(∞)=0(原点自由・遠方は真空 U=+1 を保証)。
    (1-s)^power ~ r^-power 減衰。power=2 はピオンテール r^-2 と整合。"""
    s = x.norm(dim=1) / (a + x.norm(dim=1))
    return (1.0 - s) ** power


def build_q_hard(x, net, pmodel, B=2, env_a=3.0, env_p=3.0):
    """q = normalize(q_base + g·NN(u)): 境界条件・トポロジー(=B)が構成的に厳密"""
    out = net(features(x))
    q_raw = q_base_field(x, pmodel, B=B) + envelope(x, env_a, env_p).unsqueeze(1) * out
    return q_raw / (q_raw.norm(dim=1, keepdim=True) + 1e-12)


def build_q_soft(x, net, scale=3.0, env_a=3.0, env_p=2.0):
    """soft トラック: q = normalize(q_vac + g_soft·scale·NN(u))。
    遠方 U→+1 のみ構成的に保証し、トポロジーは自由(B は損失で誘導)。
    この構成では B が量子化されて凍結し、
    勾配法で B=0 セクターから出られない(論文 6.3 節)。"""
    out = net(features(x))
    q_raw = scale * envelope_soft(x, env_a, env_p).unsqueeze(1) * out
    q_raw = q_raw + torch.tensor([1.0, 0, 0, 0], device=x.device, dtype=x.dtype)
    return q_raw / (q_raw.norm(dim=1, keepdim=True) + 1e-12)


def quat_mul(a, b):
    """単位四元数(=SU(2))の積。a,b: (N,4) [w,x,y,z] 規約"""
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=1)


def product_lumps_q(x, centers, isorots):
    """分離した粗い B=1 hedgehog 塊(f=π e^{-r})の積(product ansatz)。
    U = U1·U2·…(四元数積)。トポロジー B=塊数 のみを持ち込み、
    プロファイルは最適化されておらず、配置・iso回転が非対称なら
    最終解の対称性は一切含まない(自発的対称性形成の初期データ)。
    centers: (n,3) tensor, isorots: (n,3,3) tensor"""
    q = None
    for i in range(centers.shape[0]):
        dx = x - centers[i]
        r = dx.norm(dim=1)
        n = dx / (r.unsqueeze(1) + 1e-9)
        n = n @ isorots[i].T
        f = PI * torch.exp(-r)
        qi = torch.cat([torch.cos(f).unsqueeze(1),
                        torch.sin(f).unsqueeze(1) * n], dim=1)
        q = qi if q is None else quat_mul(q, qi)
    return q


def build_q_free(x, net):
    """free トラック(真に ansatz-free): q = normalize(NN(u))。
    境界条件は課さず、遠方真空・原点反転はすべて損失で誘導する。
    遠方場が厳密真空でないため B は学習中は量子化されず、
    トポロジカル電荷が「無限遠から流入」できる。"""
    out = net(features(x))
    return out / (out.norm(dim=1, keepdim=True) + 1e-12)


# ----------------------------------------------------------------------
# 密度と積分(ひずみテンソル表示、E2/E4 分離)
# ----------------------------------------------------------------------
def densities_split(x, q, create_graph=True):
    """バリオン密度 b、2次項密度 e2、Skyrme項密度 e4(単位 12π²)"""
    ones = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    J = torch.stack([torch.autograd.grad(q[:, k], x, ones, create_graph=create_graph,
                                         retain_graph=True)[0] for k in range(4)], dim=1)
    D = torch.einsum('nki,nkj->nij', J, J)
    trD = D.diagonal(dim1=1, dim2=2).sum(dim=1)
    trD2 = (D * D).sum(dim=(1, 2))
    e2 = trD / (12.0 * PI ** 2)
    e4 = 0.5 * (trD ** 2 - trD2) / (12.0 * PI ** 2)
    M = torch.cat([q.unsqueeze(2), J], dim=2)
    b = -torch.linalg.det(M) / (2.0 * PI ** 2)
    return b, e2, e4


def integrate_BE(x, w, q, sum_dtype=torch.float64, create_graph=True):
    """(B, E) を返す(学習用)。総和は float64。"""
    b, e2, e4 = densities_split(x, q, create_graph=create_graph)
    wd = w.to(sum_dtype)
    return ((b.to(sum_dtype) * wd).sum(),
            ((e2 + e4).to(sum_dtype) * wd).sum())


def integrate_BE_split_chunked(x, w, q_builder, chunk=8192, sum_dtype=torch.float64):
    """評価専用の分割積分。(B, E2, E4) を返す。"""
    dev = x.device
    Bt = torch.zeros((), dtype=sum_dtype, device=dev)
    E2t = torch.zeros((), dtype=sum_dtype, device=dev)
    E4t = torch.zeros((), dtype=sum_dtype, device=dev)
    for i in range(0, x.shape[0], chunk):
        xi = x[i:i + chunk].detach().clone().requires_grad_(True)
        wi = w[i:i + chunk].to(sum_dtype)
        b, e2, e4 = densities_split(xi, q_builder(xi), create_graph=False)
        Bt = Bt + (b.to(sum_dtype) * wi).sum()
        E2t = E2t + (e2.to(sum_dtype) * wi).sum()
        E4t = E4t + (e4.to(sum_dtype) * wi).sum()
        del xi, wi, b, e2, e4
    return Bt.item(), E2t.item(), E4t.item()


def eval_multi_grid(q_builder, device, quad=(48, 28, 36), r_max=60.0,
                    a_eval=(2.6, 3.0, 3.4), n_rot=1, chunk=8192, seed=1234):
    """独立な複数の高解像度格子(異なる a_map + ランダム回転)で (B,E2,E4) を評価。
    戻り値: dict(B, E, E2, E4, virial, dE_grids, rows)"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for a in a_eval:
        for _ in range(n_rot):
            x, w = make_quadrature(*quad, a_map=a, r_max=r_max, device=device)
            A = torch.randn(3, 3, generator=g).to(device)
            Q, R = torch.linalg.qr(A)
            Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
            if torch.det(Q) < 0:
                Q = Q.clone(); Q[:, 0] = -Q[:, 0]
            xr = (x.detach() @ Q.T)
            Bv, E2v, E4v = integrate_BE_split_chunked(xr, w, q_builder, chunk=chunk)
            rows.append(dict(a_map=a, B=Bv, E2=E2v, E4=E4v, E=E2v + E4v))
            del x, w, xr
            if device.type == "cuda":
                torch.cuda.empty_cache()
    Es = [r["E"] for r in rows]
    Bm = float(np.mean([r["B"] for r in rows]))
    E2m = float(np.mean([r["E2"] for r in rows]))
    E4m = float(np.mean([r["E4"] for r in rows]))
    return dict(B=Bm, E=E2m + E4m, E2=E2m, E4=E4m,
                virial=E2m / E4m if E4m != 0 else float("nan"),
                dE_grids=float(max(Es) - min(Es)), rows=rows)
