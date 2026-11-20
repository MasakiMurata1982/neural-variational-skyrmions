# -*- coding: utf-8 -*-
"""
ネットワーク定義: MLP / pykan(参照用) / EffKAN(B-spline, efficient-kan方式) /
FastKAN(RBF近似, FastKAN方式 arXiv:2405.06721)。
すべて (3,) -> (4,) の場写像。hard トラックでは最終層を 0 初期化して
「初期状態 = 基底アンザッツ」から出発させる。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


_ACTS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=4, width=16, depth=4, zero_init=True,
                 out_gain=1.0, act="tanh"):
        super().__init__()
        A = _ACTS[act]
        layers = [nn.Linear(in_dim, width), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), A()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            if zero_init:
                self.net[-1].weight.mul_(0.0); self.net[-1].bias.zero_()
            elif out_gain != 1.0:
                self.net[-1].weight.mul_(out_gain)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# EffKAN: efficient-kan 方式の B-spline KAN
# ----------------------------------------------------------------------
class EffKANLayer(nn.Module):
    def __init__(self, in_f, out_f, grid=5, k=3, grid_range=(-1.0, 1.0),
                 init_scale=0.1):
        super().__init__()
        self.in_f, self.out_f, self.k = in_f, out_f, k
        h = (grid_range[1] - grid_range[0]) / grid
        knots = torch.arange(-k, grid + k + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer("knots", knots)
        self.base_weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.spline_weight = nn.Parameter(
            init_scale * torch.randn(out_f, in_f, grid + k) / math.sqrt(in_f))

    def b_splines(self, x):
        g = self.knots
        x = x.unsqueeze(-1)
        bases = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for j in range(1, self.k + 1):
            bases = ((x - g[: -(j + 1)]) / (g[j:-1] - g[: -(j + 1)]) * bases[..., :-1]
                     + (g[j + 1:] - x) / (g[j + 1:] - g[1:-j]) * bases[..., 1:])
        return bases                                    # (N, in_f, grid+k)

    def forward(self, x):
        base = F.silu(x) @ self.base_weight.T
        spl = torch.einsum("nik,oik->no", self.b_splines(x), self.spline_weight)
        return base + spl


class EffKAN(nn.Module):
    def __init__(self, widths, grid=5, k=3, zero_init=True, out_gain=1.0):
        super().__init__()
        self.layers = nn.ModuleList([EffKANLayer(a, b, grid, k)
                                     for a, b in zip(widths[:-1], widths[1:])])
        with torch.no_grad():
            if zero_init:
                self.layers[-1].base_weight.mul_(0.0)
                self.layers[-1].spline_weight.mul_(0.0)
            elif out_gain != 1.0:
                self.layers[-1].base_weight.mul_(out_gain)
                self.layers[-1].spline_weight.mul_(out_gain)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# ----------------------------------------------------------------------
# FastKAN: B-spline を RBF(ガウス基底)で置き換えた高速近似 + LayerNorm で
# 入力を基底の定義域に保つ(arXiv:2405.06721 の構成に準拠)
# ----------------------------------------------------------------------
class FastKANLayer(nn.Module):
    def __init__(self, in_f, out_f, num_grids=8, grid_range=(-2.0, 2.0),
                 use_norm=True, init_scale=0.1):
        super().__init__()
        centers = torch.linspace(grid_range[0], grid_range[1], num_grids)
        self.register_buffer("centers", centers)
        self.inv_denom = num_grids / (grid_range[1] - grid_range[0])
        self.norm = nn.LayerNorm(in_f) if use_norm else nn.Identity()
        self.base_weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.rbf_weight = nn.Parameter(
            init_scale * torch.randn(out_f, in_f, num_grids) / math.sqrt(in_f))

    def forward(self, x):
        base = F.silu(x) @ self.base_weight.T
        xn = self.norm(x)
        phi = torch.exp(-((xn.unsqueeze(-1) - self.centers) * self.inv_denom) ** 2)
        return base + torch.einsum("nik,oik->no", phi, self.rbf_weight)


class FastKAN(nn.Module):
    def __init__(self, widths, num_grids=8, zero_init=True):
        super().__init__()
        self.layers = nn.ModuleList([FastKANLayer(a, b, num_grids)
                                     for a, b in zip(widths[:-1], widths[1:])])
        if zero_init:
            with torch.no_grad():
                self.layers[-1].base_weight.mul_(0.0)
                self.layers[-1].rbf_weight.mul_(0.0)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# ----------------------------------------------------------------------
def make_pykan(widths, grid=5, k=3, device="cpu", zero_init=True, seed=1):
    """pykan(参照実装)。auto_save 無効・save_act 無効。
    注意: pykan は __init__ 内で torch.manual_seed(seed) を呼び RNG をリセットする(既定 seed=1)。このため seed を渡す。
    論文の pykan の 15 本は seed を渡さない版で作られ、各構成の 3 シードは同じ run である(MANIFEST.md の Code versions)。"""
    from kan import KAN
    net = KAN(width=list(widths), grid=grid, k=k, device=device,
              save_act=False, auto_save=False, seed=seed)
    if zero_init:
        with torch.no_grad():
            last = net.act_fun[-1]
            for nm in ("coef", "scale_base", "scale_sp"):
                if hasattr(last, nm):
                    getattr(last, nm).mul_(0.0)
    return net


def build_net(kind, width, depth, device, kan_grid=5, zero_init=True, seed=1,
              out_gain=1.0, ff_sigma=1.0, ff_m=32, mlp_act="tanh"):
    """kind in {mlp, pykan, effkan, fastkan}(seed は pykan のみ、mlp_act は mlp のみ。
    out_gain は最終層の初期値の倍率で、論文の run はすべて 1.0。ff_sigma と ff_m は使わない: run の記録の config にあるキー)"""
    widths = [3] + [width] * depth + [4]
    if kind == "mlp":
        return MLP(width=width, depth=depth, zero_init=zero_init,
                   out_gain=out_gain, act=mlp_act).to(device)
    if kind == "pykan":
        return make_pykan(widths, grid=kan_grid, device=device,
                          zero_init=zero_init, seed=seed)
    if kind == "effkan":
        return EffKAN(widths, grid=kan_grid, zero_init=zero_init,
                      out_gain=out_gain).to(device)
    if kind == "fastkan":
        # FastKAN の num_grids は pykan の grid+k(B-spline係数数)に合わせる
        return FastKAN(widths, num_grids=kan_grid + 3, zero_init=zero_init).to(device)
    raise ValueError(kind)
