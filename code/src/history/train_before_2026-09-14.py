# -*- coding: utf-8 -*-
"""
学習ルーチン(hard / soft トラック共通)。
損失: L = E + w_topo (B - B_target)²(純変分。停留点は自動的に EOM 解)。
評価: 独立多重格子で B, E2, E4, ΔE を記録(求積悪用チェック+ビリアル検証)。
結果は outdir/<tag>.json と <tag>_hist.npz に保存(キャッシュとして再利用)。
"""
import os, json, time
import numpy as np
import torch

from physics import (RandomizedQuadrature, build_q_hard, build_q_soft,
                     build_q_free, integrate_BE, eval_multi_grid,
                     train_profile_1d, product_lumps_q, random_rotation,
                     E_STAR, E_TABLE)
from models import build_net, count_params

DEFAULTS = dict(epochs=1000, lr=2e-3, w_topo=200.0,
                quad=(32, 20, 24), quad_eval=(48, 28, 36),
                r_max=60.0, eval_chunk=8192, kan_grid=5,
                track="hard", B=2, soft_scale=3.0, init_gain=1.0, log_every=100,
                # soft/free トラック用: 原点アンカー U(0)=-1(並進ゲージ固定。
                # これが無いと真空 B=0 に潰れる)
                w_origin=0.0, n_origin=64,
                # free トラック用: 遠方真空を損失で誘導(r∈[r_far_lo, r_max])
                w_far=0.0, n_far=256, r_far_lo=30.0,
                # トポロジカル・ペナルティ用 B を打ち切り球 r<=r_max_topo で測る
                # (球面が電荷の注入口になり B が位相凍結しない)
                r_max_topo=None,
                # free トラックの初期データ: n_lumps 個の粗い B=1 塊の積に事前回帰
                # (ランダム配置・ランダム iso 回転 → 最終対称性を含まない)
                n_lumps=0, lump_spread=1.5, lump_max_r=2.5,
                pretrain_epochs=1500, pretrain_lr=2e-3, ff_sigma=1.0, ff_m=32,
                mlp_act="tanh",
                # 時間予算固定モード: 指定秒数だけ学習(epochs は上限として扱う)。
                # 最初の 10 エポックで速度を実測し、残り時間から総エポック数を
                # 見積もって cosine スケジューラを構成する(モデル間で公平)。
                time_budget_sec=None)

_PMODEL_CACHE = {}


def get_base_profile(B, device, verbose=False):
    key = (B, str(device))
    if key not in _PMODEL_CACHE:
        pm, E1d, I4 = train_profile_1d(B=B, epochs=1500, device=device, verbose=verbose)
        for p in pm.parameters():
            p.requires_grad_(False)
        _PMODEL_CACHE[key] = (pm, E1d, I4)
    return _PMODEL_CACHE[key]


def tag_of(kind, width, depth, seed, cfg):
    t = f"{cfg['track']}_B{cfg['B']}_{kind}_w{width}_d{depth}_ep{cfg['epochs']}_s{seed}"
    if kind in ("pykan", "effkan", "fastkan"):
        t += f"_g{cfg['kan_grid']}"
    if cfg.get("time_budget_sec"):
        t += f"_tb{int(cfg['time_budget_sec'])}"
    if cfg.get("lr", 2e-3) != 2e-3:
        t += f"_lr{cfg['lr']:g}"
    if kind == "mlp" and cfg.get("mlp_act", "tanh") != "tanh":
        t += f"_{cfg['mlp_act']}"
    if kind == "ffmlp" and cfg.get("ff_m", 32) != 32:
        t += f"_m{cfg['ff_m']}"
    return t


def train_one(kind, width, depth, seed=0, device=None, outdir=None,
              verbose=True, force=False, **kw):
    cfg = dict(DEFAULTS); cfg.update(kw)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = tag_of(kind, width, depth, seed, cfg)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        jpath = os.path.join(outdir, tag + ".json")
        if (not force) and os.path.exists(jpath):
            res = json.load(open(jpath))
            if verbose:
                print(f"  [cache] {tag:44s} E={res['E']:.5f}")
            return res

    torch.manual_seed(seed); np.random.seed(seed)
    Btgt = cfg["B"]

    if cfg["track"] == "hard":
        pmodel, E1d, _ = get_base_profile(Btgt, device)
        q_builder_train = lambda x, net: build_q_hard(x, net, pmodel, B=Btgt)
    elif cfg["track"] == "soft":
        pmodel = None
        q_builder_train = lambda x, net: build_q_soft(x, net, scale=cfg["soft_scale"])
    else:  # free
        pmodel = None
        q_builder_train = lambda x, net: build_q_free(x, net)

    net = build_net(kind, width, depth, device, cfg["kan_grid"],
                    zero_init=(cfg["track"] == "hard"), seed=seed,
                    out_gain=cfg["init_gain"], ff_sigma=cfg["ff_sigma"],
                    ff_m=cfg["ff_m"], mlp_act=cfg["mlp_act"])
    n_par = count_params(net)

    rq = RandomizedQuadrature(*cfg["quad"], r_max=cfg["r_max"], device=device)
    rq_topo = None
    if cfg["r_max_topo"]:
        rq_topo = RandomizedQuadrature(
            *cfg["quad"], r_max=cfg["r_max_topo"],
            a_list=(1.5, 2.0, 2.5, 3.0), device=device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    if cfg["time_budget_sec"]:
        sch = None   # 速度実測後(10 エポック時点)に構成する
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.02)

    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    lumps_info = None
    if cfg["track"] == "free" and cfg["n_lumps"] > 0:
        # ---- 事前回帰: 積アンザッツ(粗い B=1 塊 × n_lumps)に合わせる ----
        centers = cfg["lump_spread"] * torch.randn(cfg["n_lumps"], 3, device=device)
        centers = centers - centers.mean(dim=0, keepdim=True)   # 重心を原点に
        # 遠すぎる塊は u 座標で薄殻に圧縮され表現・求積が破綻する
        # ため半径制限
        rmax = centers.norm(dim=1).max()
        if rmax > cfg["lump_max_r"]:
            centers = centers * (cfg["lump_max_r"] / rmax)
        isorots = torch.stack([random_rotation(device)
                               for _ in range(cfg["n_lumps"])])
        lumps_info = dict(centers=centers.tolist(),
                          sep=[float((centers[i] - centers[j]).norm())
                               for i in range(cfg["n_lumps"])
                               for j in range(i + 1, cfg["n_lumps"])])
        # 正規化「前」の生出力を単位四元数ターゲットに直接回帰する。
        # normalize(NN) を通すと |NN|→0 近傍の勾配特異性で定数場に停留する
        # 失敗モードがある。
        popt = torch.optim.Adam(net.parameters(), lr=cfg["pretrain_lr"])
        from physics import features as _feat
        ploss = None
        for pep in range(cfg["pretrain_epochs"]):
            xs = 15.0 * (torch.rand(2048, 3, device=device) * 2 - 1)
            xs = xs[xs.norm(dim=1) < 15.0]
            popt.zero_grad(set_to_none=True)
            raw = net(_feat(xs))
            with torch.no_grad():
                qt = product_lumps_q(xs, centers, isorots)
            ploss = ((raw - qt) ** 2).mean()
            ploss.backward()
            popt.step()
        if verbose:
            print(f"    [pretrain {cfg['pretrain_epochs']}ep 完了 "
                  f"{time.time()-t0:.0f}s] 回帰損失={ploss.item():.2e} "
                  f"塊間距離 {['%.2f' % s for s in lumps_info['sep']]}")
        lumps_info["pretrain_loss"] = float(ploss.item())

    hist_E, hist_B = [], []
    t_main0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        opt.zero_grad(set_to_none=True)
        xq, wq = rq.sample()
        q = q_builder_train(xq, net)
        Bv, Ev = integrate_BE(xq, wq, q)
        if rq_topo is not None:
            xt, wt = rq_topo.sample()
            Bv, _ = integrate_BE(xt, wt, q_builder_train(xt, net))
        loss = Ev + cfg["w_topo"] * (Bv - Btgt) ** 2
        if cfg["track"] in ("soft", "free") and cfg["w_origin"] > 0:
            xo = (0.05 * torch.randn(cfg["n_origin"], 3, device=device)
                  ).requires_grad_(True)
            qo = q_builder_train(xo, net)
            tgt = torch.tensor([-1.0, 0, 0, 0], device=device)
            loss = loss + cfg["w_origin"] * ((qo - tgt) ** 2).mean()
        if cfg["track"] == "free" and cfg["w_far"] > 0:
            d = torch.randn(cfg["n_far"], 3, device=device)
            d = d / d.norm(dim=1, keepdim=True)
            rr = (cfg["r_far_lo"] + (cfg["r_max"] - cfg["r_far_lo"])
                  * torch.rand(cfg["n_far"], 1, device=device))
            xf = (rr * d).requires_grad_(True)
            qf = q_builder_train(xf, net)
            vac = torch.tensor([1.0, 0, 0, 0], device=device)
            loss = loss + cfg["w_far"] * ((qf - vac) ** 2).mean()
        loss.float().backward()
        opt.step()
        if sch is not None:
            sch.step()
        hist_E.append(Ev.item()); hist_B.append(Bv.item())
        if verbose and (ep % cfg["log_every"] == 0 or ep == 1):
            print(f"    ep{ep:5d}  E={Ev.item():.6f}  B={Bv.item():.5f}")
        if cfg["time_budget_sec"]:
            elapsed = time.time() - t_main0
            if sch is None and ep == 10:
                est_total = max(20, int(cfg["time_budget_sec"] / (elapsed / 10)))
                est_total = min(est_total, cfg["epochs"])
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(est_total - 10, 10), eta_min=cfg["lr"] * 0.02)
                if verbose:
                    print(f"    [時間予算 {cfg['time_budget_sec']}s: "
                          f"推定 {est_total} エポック]")
            if elapsed > cfg["time_budget_sec"]:
                break
    t_train = time.time() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 1024 ** 3
                if device.type == "cuda" else 0.0)

    # ---- 独立多重格子での最終評価 ----
    qb_eval = (lambda xi: q_builder_train(xi, net))
    ev = eval_multi_grid(qb_eval, device, quad=cfg["quad_eval"],
                         r_max=cfg["r_max"], chunk=cfg["eval_chunk"])

    e_star = E_STAR.get(Btgt)
    res = dict(tag=tag, kind=kind, width=width, depth=depth, seed=seed,
               track=cfg["track"], B_target=Btgt, n_params=n_par,
               epochs=cfg["epochs"], kan_grid=cfg["kan_grid"],
               B=ev["B"], E=ev["E"], E2=ev["E2"], E4=ev["E4"],
               virial=ev["virial"], dE_grids=ev["dE_grids"],
               B_dev=abs(ev["B"] - round(ev["B"])),
               E_over_B=ev["E"] / abs(ev["B"]) if ev["B"] != 0 else float("nan"),
               rel_err_vs_star=(abs(ev["E"] - e_star) / e_star * 100 if e_star else None),
               fb_bound_ok=bool(ev["E"] >= abs(ev["B"]) - 1e-6),
               time_train_sec=t_train, peak_mem_GiB=peak_mem,
               lumps=lumps_info,
               config={k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.items()})
    if verbose:
        print(f"  {tag:44s} params={n_par:7,d}  E={res['E']:.5f}  "
              f"B={res['B']:.5f}  E2/E4={res['virial']:.4f}  "
              f"ΔE={res['dE_grids']:.1e}  {t_train:.0f}s")
    if outdir:
        json.dump(res, open(os.path.join(outdir, tag + ".json"), "w"),
                  indent=1, ensure_ascii=False)
        np.savez_compressed(os.path.join(outdir, tag + "_hist.npz"),
                            E=np.array(hist_E), B=np.array(hist_B))
        if cfg.get("save_model"):
            torch.save(net.state_dict(),
                       os.path.join(outdir, tag + "_model.pt"))
    del net, opt, rq
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res
