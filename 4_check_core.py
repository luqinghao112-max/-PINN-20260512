"""
第 4 步：用物理正演内核检查单条曲线。

这个脚本专门检查 Python 物理核与 MATLAB 正演模型的一致性。
它不再使用“局部 Darcy 梯度 + 时域井储”的近似，而是直接复刻
S2_Singleprobe_Sampling_V14_compute_curve.m 中均质模型 model_choice=1 的
拉氏空间正演公式：

    1. Ke = (Kh^2 * Kv)^(1/3) 各向异性坐标变换
    2. 有限面积探针离散和 receiver-source 双面积平均
    3. Stehfest 拉普拉斯反演
    4. 拉氏空间表皮与井储映射
    5. Bourdet_Smooth.m 一致的压力导数

注意：
    这是“物理反演基准”，不是神经网络压力场 PINN。
    之所以单独做这个脚本，是因为 PyTorch 当前 Bessel K/I 函数不能自动求导，
    而 MATLAB 正演模型的关键核函数正依赖这些 Bessel 函数。强行用不一致 PDE 约束
    会得到能拟合曲线但代回 MATLAB 失真的参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import k0


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


PARAM_BOUNDS = {
    "logKh": (math.log10(50.0), math.log10(3000.0)),
    "logKv": (math.log10(1.0), math.log10(300.0)),
    "S": (-3.0, 8.0),
    "logC": (math.log10(1e-10), math.log10(1e-6)),
}


def load_mat_dataset(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取 MATLAB v7.3 MAT 文件。"""
    with h5py.File(mat_path, "r") as f:
        X = np.asarray(f["X"], dtype=np.float64).T
        Y = np.asarray(f["Y"], dtype=np.float64).T
        D = np.asarray(f["D"], dtype=np.float64).T
        t_hours = np.asarray(f["t_hours"], dtype=np.float64).reshape(-1)
    return X, Y, D, t_hours


def load_initial_guess(csv_path: Path, sample_idx: int, x_true: np.ndarray) -> np.ndarray:
    """优先使用代理模型反演结果作为初值，缺失时使用真实值只做调试兜底。"""
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row["sample_idx"])) == int(sample_idx):
                    return np.array([
                        math.log10(float(row["est_Kh_mD"])),
                        math.log10(float(row["est_Kv_mD"])),
                        float(row["est_S"]),
                        math.log10(float(row["est_C_m3_per_MPa"])),
                    ], dtype=np.float64)
    return np.array([
        math.log10(float(x_true[0])),
        math.log10(float(x_true[1])),
        float(x_true[2]),
        math.log10(float(x_true[3])),
    ], dtype=np.float64)


def stehfest_weights(n: int) -> np.ndarray:
    """与 LaplaceInvP.m 一致的 Stehfest 权重。"""
    if n % 2 != 0:
        raise ValueError("Stehfest N 必须为偶数")
    weights = np.zeros(n, dtype=np.float64)
    for i in range(1, n + 1):
        total = 0.0
        for k in range((i + 1) // 2, min(i, n // 2) + 1):
            total += (
                k ** (n // 2)
                * math.factorial(2 * k)
                / (
                    math.factorial(n // 2 - k)
                    * math.factorial(k)
                    * math.factorial(k - 1)
                    * math.factorial(i - k)
                    * math.factorial(2 * k - i)
                )
            )
        weights[i - 1] = (-1) ** (n // 2 + i) * total
    return weights


def build_probe_points_from_radius_v14(rp_cm: float, hw_center_cm: float, n_target: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """完全按 V14 的 circle 探针规则生成有限面积离散点。"""
    ngrid = max(5, math.ceil(math.sqrt(n_target * 4.0 / math.pi)) + 2)
    yy = np.linspace(-rp_cm, rp_cm, ngrid)
    zz = np.linspace(-rp_cm, rp_cm, ngrid)
    y_grid, z_grid = np.meshgrid(yy, zz)
    mask = (y_grid ** 2 + z_grid ** 2) <= rp_cm ** 2
    y_cm = y_grid[mask].reshape(-1)
    z_cm = (hw_center_cm + z_grid[mask]).reshape(-1)
    w = np.ones_like(y_cm) / len(y_cm)
    return y_cm, z_cm, w


def bourdet_smooth(t: np.ndarray, p: np.ndarray, window_log10: float = 0.1) -> np.ndarray:
    """与 Bourdet_Smooth.m 一致的压力导数 dP/d(ln t)。"""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    n = len(t)
    d_p = np.zeros_like(p)
    if n < 3:
        return d_p

    log10_t = np.log10(t)
    ln_t = np.log(t)
    for i in range(n):
        d_left = log10_t[i] - log10_t[0]
        d_right = log10_t[-1] - log10_t[i]
        l_eff = min(window_log10, d_left, d_right) + 1e-5

        k = i
        while k > 0 and (log10_t[i] - log10_t[k]) < l_eff:
            k -= 1
        j = i
        while j < n - 1 and (log10_t[j] - log10_t[i]) < l_eff:
            j += 1

        if k == i and i > 0:
            k = i - 1
        if j == i and i < n - 1:
            j = i + 1

        dx_l = ln_t[i] - ln_t[k]
        dx_r = ln_t[j] - ln_t[i]
        m_l = 0.0 if dx_l == 0.0 else (p[i] - p[k]) / dx_l
        m_r = 0.0 if dx_r == 0.0 else (p[j] - p[i]) / dx_r

        if dx_l == 0.0 and dx_r == 0.0:
            d_p[i] = 0.0
        elif dx_l == 0.0:
            d_p[i] = m_r
        elif dx_r == 0.0:
            d_p[i] = m_l
        else:
            d_p[i] = (m_l * dx_r + m_r * dx_l) / (dx_l + dx_r)
    return d_p


def make_v14_geometry(params: Dict[str, float], known: Dict[str, float], opts: argparse.Namespace) -> Dict[str, np.ndarray | float]:
    """构造 V14 均质模型所需的几何、尺度和有限面积平均项。"""
    kh = params["Kh_mD"]
    kv = params["Kv_mD"]
    skin = params["S"]
    ke = (kh ** 2 * kv) ** (1.0 / 3.0)

    as_cm2 = known["As_in2"] * 2.54 ** 2
    rp_cm = math.sqrt(as_cm2 / math.pi)
    h_cm = known["h_m"] * 100.0
    hw_cm = known["hw_m"] * 100.0
    length_cm = h_cm

    if skin < 0.0:
        rp_skin = rp_cm * math.exp(-skin)
        rp_effective = min(rp_skin, opts.negative_skin_max_factor * rp_cm)
        skin_residual = 0.0
    else:
        rp_effective = rp_cm
        skin_residual = skin

    if (hw_cm - rp_effective) <= 0 or (hw_cm + rp_effective) >= h_cm:
        raise ValueError("探针有限面积超出上下边界")

    scale_r1 = math.sqrt(ke / kh)
    scale_z1 = math.sqrt(ke / kv)
    h_d = h_cm * scale_z1 / length_cm
    c_d = (
        params["C_m3_per_MPa"]
        / (2.0 * math.pi * known["phi"] * known["ct_1_per_MPa"] * known["h_m"] * (length_cm / 100.0) ** 2)
        * (ke / kh)
    )

    y_probe_cm, z_probe_cm, w_probe = build_probe_points_from_radius_v14(
        rp_effective, hw_cm, opts.np_probe_target
    )
    yr, ys = np.meshgrid(y_probe_cm, y_probe_cm, indexing="ij")
    zr, zs = np.meshgrid(z_probe_cm, z_probe_cm, indexing="ij")
    wr, ws = np.meshgrid(w_probe, w_probe, indexing="ij")

    pair_w = (wr.reshape(-1) * ws.reshape(-1))
    pair_w = pair_w / np.sum(pair_w)
    probe_point_count = len(w_probe)
    cell_radius_cm = math.sqrt((math.pi * rp_effective ** 2 / probe_point_count) / math.pi)
    rho_cm = np.sqrt((yr.reshape(-1) - ys.reshape(-1)) ** 2 + cell_radius_cm ** 2)
    pair_rho_d = rho_cm * scale_r1 / length_cm
    pair_zr_frac = zr.reshape(-1) / h_cm
    pair_zs_frac = zs.reshape(-1) / h_cm

    z_cache = np.zeros((len(pair_w), opts.series_max_terms), dtype=np.float64)
    for n in range(1, opts.series_max_terms + 1):
        z_cache[:, n - 1] = np.cos(n * math.pi * pair_zr_frac) * np.cos(n * math.pi * pair_zs_frac)

    return {
        "Ke_mD": ke,
        "hD": h_d,
        "C_D": c_d,
        "skin_residual": skin_residual,
        "pair_w": pair_w,
        "pair_rhoD": pair_rho_d,
        "z_cache": z_cache,
        "length_cm": length_cm,
        "h_cm": h_cm,
        "pressure_scale": (101.325 / (2.0 * math.pi)) * known["q_cm3_per_s"] * known["mu_cP"] / (kh * h_cm),
        "time_scale_sec": 101.325 * (length_cm ** 2 * known["phi"] * known["ct_1_per_MPa"] * known["mu_cP"]) / ke,
    }


def v14_laplace_homogeneous(s: np.ndarray, geom: Dict[str, np.ndarray | float], opts: argparse.Namespace) -> np.ndarray:
    """V14 model_choice=1 的拉氏空间无因次井底压力。"""
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    pair_w = geom["pair_w"]
    pair_rho_d = geom["pair_rhoD"]
    z_cache = geom["z_cache"]
    h_d = float(geom["hD"])
    c_d = float(geom["C_D"])
    skin = float(geom["skin_residual"])

    # n=0 项：V14 中 A=0，因此只有 K0 无限域核。
    p_sum = np.sum(pair_w[None, :] * k0(np.sqrt(s)[:, None] * pair_rho_d[None, :]), axis=1)

    # n>=1 项：保持 V14 的逐 s 相对误差截断逻辑，但对仍活跃的 s 批量计算。
    consecutive_small = np.zeros_like(s, dtype=np.int32)
    active = np.ones_like(s, dtype=bool)
    n = 1
    while n <= opts.series_max_terms and np.any(active):
        idx = np.where(active)[0]
        term_z = (n * math.pi / h_d) ** 2
        g = np.sqrt(s[idx] + term_z)
        add_sum = 2.0 * np.sum(
            pair_w[None, :] * k0(g[:, None] * pair_rho_d[None, :]) * z_cache[None, :, n - 1],
            axis=1,
        )
        p_sum[idx] += add_sum

        small = np.abs(add_sum) < opts.series_tol * np.maximum(1.0, np.abs(p_sum[idx]))
        consecutive_small[idx] = np.where(small, consecutive_small[idx] + 1, 0)
        if n >= opts.series_min_terms:
            active[idx] = consecutive_small[idx] < opts.series_small_count
        n += 1

    p_d_form = p_sum / s
    p_d_skin = p_d_form + skin / s
    return p_d_skin / (1.0 + c_d * s ** 2 * p_d_skin)


def v14_forward_pressure(theta: np.ndarray, t_hours: np.ndarray, known: Dict[str, float], opts: argparse.Namespace) -> np.ndarray:
    """用 Python 复刻 V14 均质模型，返回压力 MPa。"""
    params = theta_to_params(theta)
    geom = make_v14_geometry(params, known, opts)
    t_d = (t_hours * 3600.0) / float(geom["time_scale_sec"])
    weights = stehfest_weights(opts.stehfest_n)
    p_d = np.zeros_like(t_d, dtype=np.float64)
    ln2 = math.log(2.0)
    for i, weight in enumerate(weights, start=1):
        s = ln2 * i / t_d
        p_d += weight * v14_laplace_homogeneous(s, geom, opts)
    p_d = ln2 * p_d / t_d
    return float(geom["pressure_scale"]) * p_d


def theta_to_params(theta: np.ndarray) -> Dict[str, float]:
    """优化变量转物理参数。"""
    return {
        "Kh_mD": 10.0 ** float(theta[0]),
        "Kv_mD": 10.0 ** float(theta[1]),
        "S": float(theta[2]),
        "C_m3_per_MPa": 10.0 ** float(theta[3]),
    }


def residual_pressure(theta: np.ndarray, t_hours: np.ndarray, p_obs: np.ndarray, known: Dict[str, float], opts: argparse.Namespace) -> np.ndarray:
    """以 log 压力差为残差，降低大压差平台对优化的绝对支配。"""
    try:
        p_pred = v14_forward_pressure(theta, t_hours, known, opts)
    except Exception:
        return np.full_like(p_obs, 1e6, dtype=np.float64)
    if np.any(~np.isfinite(p_pred)) or np.any(p_pred <= 0):
        return np.full_like(p_obs, 1e6, dtype=np.float64)
    return np.log(np.maximum(p_pred, 1e-12)) - np.log(np.maximum(p_obs, 1e-12))


def objective_pressure(theta: np.ndarray, t_hours: np.ndarray, p_obs: np.ndarray, known: Dict[str, float], opts: argparse.Namespace) -> float:
    """全局搜索使用的标量目标函数。"""
    r = residual_pressure(theta, t_hours, p_obs, known, opts)
    return float(np.mean(r ** 2))


def invert_one_curve(args: argparse.Namespace) -> Dict[str, float]:
    """执行 V14 一致单曲线反演。"""
    X, Y, D, t_hours = load_mat_dataset(args.mat)
    x_true = X[args.sample_idx]
    p_obs = Y[args.sample_idx]
    d_obs = D[args.sample_idx]
    mask = np.isfinite(p_obs) & (p_obs > 0)
    t_hours = t_hours[mask]
    p_obs = p_obs[mask]
    d_obs = d_obs[mask]

    known = {
        "phi": 0.2,
        "ct_1_per_MPa": 5e-4,
        "h_m": 20.0,
        "q_cm3_per_s": float(x_true[4]) * 1e6 / 86400.0,
        "mu_cP": float(x_true[5]),
        "As_in2": float(x_true[6]),
        "hw_m": float(x_true[7]),
    }
    x0 = load_initial_guess(args.init_csv, args.sample_idx, x_true)
    bounds = np.array([
        PARAM_BOUNDS["logKh"],
        PARAM_BOUNDS["logKv"],
        PARAM_BOUNDS["S"],
        PARAM_BOUNDS["logC"],
    ], dtype=np.float64)

    print("Initial guess:")
    print(json.dumps(theta_to_params(x0), indent=2, ensure_ascii=False))

    if args.global_search:
        print("Running differential evolution global search...")
        de = differential_evolution(
            objective_pressure,
            bounds=bounds,
            args=(t_hours, p_obs, known, args),
            seed=args.seed,
            maxiter=args.de_maxiter,
            popsize=args.de_popsize,
            polish=False,
            updating="immediate",
            workers=1,
            tol=1e-5,
        )
        x0 = de.x
        print(f"DE objective={de.fun:.6e}")

    print("Running least_squares local refinement...")
    ls = least_squares(
        residual_pressure,
        x0=x0,
        bounds=(bounds[:, 0], bounds[:, 1]),
        args=(t_hours, p_obs, known, args),
        max_nfev=args.max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        verbose=2 if args.verbose else 0,
    )

    params_est = theta_to_params(ls.x)
    p_fit = v14_forward_pressure(ls.x, t_hours, known, args)
    d_fit = bourdet_smooth(t_hours, p_fit, args.derivative_window)

    rel = np.abs(p_fit - p_obs) / np.maximum(np.abs(p_obs), 1e-12)
    d_logerr = np.abs(np.log10(np.maximum(d_fit, 1e-12)) - np.log10(np.maximum(d_obs, 1e-12)))
    result = {
        "sample_idx": int(args.sample_idx),
        "success": bool(ls.success),
        "message": str(ls.message),
        "objective_mse_log_pressure": float(np.mean(residual_pressure(ls.x, t_hours, p_obs, known, args) ** 2)),
        "pressure_mean_relerr": float(np.nanmean(rel)),
        "pressure_p95_relerr": float(np.nanpercentile(rel, 95)),
        "derivative_median_log10err": float(np.nanmedian(d_logerr)),
        "true": {
            "Kh_mD": float(x_true[0]),
            "Kv_mD": float(x_true[1]),
            "S": float(x_true[2]),
            "C_m3_per_MPa": float(x_true[3]),
        },
        "estimated": params_est,
    }
    save_outputs(args, result, t_hours, p_obs, p_fit, d_obs, d_fit)
    return result


def save_outputs(
    args: argparse.Namespace,
    result: Dict[str, float],
    t_hours: np.ndarray,
    p_obs: np.ndarray,
    p_fit: np.ndarray,
    d_obs: np.ndarray,
    d_fit: np.ndarray,
) -> None:
    """保存 JSON、CSV 和科研对比图。"""
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    result_path = args.outdir / f"result_{args.sample_idx}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = args.outdir / f"result_{args.sample_idx}.csv"
    flat = {
        "sample_idx": result["sample_idx"],
        "pressure_mean_relerr": result["pressure_mean_relerr"],
        "pressure_p95_relerr": result["pressure_p95_relerr"],
        "derivative_median_log10err": result["derivative_median_log10err"],
        "true_Kh_mD": result["true"]["Kh_mD"],
        "est_Kh_mD": result["estimated"]["Kh_mD"],
        "true_Kv_mD": result["true"]["Kv_mD"],
        "est_Kv_mD": result["estimated"]["Kv_mD"],
        "true_S": result["true"]["S"],
        "est_S": result["estimated"]["S"],
        "true_C_m3_per_MPa": result["true"]["C_m3_per_MPa"],
        "est_C_m3_per_MPa": result["estimated"]["C_m3_per_MPa"],
    }
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    fig_path = args.figdir / f"fit_{args.sample_idx}.png"
    plt.figure(figsize=(8.5, 5.8))
    plt.loglog(t_hours, np.maximum(p_obs, 1e-12), "k-", lw=1.8, label="Original pressure")
    plt.loglog(t_hours, np.maximum(p_fit, 1e-12), "r--", lw=1.6, label="Python physical fit")
    plt.loglog(t_hours, np.maximum(d_obs, 1e-12), color="0.35", ls="-.", lw=1.4, label="Original derivative")
    plt.loglog(t_hours, np.maximum(d_fit, 1e-12), color="m", ls=":", lw=1.8, label="Fit derivative")
    plt.xlabel("Time t [h]")
    plt.ylabel("Pressure drop and derivative [MPa]")
    plt.title(
        f"Physical inversion sample {args.sample_idx} | "
        f"pressure mean rel={result['pressure_mean_relerr']:.3%}"
    )
    txt = (
        "True / estimated\n"
        f"Kh: {result['true']['Kh_mD']:.4g} / {result['estimated']['Kh_mD']:.4g} mD\n"
        f"Kv: {result['true']['Kv_mD']:.4g} / {result['estimated']['Kv_mD']:.4g} mD\n"
        f"S : {result['true']['S']:.4g} / {result['estimated']['S']:.4g}\n"
        f"C : {result['true']['C_m3_per_MPa']:.3g} / {result['estimated']['C_m3_per_MPa']:.3g} m3/MPa"
    )
    plt.text(
        0.57, 0.12, txt, transform=plt.gca().transAxes,
        fontsize=10, family="monospace",
        bbox=dict(facecolor="white", edgecolor="0.25", alpha=0.92),
    )
    plt.grid(True, which="both", alpha=0.28)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    print("\nSaved:")
    print(f"  {result_path}")
    print(f"  {csv_path}")
    print(f"  {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=PROJECT_DIR / "data_2000.mat")
    parser.add_argument("--init-csv", type=Path, default=SCRIPT_DIR / "init.csv")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "core_out")
    parser.add_argument("--figdir", type=Path, default=SCRIPT_DIR / "core_fig")
    parser.add_argument("--sample-idx", type=int, default=998)
    parser.add_argument("--stehfest-n", type=int, default=10)
    parser.add_argument("--np-probe-target", type=int, default=5)
    parser.add_argument("--series-min-terms", type=int, default=10)
    parser.add_argument("--series-max-terms", type=int, default=80)
    parser.add_argument("--series-tol", type=float, default=1e-6)
    parser.add_argument("--series-small-count", type=int, default=3)
    parser.add_argument("--negative-skin-max-factor", type=float, default=20.0)
    parser.add_argument("--derivative-window", type=float, default=0.1)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--global-search", action="store_true", help="先做差分进化全局搜索，再做最小二乘局部优化")
    parser.add_argument("--de-maxiter", type=int, default=18)
    parser.add_argument("--de-popsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = invert_one_curve(parse_args())
    print("\nResult:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
