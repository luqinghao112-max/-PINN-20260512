"""
第 7 步：批量验证网络反演结果。

推荐先跑测试集：

    python 7_check_batch.py --split test --phys-check all

脚本输出：
    1. results_<split>.csv
    2. summary_<split>.json
    3. 参数交会图、误差直方图、箱线图、物理复查质量图等

说明：
    训练脚本按曲线编号划分 train/val/test。
    本脚本默认使用 test split，因此这些曲线没有参与网络训练。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CORE_PATH = SCRIPT_DIR / "4_check_core.py"

PARAM_BOUNDS = {
    "logKh": (math.log10(50.0), math.log10(3000.0)),
    "logKv": (math.log10(1.0), math.log10(300.0)),
    "S": (-3.0, 8.0),
    "logC": (math.log10(1e-10), math.log10(1e-6)),
}


class OperatorMLP(nn.Module):
    """与 3_train_net.py 保持一致的网络结构。"""

    def __init__(self, in_dim: int = 5, hidden: int = 192, depth: int = 5):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


class BoundedBatchParameters(nn.Module):
    """批量参数优化变量。每条曲线一组 Kh/Kv/S/C。"""

    def __init__(self, init_theta: np.ndarray):
        super().__init__()
        raws = np.column_stack([
            self.to_raw_array(init_theta[:, 0], PARAM_BOUNDS["logKh"]),
            self.to_raw_array(init_theta[:, 1], PARAM_BOUNDS["logKv"]),
            self.to_raw_array(init_theta[:, 2], PARAM_BOUNDS["S"]),
            self.to_raw_array(init_theta[:, 3], PARAM_BOUNDS["logC"]),
        ])
        self.raw = nn.Parameter(torch.tensor(raws, dtype=torch.float32))

    @staticmethod
    def to_raw_array(values: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        low, high = bounds
        ratio = (values.astype(np.float64) - low) / max(high - low, 1e-12)
        ratio = np.clip(ratio, 1e-5, 1.0 - 1e-5)
        return np.log(ratio / (1.0 - ratio))

    @staticmethod
    def bounded(raw: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    def theta(self) -> torch.Tensor:
        return torch.stack([
            self.bounded(self.raw[:, 0], PARAM_BOUNDS["logKh"]),
            self.bounded(self.raw[:, 1], PARAM_BOUNDS["logKv"]),
            self.bounded(self.raw[:, 2], PARAM_BOUNDS["S"]),
            self.bounded(self.raw[:, 3], PARAM_BOUNDS["logC"]),
        ], dim=1)


def load_mat_dataset(mat_path: Path):
    """读取 MATLAB v7.3 mat 文件。"""
    with h5py.File(mat_path, "r") as f:
        X = np.asarray(f["X"], dtype=np.float64).T
        Y = np.asarray(f["Y"], dtype=np.float64).T
        D = np.asarray(f["D"], dtype=np.float64).T
        t_hours = np.asarray(f["t_hours"], dtype=np.float64).reshape(-1)
    return X, Y, D, t_hours


def load_model(model_path: Path, norm_path: Path, device: torch.device):
    """读取算子网络、归一化参数和数据划分索引。"""
    ckpt = torch.load(model_path, map_location=device)
    model = OperatorMLP(
        in_dim=int(ckpt.get("input_dim", 5)),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = np.load(norm_path)
    stats = {
        "x_mean": torch.tensor(norm["x_mean"], dtype=torch.float32, device=device),
        "x_std": torch.tensor(norm["x_std"], dtype=torch.float32, device=device),
        "y_mean": torch.tensor(float(norm["y_mean"]), dtype=torch.float32, device=device),
        "y_std": torch.tensor(float(norm["y_std"]), dtype=torch.float32, device=device),
        "train_idx": norm["train_idx"].astype(int),
        "val_idx": norm["val_idx"].astype(int),
        "test_idx": norm["test_idx"].astype(int),
    }
    return model, stats


def choose_indices(stats: Dict, split: str, n_samples: int) -> np.ndarray:
    """选择需要批量验证的曲线编号。"""
    if split == "train":
        idx = stats["train_idx"]
    elif split == "val":
        idx = stats["val_idx"]
    elif split == "test":
        idx = stats["test_idx"]
    elif split == "all":
        idx = np.concatenate([stats["train_idx"], stats["val_idx"], stats["test_idx"]])
    else:
        raise ValueError(f"未知 split: {split}")
    idx = np.asarray(idx, dtype=int)
    if n_samples > 0:
        idx = idx[:n_samples]
    return idx


def load_initial_guesses(csv_path: Path, X: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """优先使用已有 4 参数代理反演结果作为初值。"""
    by_idx = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sample_idx = int(float(row["sample_idx"]))
                by_idx[sample_idx] = np.array([
                    math.log10(float(row["est_Kh_mD"])),
                    math.log10(float(row["est_Kv_mD"])),
                    float(row["est_S"]),
                    math.log10(float(row["est_C_m3_per_MPa"])),
                ], dtype=np.float64)

    init = np.zeros((indices.size, 4), dtype=np.float64)
    for j, sample_idx in enumerate(indices):
        if int(sample_idx) in by_idx:
            init[j] = by_idx[int(sample_idx)]
        else:
            row = X[int(sample_idx)]
            init[j] = [
                math.log10(float(row[0])),
                math.log10(float(row[1])),
                float(row[2]),
                math.log10(float(row[3])),
            ]
    return init


def predict_log_pressure_batch(model, stats, theta: torch.Tensor, logt: torch.Tensor) -> torch.Tensor:
    """批量预测 log10 压力，返回 [n_curve, n_time]。"""
    n_curve = theta.shape[0]
    n_time = logt.numel()
    theta_expand = theta[:, None, :].repeat(1, n_time, 1)
    time_expand = logt[None, :, None].repeat(n_curve, 1, 1)
    x = torch.cat([theta_expand, time_expand], dim=2).reshape(-1, 5)
    x_std = (x - stats["x_mean"]) / stats["x_std"]
    y_std = model(x_std)
    y = y_std * stats["y_std"] + stats["y_mean"]
    return y.reshape(n_curve, n_time)


def import_core_module():
    """导入 Python 版物理核。"""
    spec = importlib.util.spec_from_file_location("core_model", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def theta_to_physical(theta: np.ndarray) -> np.ndarray:
    """theta -> [Kh, Kv, S, C]。"""
    out = np.empty_like(theta, dtype=np.float64)
    out[:, 0] = 10.0 ** theta[:, 0]
    out[:, 1] = 10.0 ** theta[:, 1]
    out[:, 2] = theta[:, 2]
    out[:, 3] = 10.0 ** theta[:, 3]
    return out


def compute_parameter_errors(true_params: np.ndarray, pred_params: np.ndarray) -> Dict[str, np.ndarray]:
    """计算 4 个参数误差。"""
    return {
        "Kh_rel_error": np.abs(pred_params[:, 0] - true_params[:, 0]) / np.maximum(np.abs(true_params[:, 0]), 1e-12),
        "Kv_rel_error": np.abs(pred_params[:, 1] - true_params[:, 1]) / np.maximum(np.abs(true_params[:, 1]), 1e-12),
        "S_abs_error": np.abs(pred_params[:, 2] - true_params[:, 2]),
        "C_log10_error": np.abs(np.log10(pred_params[:, 3]) - np.log10(true_params[:, 3])),
    }


def run_physics_check(indices, theta_final, X, Y, D, t_hours, args):
    """逐条代回物理核复查压力和导数。"""
    if args.phys_check == "none":
        return None

    module = import_core_module()
    core_args = SimpleNamespace(
        stehfest_n=10,
        np_probe_target=5,
        series_min_terms=10,
        series_max_terms=80,
        series_tol=1e-6,
        series_small_count=3,
        negative_skin_max_factor=20.0,
        derivative_window=0.1,
    )
    pressure_mean = np.full(indices.size, np.nan)
    pressure_p95 = np.full(indices.size, np.nan)
    derivative_median_log = np.full(indices.size, np.nan)
    derivative_p95_log = np.full(indices.size, np.nan)

    for j, sample_idx in enumerate(indices):
        row = X[int(sample_idx)]
        known = {
            "phi": 0.2,
            "ct_1_per_MPa": 5e-4,
            "h_m": 20.0,
            "q_cm3_per_s": float(row[4]) * 1e6 / 86400.0,
            "mu_cP": float(row[5]),
            "As_in2": float(row[6]),
            "hw_m": float(row[7]),
        }
        p_obs = Y[int(sample_idx)]
        d_obs = D[int(sample_idx)]
        valid = np.isfinite(p_obs) & (p_obs > 0)
        tt = t_hours[valid]
        p_obs = p_obs[valid]
        d_obs = d_obs[valid]
        try:
            p_core = module.v14_forward_pressure(theta_final[j], tt, known, core_args)
            d_core = module.bourdet_smooth(tt, p_core, core_args.derivative_window)
            p_rel = np.abs(p_core - p_obs) / np.maximum(np.abs(p_obs), 1e-12)
            d_mask = np.isfinite(d_core) & np.isfinite(d_obs) & (d_core > 0) & (d_obs > 0)
            d_log = np.full_like(d_obs, np.nan, dtype=np.float64)
            d_log[d_mask] = np.abs(np.log10(d_core[d_mask]) - np.log10(d_obs[d_mask]))
            pressure_mean[j] = np.nanmean(p_rel)
            pressure_p95[j] = np.nanpercentile(p_rel, 95)
            derivative_median_log[j] = np.nanmedian(d_log)
            derivative_p95_log[j] = np.nanpercentile(d_log, 95)
        except Exception as exc:
            print(f"physical check failed, sample_idx={sample_idx}: {exc}")

        if (j + 1) % args.recheck_print_every == 0 or (j + 1) == indices.size:
            print(f"physical check {j + 1}/{indices.size}")

    return {
        "phys_pressure_mean_relerr": pressure_mean,
        "phys_pressure_p95_relerr": pressure_p95,
        "phys_derivative_median_log10err": derivative_median_log,
        "phys_derivative_p95_log10err": derivative_p95_log,
    }


def summarize(values: np.ndarray) -> Dict[str, float]:
    """统计 mean/median/p90/p95/max。"""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "p90": float(np.nanpercentile(values, 90)),
        "p95": float(np.nanpercentile(values, 95)),
        "max": float(np.nanmax(values)),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, stats = load_model(args.model, args.norm, device)
    X, Y, D, t_hours = load_mat_dataset(args.mat)
    indices = choose_indices(stats, args.split, args.n_samples)
    print(f"split={args.split}, n_curve={indices.size}")

    init_theta = load_initial_guesses(args.init_csv, X, indices)
    inv = BoundedBatchParameters(init_theta).to(device)
    optimizer = torch.optim.AdamW(inv.parameters(), lr=args.lr)

    logt = torch.tensor(np.log10(t_hours), dtype=torch.float32, device=device)
    y_obs_np = np.log10(np.where((Y[indices] > 0) & np.isfinite(Y[indices]), Y[indices], np.nan))
    y_obs = torch.tensor(y_obs_np, dtype=torch.float32, device=device)
    mask = torch.isfinite(y_obs)
    init_t = torch.tensor(init_theta, dtype=torch.float32, device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        theta = inv.theta()
        pred = predict_log_pressure_batch(model, stats, theta, logt)
        data_loss = torch.mean((pred[mask] - y_obs[mask]) ** 2)
        prior_loss = torch.mean((theta - init_t) ** 2)
        loss = data_loss + args.w_prior * prior_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "data_loss": float(data_loss.detach().cpu()),
                "prior_loss": float(prior_loss.detach().cpu()),
            }
            history.append(row)
            print(
                f"epoch {epoch:04d} loss={row['loss']:.6e} "
                f"data={row['data_loss']:.6e} prior={row['prior_loss']:.6e}"
            )

    theta_final = inv.theta().detach().cpu().numpy()
    pred_params = theta_to_physical(theta_final)
    true_params = X[indices, :4].astype(np.float64)
    param_errors = compute_parameter_errors(true_params, pred_params)

    with torch.no_grad():
        pred_log = predict_log_pressure_batch(model, stats, inv.theta(), logt).detach().cpu().numpy()
    pred_pressure = 10.0 ** pred_log
    pressure_rel = np.abs(pred_pressure - Y[indices]) / np.maximum(np.abs(Y[indices]), 1e-12)
    pressure_rel[~np.isfinite(y_obs_np)] = np.nan
    operator_pressure_mean = np.nanmean(pressure_rel, axis=1)
    operator_pressure_p95 = np.nanpercentile(pressure_rel, 95, axis=1)

    recheck = run_physics_check(indices, theta_final, X, Y, D, t_hours, args)
    rows = build_rows(indices, true_params, pred_params, param_errors, operator_pressure_mean, operator_pressure_p95, recheck)
    save_tables(args, rows, history, param_errors, operator_pressure_mean, operator_pressure_p95, recheck)
    save_all_figures(args, rows, true_params, pred_params, param_errors, operator_pressure_mean, operator_pressure_p95, recheck)


def build_rows(indices, true_params, pred_params, param_errors, operator_pressure_mean, operator_pressure_p95, recheck):
    """构造 CSV 行。"""
    rows = []
    for j, sample_idx in enumerate(indices):
        row = {
            "sample_idx": int(sample_idx),
            "true_Kh_mD": true_params[j, 0],
            "inv_Kh_mD": pred_params[j, 0],
            "Kh_rel_error": param_errors["Kh_rel_error"][j],
            "true_Kv_mD": true_params[j, 1],
            "inv_Kv_mD": pred_params[j, 1],
            "Kv_rel_error": param_errors["Kv_rel_error"][j],
            "true_S": true_params[j, 2],
            "inv_S": pred_params[j, 2],
            "S_abs_error": param_errors["S_abs_error"][j],
            "true_C_m3_per_MPa": true_params[j, 3],
            "inv_C_m3_per_MPa": pred_params[j, 3],
            "C_log10_error": param_errors["C_log10_error"][j],
            "net_pressure_mean_relerr": operator_pressure_mean[j],
            "net_pressure_p95_relerr": operator_pressure_p95[j],
        }
        if recheck is not None:
            for key, arr in recheck.items():
                row[key] = arr[j]
        rows.append(row)
    return rows


def save_tables(args, rows, history, param_errors, operator_pressure_mean, operator_pressure_p95, recheck) -> None:
    """保存 CSV、history 和 summary JSON。"""
    csv_path = args.outdir / f"results_{args.split}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hist_path = args.outdir / f"history_{args.split}.csv"
    with hist_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "split": args.split,
        "n_curve": len(rows),
        "Kh_rel_error": summarize(param_errors["Kh_rel_error"]),
        "Kv_rel_error": summarize(param_errors["Kv_rel_error"]),
        "S_abs_error": summarize(param_errors["S_abs_error"]),
        "C_log10_error": summarize(param_errors["C_log10_error"]),
        "net_pressure_mean_relerr": summarize(operator_pressure_mean),
        "net_pressure_p95_relerr": summarize(operator_pressure_p95),
    }
    if recheck is not None:
        for key, arr in recheck.items():
            summary[key] = summarize(arr)

    summary_path = args.outdir / f"summary_{args.split}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    worst_path = args.outdir / f"worst_{args.split}.csv"
    sort_key = "phys_pressure_mean_relerr" if recheck is not None else "net_pressure_mean_relerr"
    worst = sorted(rows, key=lambda r: float(r.get(sort_key, np.nan)), reverse=True)[: min(30, len(rows))]
    with worst_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(worst)

    print("\nSaved tables:")
    print(f"  {csv_path}")
    print(f"  {hist_path}")
    print(f"  {summary_path}")
    print(f"  {worst_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def save_all_figures(args, rows, true_params, pred_params, param_errors, operator_pressure_mean, operator_pressure_p95, recheck) -> None:
    """保存全部批量统计图。"""
    save_parameter_crossplot(args.figdir / f"param_{args.split}.png", true_params, pred_params, param_errors)
    save_parameter_error_histogram(args.figdir / f"hist_{args.split}.png", param_errors)
    save_error_boxplot(args.figdir / f"box_{args.split}.png", param_errors, operator_pressure_mean, recheck)
    save_quality_histogram(args.figdir / f"quality_{args.split}.png", operator_pressure_mean, operator_pressure_p95, recheck)
    save_sample_error_scatter(args.figdir / f"sample_{args.split}.png", rows, recheck)

    print("\nSaved figures:")
    for path in sorted(args.figdir.glob(f"*_{args.split}.png")):
        print(f"  {path}")


def save_parameter_crossplot(path: Path, true_params, pred_params, param_errors) -> None:
    """真实参数 vs 反演参数交会图。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    specs = [
        (0, "Kh", "mD", True, param_errors["Kh_rel_error"], "median err = {:.2%}"),
        (1, "Kv", "mD", True, param_errors["Kv_rel_error"], "median err = {:.2%}"),
        (2, "S", "", False, param_errors["S_abs_error"], "median abs err = {:.3g}"),
        (3, "C", "m3/MPa", True, param_errors["C_log10_error"], "median log err = {:.3g}"),
    ]
    for ax, (col, name, unit, logscale, err, title_fmt) in zip(axes.ravel(), specs):
        ax.scatter(true_params[:, col], pred_params[:, col], s=20, alpha=0.75, edgecolor="k", linewidth=0.3)
        finite = np.isfinite(true_params[:, col]) & np.isfinite(pred_params[:, col])
        lo = min(np.nanmin(true_params[finite, col]), np.nanmin(pred_params[finite, col]))
        hi = max(np.nanmax(true_params[finite, col]), np.nanmax(pred_params[finite, col]))
        if logscale:
            lo = max(lo, 1e-12)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xlabel(f"True {name} [{unit}]" if unit else f"True {name}")
        ax.set_ylabel(f"Inverted {name} [{unit}]" if unit else f"Inverted {name}")
        ax.set_title(f"{name} | " + title_fmt.format(np.nanmedian(err)))
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("True vs Inverted Parameters")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_parameter_error_histogram(path: Path, param_errors) -> None:
    """参数误差直方图。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    items = [
        ("Kh relative error", param_errors["Kh_rel_error"]),
        ("Kv relative error", param_errors["Kv_rel_error"]),
        ("S absolute error", param_errors["S_abs_error"]),
        ("C log10 error", param_errors["C_log10_error"]),
    ]
    for ax, (title, values) in zip(axes.ravel(), items):
        ax.hist(values[np.isfinite(values)], bins=30, color="#4C78A8", edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_quality_histogram(path: Path, operator_pressure_mean, operator_pressure_p95, recheck) -> None:
    """曲线拟合质量分布。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    items = [
        ("Network pressure mean relerr", operator_pressure_mean),
        ("Network pressure p95 relerr", operator_pressure_p95),
    ]
    if recheck is not None:
        items += [
            ("Physical pressure mean relerr", recheck["phys_pressure_mean_relerr"]),
            ("Physical derivative median log10err", recheck["phys_derivative_median_log10err"]),
        ]
    else:
        items += [("Physical pressure mean relerr", np.array([])), ("Physical derivative median log10err", np.array([]))]
    for ax, (title, values) in zip(axes.ravel(), items):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=30, color="#59A14F", edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_error_boxplot(path: Path, param_errors, operator_pressure_mean, recheck) -> None:
    """参数误差和曲线误差箱线图。"""
    labels = ["Kh rel", "Kv rel", "S abs", "C log", "Op p mean"]
    data = [
        param_errors["Kh_rel_error"],
        param_errors["Kv_rel_error"],
        param_errors["S_abs_error"],
        param_errors["C_log10_error"],
        operator_pressure_mean,
    ]
    if recheck is not None:
        labels += ["Phys p mean", "Phys d log"]
        data += [recheck["phys_pressure_mean_relerr"], recheck["phys_derivative_median_log10err"]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    clean = [np.asarray(x)[np.isfinite(x)] for x in data]
    ax.boxplot(clean, tick_labels=labels, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_sample_error_scatter(path: Path, rows, recheck) -> None:
    """误差随样本编号散点图，便于定位异常样本。"""
    sample_idx = np.asarray([r["sample_idx"] for r in rows])
    op_err = np.asarray([r["net_pressure_mean_relerr"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(sample_idx, op_err, s=18, alpha=0.75, label="Network pressure mean relerr")
    if recheck is not None:
        phys_err = np.asarray([r["phys_pressure_mean_relerr"] for r in rows], dtype=float)
        ax.scatter(sample_idx, phys_err, s=18, alpha=0.75, label="Physical pressure mean relerr")
    ax.set_yscale("log")
    ax.set_xlabel("sample_idx")
    ax.set_ylabel("Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=PROJECT_DIR / "data_2000.mat")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "model" / "net.pt")
    parser.add_argument("--norm", type=Path, default=SCRIPT_DIR / "model" / "norm.npz")
    parser.add_argument("--init-csv", type=Path, default=SCRIPT_DIR / "init.csv")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "batch_out")
    parser.add_argument("--figdir", type=Path, default=SCRIPT_DIR / "batch_fig")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--n-samples", type=int, default=0, help="只跑前 N 条所选 split；0 表示全部")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--w-prior", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--phys-check", choices=["all", "none"], default="all")
    parser.add_argument("--recheck-print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
