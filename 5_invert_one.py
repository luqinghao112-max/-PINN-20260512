"""
第 5 步：用训练好的网络反演单条曲线。

流程：
    1. 读取训练好的 net.pt
    2. 读取理论观测曲线 data_2000.mat
    3. 固定神经网络权重，只优化 4 个参数
    4. 输出网络反演结果
    5. 可选：用 Python 物理核复算最终参数，检查网络反演是否物理一致
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from types import SimpleNamespace
from pathlib import Path
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
    """必须和 3_train_net.py 中的网络结构一致。"""

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


class BoundedParameters(nn.Module):
    """把无界优化变量映射到参数物理范围内。"""

    def __init__(self, init_theta: np.ndarray):
        super().__init__()
        self.raw_logKh = nn.Parameter(self.to_raw(init_theta[0], PARAM_BOUNDS["logKh"]))
        self.raw_logKv = nn.Parameter(self.to_raw(init_theta[1], PARAM_BOUNDS["logKv"]))
        self.raw_S = nn.Parameter(self.to_raw(init_theta[2], PARAM_BOUNDS["S"]))
        self.raw_logC = nn.Parameter(self.to_raw(init_theta[3], PARAM_BOUNDS["logC"]))

    @staticmethod
    def to_raw(value: float, bounds: Tuple[float, float]) -> torch.Tensor:
        low, high = bounds
        ratio = (float(value) - low) / max(high - low, 1e-12)
        ratio = min(max(ratio, 1e-5), 1.0 - 1e-5)
        return torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)

    @staticmethod
    def bounded(raw: torch.Tensor, bounds: Tuple[float, float]) -> torch.Tensor:
        low, high = bounds
        return low + (high - low) * torch.sigmoid(raw)

    def theta(self) -> torch.Tensor:
        return torch.stack([
            self.bounded(self.raw_logKh, PARAM_BOUNDS["logKh"]),
            self.bounded(self.raw_logKv, PARAM_BOUNDS["logKv"]),
            self.bounded(self.raw_S, PARAM_BOUNDS["S"]),
            self.bounded(self.raw_logC, PARAM_BOUNDS["logC"]),
        ])


def load_mat_dataset(mat_path: Path):
    """读取 MATLAB v7.3 mat 文件。"""
    with h5py.File(mat_path, "r") as f:
        X = np.asarray(f["X"], dtype=np.float64).T
        Y = np.asarray(f["Y"], dtype=np.float64).T
        D = np.asarray(f["D"], dtype=np.float64).T
        t_hours = np.asarray(f["t_hours"], dtype=np.float64).reshape(-1)
    return X, Y, D, t_hours


def load_model(model_path: Path, norm_path: Path, device: torch.device):
    """读取训练好的算子网络和归一化参数。"""
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
    }
    return model, stats


def load_initial_guess(csv_path: Path, sample_idx: int, x_true: np.ndarray) -> np.ndarray:
    """优先使用 4 参数代理模型反演结果作为初值。"""
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


def predict_log_pressure(model, stats, theta: torch.Tensor, logt: torch.Tensor) -> torch.Tensor:
    """预测 log10 压力。"""
    n = logt.numel()
    theta_expand = theta.reshape(1, 4).repeat(n, 1)
    x = torch.cat([theta_expand, logt.reshape(-1, 1)], dim=1)
    x_std = (x - stats["x_mean"]) / stats["x_std"]
    y_std = model(x_std)
    return y_std * stats["y_std"] + stats["y_mean"]


def theta_to_params(theta: np.ndarray) -> Dict[str, float]:
    """优化变量转物理参数。"""
    return {
        "Kh_mD": float(10.0 ** theta[0]),
        "Kv_mD": float(10.0 ** theta[1]),
        "S": float(theta[2]),
        "C_m3_per_MPa": float(10.0 ** theta[3]),
    }


def import_core_module():
    """动态导入物理核，用于最终复查。"""
    spec = importlib.util.spec_from_file_location("core_model", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.figdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model, stats = load_model(args.model, args.norm, device)
    X, Y, D, t_hours = load_mat_dataset(args.mat)
    x_true = X[args.sample_idx]
    p_obs = Y[args.sample_idx]
    d_obs = D[args.sample_idx]
    valid = np.isfinite(p_obs) & (p_obs > 0)
    t_hours = t_hours[valid]
    p_obs = p_obs[valid]
    d_obs = d_obs[valid]

    init = load_initial_guess(args.init_csv, args.sample_idx, x_true)
    inv_params = BoundedParameters(init).to(device)
    optimizer = torch.optim.AdamW(inv_params.parameters(), lr=args.lr, weight_decay=0.0)

    logt = torch.tensor(np.log10(t_hours), dtype=torch.float32, device=device)
    y_obs = torch.tensor(np.log10(p_obs), dtype=torch.float32, device=device)
    init_t = torch.tensor(init, dtype=torch.float32, device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        theta = inv_params.theta()
        pred = predict_log_pressure(model, stats, theta, logt)
        data_loss = torch.mean((pred - y_obs) ** 2)
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
            theta_np = theta.detach().cpu().numpy()
            row.update(theta_to_params(theta_np))
            history.append(row)
            print(
                f"epoch {epoch:04d} loss={row['loss']:.4e} data={row['data_loss']:.4e} "
                f"Kh={row['Kh_mD']:.4g} Kv={row['Kv_mD']:.4g} S={row['S']:.4g} C={row['C_m3_per_MPa']:.3e}"
            )

    theta_final = inv_params.theta().detach().cpu().numpy()
    params_est = theta_to_params(theta_final)
    with torch.no_grad():
        y_fit_log = predict_log_pressure(model, stats, inv_params.theta(), logt).detach().cpu().numpy()
    p_fit = 10.0 ** y_fit_log
    rel = np.abs(p_fit - p_obs) / np.maximum(np.abs(p_obs), 1e-12)

    phys_check = None
    if args.phys_check:
        phys_check = run_physics_check(theta_final, t_hours, x_true, args, p_obs, d_obs)

    result = {
        "sample_idx": int(args.sample_idx),
        "true": {
            "Kh_mD": float(x_true[0]),
            "Kv_mD": float(x_true[1]),
            "S": float(x_true[2]),
            "C_m3_per_MPa": float(x_true[3]),
        },
        "estimated": params_est,
        "net_pressure_mean_relerr": float(np.nanmean(rel)),
        "net_pressure_p95_relerr": float(np.nanpercentile(rel, 95)),
        "phys_check": phys_check,
    }
    save_outputs(args, result, history, t_hours, p_obs, p_fit, d_obs, phys_check)


def run_physics_check(
    theta_final: np.ndarray,
    t_hours: np.ndarray,
    x_true: np.ndarray,
    args: argparse.Namespace,
    p_obs: np.ndarray,
    d_obs: np.ndarray,
):
    """用 Python 物理核复算最终参数。"""
    module = import_core_module()
    known = {
        "phi": 0.2,
        "ct_1_per_MPa": 5e-4,
        "h_m": 20.0,
        "q_cm3_per_s": float(x_true[4]) * 1e6 / 86400.0,
        "mu_cP": float(x_true[5]),
        "As_in2": float(x_true[6]),
        "hw_m": float(x_true[7]),
    }
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
    p_core = module.v14_forward_pressure(theta_final, t_hours, known, core_args)
    d_core = module.bourdet_smooth(t_hours, p_core, core_args.derivative_window)
    p_rel = np.abs(p_core - p_obs) / np.maximum(np.abs(p_obs), 1e-12)
    d_mask = np.isfinite(d_core) & np.isfinite(d_obs) & (d_core > 0) & (d_obs > 0)
    d_logerr = np.full_like(d_obs, np.nan, dtype=np.float64)
    d_logerr[d_mask] = np.abs(np.log10(d_core[d_mask]) - np.log10(d_obs[d_mask]))
    return {
        "pressure_mean_relerr": float(np.nanmean(p_rel)),
        "pressure_p95_relerr": float(np.nanpercentile(p_rel, 95)),
        "derivative_median_log10err": float(np.nanmedian(d_logerr)),
        "pressure": p_core.tolist(),
        "derivative": d_core.tolist(),
    }


def save_outputs(args, result, history, t_hours, p_obs, p_fit, d_obs, phys_check) -> None:
    """保存结果和图件。"""
    result_path = args.outdir / f"fit_{args.sample_idx}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    hist_path = args.outdir / f"hist_{args.sample_idx}.csv"
    with hist_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    fig_path = args.figdir / f"fit_{args.sample_idx}.png"
    plt.figure(figsize=(8.5, 5.8))
    plt.loglog(t_hours, np.maximum(p_obs, 1e-12), "k-", lw=1.8, label="Observed pressure")
    plt.loglog(t_hours, np.maximum(p_fit, 1e-12), "r--", lw=1.6, label="Network pressure")
    plt.loglog(t_hours, np.maximum(d_obs, 1e-12), color="0.35", ls="-.", lw=1.4, label="Observed derivative")
    if phys_check is not None:
        plt.loglog(t_hours, np.maximum(np.asarray(phys_check["pressure"]), 1e-12), "b:", lw=1.8, label="Physical check pressure")
        plt.loglog(t_hours, np.maximum(np.asarray(phys_check["derivative"]), 1e-12), color="m", ls=":", lw=1.6, label="Physical check derivative")
    plt.xlabel("Time t [h]")
    plt.ylabel("Pressure drop and derivative [MPa]")
    plt.title(f"Network inversion sample {args.sample_idx}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    print("\nSaved:")
    print(f"  {result_path}")
    print(f"  {hist_path}")
    print(f"  {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=PROJECT_DIR / "data_2000.mat")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "model" / "net.pt")
    parser.add_argument("--norm", type=Path, default=SCRIPT_DIR / "model" / "norm.npz")
    parser.add_argument("--init-csv", type=Path, default=SCRIPT_DIR / "init.csv")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "out")
    parser.add_argument("--figdir", type=Path, default=SCRIPT_DIR / "out_fig")
    parser.add_argument("--sample-idx", type=int, default=998)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--w-prior", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--phys-check", action="store_true", help="反演结束后用物理核复算检查")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
