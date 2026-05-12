"""
第 2 步：整理神经网络训练数据。

本脚本不重新计算 2000 条理论曲线，而是把 MATLAB 第 1 步生成的
data_2000.mat 整理成神经网络更容易读取的 npz 数据集。

数据含义：
    输入参数：log10(Kh), log10(Kv), S, log10(C)
    时间输入：log10(t_hour)
    网络目标：log10(Delta_p)

注意：
    q、mu、As、hw 在 4 参数数据集中应为固定已知参数。
    如果它们不是常数，4 参数 PINN/算子模型就不唯一。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

INVERTED_PARAM_NAMES = ["Kh_mD", "Kv_mD", "S", "C_m3_per_MPa"]
KNOWN_PARAM_NAMES = ["q_m3_per_d", "mu_cP", "As_in2", "hw_m"]


def load_mat_dataset(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取 MATLAB v7.3 mat 文件，转成样本在前的维度。"""
    with h5py.File(mat_path, "r") as f:
        X = np.asarray(f["X"], dtype=np.float64).T
        Y = np.asarray(f["Y"], dtype=np.float64).T
        D = np.asarray(f["D"], dtype=np.float64).T
        t_hours = np.asarray(f["t_hours"], dtype=np.float64).reshape(-1)
    return X, Y, D, t_hours


def transform_parameters(X: np.ndarray) -> np.ndarray:
    """把 4 个反演参数变成适合神经网络学习的形式。"""
    X4 = np.empty((X.shape[0], 4), dtype=np.float64)
    X4[:, 0] = np.log10(X[:, 0])
    X4[:, 1] = np.log10(X[:, 1])
    X4[:, 2] = X[:, 2]
    X4[:, 3] = np.log10(X[:, 3])
    return X4


def check_known_parameters_fixed(X: np.ndarray) -> Dict[str, float]:
    """检查已知参数是否真的固定。"""
    spans = {}
    for i, name in enumerate(KNOWN_PARAM_NAMES, start=4):
        col = X[:, i]
        spans[name] = float(np.nanmax(col) - np.nanmin(col))
    return spans


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    X, Y, D, t_hours = load_mat_dataset(args.mat)
    if args.max_curves > 0:
        X = X[: args.max_curves]
        Y = Y[: args.max_curves]
        D = D[: args.max_curves]

    X4 = transform_parameters(X)
    logt = np.log10(t_hours)
    pressure_mask = np.isfinite(Y) & (Y > 0)
    log_pressure = np.full_like(Y, np.nan, dtype=np.float64)
    log_pressure[pressure_mask] = np.log10(Y[pressure_mask])
    log_derivative = np.full_like(D, np.nan, dtype=np.float64)
    mask_d = np.isfinite(D) & (D > 0)
    log_derivative[mask_d] = np.log10(D[mask_d])

    known_spans = check_known_parameters_fixed(X)
    varying = {k: v for k, v in known_spans.items() if v > 1e-10}
    if varying:
        raise ValueError(f"已知参数不是常数，不能训练 4 参数算子：{varying}")

    npz_path = args.outdir / args.output_name
    np.savez_compressed(
        npz_path,
        X4=X4,
        X_raw=X,
        logt=logt,
        t_hours=t_hours,
        log_pressure=log_pressure,
        log_derivative=log_derivative,
        pressure_mask=pressure_mask,
        pressure=Y,
        derivative=D,
        param_names=np.asarray(INVERTED_PARAM_NAMES),
        known_param_names=np.asarray(KNOWN_PARAM_NAMES),
    )

    meta = {
        "source_mat": str(args.mat),
        "output_npz": str(npz_path),
        "n_curve": int(X.shape[0]),
        "n_time": int(t_hours.size),
        "n_valid_pressure_points": int(np.sum(pressure_mask)),
        "n_invalid_pressure_points": int(np.size(pressure_mask) - np.sum(pressure_mask)),
        "n_curves_with_invalid_pressure": int(np.sum(np.any(~pressure_mask, axis=1))),
        "param_names": INVERTED_PARAM_NAMES,
        "known_param_names": KNOWN_PARAM_NAMES,
        "known_parameter_values": {
            name: float(X[0, i]) for i, name in enumerate(KNOWN_PARAM_NAMES, start=4)
        },
        "known_parameter_spans": known_spans,
        "target": "log10(Delta_p)",
    }
    meta_path = args.outdir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved:")
    print(f"  {npz_path}")
    print(f"  {meta_path}")
    print(f"curves={X.shape[0]}, time_points={t_hours.size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=PROJECT_DIR / "data_2000.mat")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--output-name", default="train_data.npz")
    parser.add_argument("--max-curves", type=int, default=0, help="调试时只导出前 N 条曲线，0 表示全部")
    return parser.parse_args()


if __name__ == "__main__":
    main()
