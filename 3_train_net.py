"""
第 3 步：训练物理算子网络。

这里的神经网络采用 MATLAB 理论模型生成的数据作为物理教师：

    输入:  log10(Kh), log10(Kv), S, log10(C), log10(t)
    输出:  log10(Delta_p)

它不是旧版简化 PDE PINN。旧版的问题是 PDE、表皮、井储和有限面积探针
与 MATLAB 拉氏空间模型不完全一致。当前脚本的物理信息来自理论数据，
后续所有反演结果仍应代回 MATLAB 正演检查。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent


class OperatorMLP(nn.Module):
    """用 4 个参数和时间预测压力响应的全连接网络。"""

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


def load_teacher_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取教师数据。"""
    data = np.load(path, allow_pickle=True)
    return data["X4"], data["logt"], data["log_pressure"], data["t_hours"]


def split_curve_indices(n_curve: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按曲线编号划分训练、验证、测试，避免同一条曲线泄漏到多个集合。"""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_curve)
    rng.shuffle(indices)
    n_train = int(0.70 * n_curve)
    n_val = int(0.15 * n_curve)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_point_dataset(X4: np.ndarray, logt: np.ndarray, y: np.ndarray, curve_idx: np.ndarray):
    """把曲线数据展开为逐时间点训练数据。"""
    n_time = logt.size
    x_curve = X4[curve_idx]
    x_param = np.repeat(x_curve, n_time, axis=0)
    x_time = np.tile(logt, curve_idx.size).reshape(-1, 1)
    x = np.hstack([x_param, x_time]).astype(np.float32)
    target = y[curve_idx].reshape(-1).astype(np.float32)
    valid = np.isfinite(target)
    return x[valid], target[valid]


def standardize_train_val_test(x_train, y_train, x_val, y_val, x_test, y_test):
    """用训练集统计量做标准化。"""
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() if y_train.std() > 1e-12 else 1.0)

    def sx(x):
        return ((x - x_mean) / x_std).astype(np.float32)

    def sy(y):
        return ((y - y_mean) / y_std).astype(np.float32)

    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return sx(x_train), sy(y_train), sx(x_val), sy(y_val), sx(x_test), sy(y_test), stats


def evaluate(model, loader, device) -> float:
    """计算标准化 MSE。"""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total += torch.sum((pred - yb) ** 2).item()
            count += yb.numel()
    return total / max(count, 1)


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

    X4, logt, y, t_hours = load_teacher_npz(args.teacher)
    train_idx, val_idx, test_idx = split_curve_indices(X4.shape[0], args.seed)
    x_train, y_train = make_point_dataset(X4, logt, y, train_idx)
    x_val, y_val = make_point_dataset(X4, logt, y, val_idx)
    x_test, y_test = make_point_dataset(X4, logt, y, test_idx)
    x_train, y_train, x_val, y_val, x_test, y_test, stats = standardize_train_val_test(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=args.batch_size)

    model = OperatorMLP(hidden=args.hidden, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val = math.inf
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            running += loss.item() * yb.numel()
            seen += yb.numel()
        scheduler.step()

        train_mse = running / max(seen, 1)
        val_mse = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} train={train_mse:.6e} val={val_mse:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse_std = evaluate(model, test_loader, device)
    test_rmse_log10 = math.sqrt(test_mse_std) * stats["y_std"]

    model_path = args.outdir / "net.pt"
    norm_path = args.outdir / "norm.npz"
    metrics_path = args.outdir / "metrics.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden": args.hidden,
            "depth": args.depth,
            "input_dim": 5,
        },
        model_path,
    )
    np.savez(
        norm_path,
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=np.asarray(stats["y_mean"]),
        y_std=np.asarray(stats["y_std"]),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        t_hours=t_hours,
    )
    metrics = {
        "n_curve": int(X4.shape[0]),
        "n_train_curve": int(train_idx.size),
        "n_val_curve": int(val_idx.size),
        "n_test_curve": int(test_idx.size),
        "best_val_mse_standardized": float(best_val),
        "test_mse_standardized": float(test_mse_std),
        "test_rmse_log10_pressure": float(test_rmse_log10),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_loss_figure(args.figdir / "train_loss.png", history)

    print("\nSaved:")
    print(f"  {model_path}")
    print(f"  {norm_path}")
    print(f"  {metrics_path}")
    print(f"  {args.figdir / 'train_loss.png'}")
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def save_loss_figure(path: Path, history) -> None:
    """保存训练损失图。"""
    epochs = [r["epoch"] for r in history]
    train = [r["train_mse"] for r in history]
    val = [r["val_mse"] for r in history]
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(epochs, train, label="train")
    plt.semilogy(epochs, val, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Standardized MSE")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, default=SCRIPT_DIR / "data" / "train_data.npz")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "model")
    parser.add_argument("--figdir", type=Path, default=SCRIPT_DIR / "fig")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
