#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from minimal_pipeline import GridAerosolDataset, ConvLSTMForecast


def _make_loaders(data_root_path, seq_len, target_delay, include_t2m_input, batch_size):
    ds = GridAerosolDataset(
        data_root_path,
        sequence_length=seq_len,
        target_delay=target_delay,
        include_t2m_input=include_t2m_input,
        predict_delta=False,
    )
    n = len(ds)
    tr = int(n * 0.8)
    va = max(1, int(n * 0.1))
    idx = list(range(n))

    from torch.utils.data import DataLoader, Subset

    train_loader = DataLoader(Subset(ds, idx[:tr]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, idx[tr:tr + va]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, idx[tr + va:]), batch_size=batch_size, shuffle=False)
    return ds, train_loader, val_loader, test_loader


# ✅ Manual SSIM (global)
def _eval_ssim(model, loader, device):
    model.eval()

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_sum = 0.0
    n_items = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred, _ = model(xb)

            # Optional but recommended (stability)
            pred = torch.clamp(pred, 0, 1)
            yb = torch.clamp(yb, 0, 1)

            # Flatten spatial dimensions
            pred_flat = pred.view(pred.size(0), -1)
            yb_flat = yb.view(yb.size(0), -1)

            # Mean
            mu_x = pred_flat.mean(dim=1)
            mu_y = yb_flat.mean(dim=1)

            # Variance
            sigma_x = pred_flat.var(dim=1, unbiased=False)
            sigma_y = yb_flat.var(dim=1, unbiased=False)

            # Covariance
            sigma_xy = ((pred_flat - mu_x.unsqueeze(1)) *
                        (yb_flat - mu_y.unsqueeze(1))).mean(dim=1)

            # SSIM formula
            numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

            ssim_batch = numerator / denominator

            ssim_sum += ssim_batch.sum().item()
            n_items += xb.size(0)

    return ssim_sum / max(1, n_items)


def train_and_log_ssim(
    data_root_path,
    horizon_label,
    target_delay,
    out_dir,
    seq_len=6,
    include_t2m_input=True,
    hidden_dim=64,
    batch_size=8,
    lr=5e-4,
    num_epochs=30,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds, train_loader, val_loader, test_loader = _make_loaders(
        data_root_path=data_root_path,
        seq_len=seq_len,
        target_delay=target_delay,
        include_t2m_input=include_t2m_input,
        batch_size=batch_size,
    )

    model = ConvLSTMForecast(in_ch=ds.in_ch, hidden_dim=hidden_dim, out_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.HuberLoss(delta=1.0)

    rows = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        test_ssim = _eval_ssim(model, test_loader, device)

        rows.append({'horizon': horizon_label, 'epoch': epoch, 'ssim': test_ssim})
        print(f"h={horizon_label:>2}  delay={target_delay}  epoch={epoch:>3}  test_ssim={test_ssim:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"ssim_vs_epoch_h{horizon_label}.csv", index=False)
    return df


def plot_combined(dfs, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for df in dfs:
        h = int(df['horizon'].iloc[0])
        ax.plot(df['epoch'], df['ssim'], linewidth=2.2, marker='o', markersize=4.5, label=str(h))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('SSIM vs Epoch (Different Horizons)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(title='Horizon', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--data_root_path', required=True)
    p.add_argument('--out_dir', default='output_plots/ssim_vs_epoch_horizons_6_12_18_24')
    p.add_argument('--num_epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seq_len', type=int, default=6)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--include_t2m_input', action='store_true')
    args = p.parse_args()

    horizons = [6, 12, 18, 24]
    delays = [1, 2, 3, 4]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for h, d in zip(horizons, delays):
        df = train_and_log_ssim(
            data_root_path=args.data_root_path,
            horizon_label=h,
            target_delay=d,
            out_dir=out_dir,
            seq_len=args.seq_len,
            include_t2m_input=args.include_t2m_input,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
        )
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_dir / 'ssim_vs_epoch_horizons_6_12_18_24.csv', index=False)

    plot_combined(dfs, out_dir / 'ssim_vs_epoch_horizons_6_12_18_24.png')
    print(str((out_dir / 'ssim_vs_epoch_horizons_6_12_18_24.png').resolve()))


if __name__ == '__main__':
    main()