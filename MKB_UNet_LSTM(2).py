import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
import time
import pandas as pd
from typing import Dict, List, Tuple
import warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import geopandas as gpd
from shapely.geometry import box
from shapely.prepared import prep
from shapely.geometry import Point
import argparse
import random

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

warnings.filterwarnings('ignore')


# 1. Dataset Definition

class GridAerosolDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, target_delay=1, train_split_idx=418, include_t2m_input=False, predict_delta=False):
        self.sequence_length = sequence_length
        self.target_delay = target_delay
        self.include_t2m_input = include_t2m_input
        self.predict_delta = predict_delta
        self.vars = ['BC_AOD', 'SU_AOD', 'DU_AOD_pm25']

        root = Path(root_dir)

        # Load Data
        data = {}
        for v in self.vars + ['T2m']:
            fpath = root / f'{v}_time_series.csv'
            if not fpath.exists():
                raise FileNotFoundError(f"Missing file: {fpath}")
            arr = pd.read_csv(fpath, header=0).values
            # Expecting 361 pixels (19x19) by time. If transposed, fix.
            if arr.shape[0] != 361 and arr.shape[1] == 361:
                arr = arr.T
            if arr.shape[0] != 361:
                raise ValueError(
                    f"Unexpected shape for {v}: {arr.shape}. Expected (361, Time) or (Time, 361)."
                )
            data[v] = arr

        print("Generating Sine/Cosine Month Channels & Processing Frames...")
        all_X_frames = []
        all_Y_frames = []

        # Assuming data columns are time steps.
        # CAUTION: Ensure your CSV shape is (361, Time) or (Time, 361).
        # The code assumes (Pixels, Time) based on previous context.

        total_time_steps = data['T2m'].shape[1]

        for t in range(2, total_time_steps):
            # 1. Input Features
            x_stack = [data[v][:, t].reshape(19, 19)[::-1, :] for v in self.vars]

            if self.include_t2m_input:
                # Autoregressive signal: use lagged temperature (t-1)
                x_stack.append(data['T2m'][:, t - 1].reshape(19, 19)[::-1, :])

            # Time Encoding
            month_idx = t % 12
            month_angle = 2 * np.pi * (month_idx / 12.0)
            x_stack.append(np.full((19, 19), np.sin(month_angle), dtype=np.float32))
            x_stack.append(np.full((19, 19), np.cos(month_angle), dtype=np.float32))

            all_X_frames.append(np.stack(x_stack, axis=0))

            # 2. Target Variable
            temp = data['T2m'][:, t].reshape(19, 19)[::-1, :].reshape(1, 19, 19)
            all_Y_frames.append(temp)

        all_X_frames = np.stack(all_X_frames).astype(np.float32)
        all_Y_frames = np.stack(all_Y_frames).astype(np.float32)

        self.in_ch = all_X_frames.shape[1]

        # --- Normalization (Prevents Data Leakage) ---
        # Only fit statistics on the TRAINING portion of the data
        print("Normalizing AOD channels (0-3) using Training Split stats...")
        train_stats_data = all_X_frames[:train_split_idx, 0:3, :, :]
        self.mu = train_stats_data.mean(axis=(0, 2, 3), keepdims=True)
        self.sd = train_stats_data.std(axis=(0, 2, 3), keepdims=True) + 1e-6

        # Apply to ALL data
        all_X_frames[:, 0:3, :, :] = (all_X_frames[:, 0:3, :, :] - self.mu) / self.sd

        # --- Target Normalization (Train-only stats; helps optimization) ---
        print("Normalizing target channel (T2m) using Training Split stats...")
        train_y = all_Y_frames[:train_split_idx]
        self.mu_y = train_y.mean(axis=(0, 2, 3), keepdims=True)
        self.sd_y = train_y.std(axis=(0, 2, 3), keepdims=True) + 1e-6

        # If predicting deltas, convert target to delta in raw space first.
        if self.predict_delta:
            # delta_t = T[t] - T[t-1] (same indexing as frame generation)
            delta = all_Y_frames.copy()
            delta[1:] = all_Y_frames[1:] - all_Y_frames[:-1]
            delta[0] = 0.0
            all_Y_frames = delta

        all_Y_frames = (all_Y_frames - self.mu_y) / self.sd_y

        if self.include_t2m_input:
            t2m_ch = 3
            all_X_frames[:, t2m_ch:t2m_ch+1, :, :] = (all_X_frames[:, t2m_ch:t2m_ch+1, :, :] - self.mu_y) / self.sd_y

        # --- Sequence Generation (FIXED) ---
        self.X_data = []
        self.Y_data = []
        num_total_frames = all_X_frames.shape[0]

        # Calculate valid end index
        # We need (i + target_delay) < num_total_frames
        max_idx = num_total_frames - self.target_delay

        for i in range(self.sequence_length, max_idx):
            # Input window: [i - seq_len : i]
            x_seq = all_X_frames[i - self.sequence_length : i]

            # Target window: [i - seq_len + delay : i + delay]
            y_seq = all_Y_frames[i - self.sequence_length + self.target_delay : i + self.target_delay]

            # Consistency Check
            if x_seq.shape[0] == self.sequence_length and y_seq.shape[0] == self.sequence_length:
                self.X_data.append(x_seq)
                self.Y_data.append(y_seq)

        self.X_data = np.stack(self.X_data).astype(np.float32)
        self.Y_data = np.stack(self.Y_data).astype(np.float32)

        print(f"Dataset created. Input Shape: {self.X_data.shape}")
        print(f"Target Shape: {self.Y_data.shape}")

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X_data[idx]), torch.from_numpy(self.Y_data[idx])



# 2. Data Loaders


def setup_data_loaders(data_root_path, batch_size=8, sequence_length=5):
    """Setup data loaders"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Split sizes are based on samples in the sequence dataset.
    # We pass a train_split_idx (frames) only to compute normalization stats.
    full_dataset = GridAerosolDataset(data_root_path, sequence_length=sequence_length)
    total_size = len(full_dataset)

    # train_count = int(total_size * 0.8)
    # val_count = int(total_size * 0.1)

    # Dynamic splits (time-series aware: contiguous blocks).
    train_count = int(total_size * 0.8)
    val_count = max(1, int(total_size * 0.1))

    indices = list(range(total_size))
    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[train_count + val_count:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


    # Shuffling helps generalization; keep val/test ordered.
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, device



# 3. Physics-Informed Loss Function (Novelity)


class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_huber=1.0, lambda_temp=0.5, lambda_grad=0.1, lambda_ssim=0.0, ssim_window_size=7, time_weight_gamma=0.0, use_mse_magnitude=False, huber_delta=2.0, ssim_all_timesteps=False):
        super(PhysicsInformedLoss, self).__init__()
        self.w1 = lambda_huber
        self.w2 = lambda_temp
        self.w3 = lambda_grad
        self.w4 = lambda_ssim
        self.ssim_window_size = ssim_window_size
        self.time_weight_gamma = time_weight_gamma
        self.use_mse_magnitude = use_mse_magnitude
        self.huber_delta = huber_delta
        self.ssim_all_timesteps = ssim_all_timesteps
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.huber_none = nn.HuberLoss(delta=huber_delta, reduction='none')
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    @staticmethod
    def _gaussian_window(window_size, sigma, device, dtype):
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        return w

    def ssim_loss(self, pred, target, data_range=1.0, k1=0.01, k2=0.03):
        """Differentiable SSIM loss. Expects [N, 1, H, W]"""
        if pred.shape[1] != 1 or target.shape[1] != 1:
            pred = pred[:, :1]
            target = target[:, :1]

        device, dtype = pred.device, pred.dtype
        ws = int(self.ssim_window_size)
        ws = max(3, ws)
        if ws % 2 == 0:
            ws += 1
        sigma = 1.5
        window = self._gaussian_window(ws, sigma, device, dtype)

        mu_x = F.conv2d(pred, window, padding=ws // 2, groups=1)
        mu_y = F.conv2d(target, window, padding=ws // 2, groups=1)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(pred * pred, window, padding=ws // 2, groups=1) - mu_x2
        sigma_y2 = F.conv2d(target * target, window, padding=ws // 2, groups=1) - mu_y2
        sigma_xy = F.conv2d(pred * target, window, padding=ws // 2, groups=1) - mu_xy

        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
        return 1.0 - ssim_map.mean()

    def gradient_loss(self, pred, target):
        # Calculate spatial gradients (difference between adjacent pixels)
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        return self.l1(dx_pred, dx_target) + self.l1(dy_pred, dy_target)

    def forward(self, pred_seq, target_seq):
        """
        Expects inputs of shape [Batch, Time, Channels, Height, Width]
        """
        # 1. Magnitude loss (Huber by default; optional MSE) with optional timestep weighting
        if self.time_weight_gamma and pred_seq.shape[1] > 1:
            t = pred_seq.shape[1]
            weights = torch.linspace(0.0, 1.0, steps=t, device=pred_seq.device, dtype=pred_seq.dtype)
            weights = torch.exp(self.time_weight_gamma * weights)
            weights = weights / weights.sum()
            if self.use_mse_magnitude:
                per_elem = (pred_seq - target_seq) ** 2
            else:
                per_elem = self.huber_none(pred_seq, target_seq)  # [B, T, C, H, W]
            per_t = per_elem.mean(dim=(2, 3, 4))             # [B, T]
            loss_huber = (per_t * weights.view(1, t)).sum(dim=1).mean()
        else:
            if self.use_mse_magnitude:
                loss_huber = self.mse(pred_seq, target_seq)
            else:
                loss_huber = self.huber(pred_seq, target_seq)

        # 2. Temporal Consistency (Physics Inertia)
        # Calculate (T_t - T_{t-1}) for both pred and target
        if pred_seq.shape[1] > 1:  # Check if sequence length > 1
            diff_pred = pred_seq[:, 1:] - pred_seq[:, :-1]
            diff_target = target_seq[:, 1:] - target_seq[:, :-1]
            loss_temp = self.mse(diff_pred, diff_target)
        else:
            loss_temp = torch.tensor(0.0, device=pred_seq.device)

        # 3. Spatial Gradient (Sharpness)
        b, t, c, h, w = pred_seq.shape
        loss_grad = self.gradient_loss(
            pred_seq.view(b*t, c, h, w),
            target_seq.view(b*t, c, h, w)
        )

        # 4. SSIM (Structure) on last frame only
        if self.w4 and self.w4 != 0.0:
            if self.ssim_all_timesteps:
                pred_bt = pred_seq.view(b * t, c, h, w)
                targ_bt = target_seq.view(b * t, c, h, w)
                loss_ssim = self.ssim_loss(pred_bt, targ_bt, data_range=1.0)
            else:
                pred_last = pred_seq[:, -1].view(b, c, h, w)
                target_last = target_seq[:, -1].view(b, c, h, w)
                loss_ssim = self.ssim_loss(pred_last, target_last, data_range=1.0)
        else:
            loss_ssim = torch.tensor(0.0, device=pred_seq.device)

        # Composite Loss
        total_loss = (self.w1 * loss_huber) + (self.w2 * loss_temp) + (self.w3 * loss_grad) + (self.w4 * loss_ssim)

        #print(f"MSE: {loss_mse.item():.4f}, Temp: {loss_temp.item():.4f}, Grad: {loss_grad.item():.4f}")

        return total_loss, loss_huber.item(), loss_temp.item(), loss_grad.item()



# 4. Model Architecture


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DoubleConvSE(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x))

class UNetOdd(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=16, use_se=False, bottleneck_1x1=False):
        super().__init__()
        c1, c2, c3, c4, bott = base, base*2, base*4, base*8, base*16
        self.enc0, self.enc1 = DoubleConvSE(in_ch, c1, use_se), DoubleConvSE(c1, c2, use_se)
        self.enc2, self.enc3 = DoubleConvSE(c2, c3, use_se), DoubleConvSE(c3, c4, use_se)
        if bottleneck_1x1:
            self.bott = nn.Sequential(
                nn.Conv2d(c4, bott, 1),
                nn.BatchNorm2d(bott),
                nn.ReLU(inplace=True),
                nn.Conv2d(bott, bott, 1),
                nn.BatchNorm2d(bott),
                nn.ReLU(inplace=True),
            )
        else:
            self.bott = DoubleConvSE(c4, bott, use_se)
        self.up3, self.dec3 = nn.Conv2d(bott, c4, 1), DoubleConvSE(c4*2, c3, use_se)
        self.up2, self.dec2 = nn.Conv2d(c3, c3, 1), DoubleConvSE(c3*2, c2, use_se)
        self.up1, self.dec1 = nn.Conv2d(c2, c2, 1), DoubleConvSE(c2*2, c1, use_se)
        self.up0, self.dec0 = nn.Conv2d(c1, c1, 1), DoubleConvSE(c1*2, c1, use_se)
        self.head = nn.Conv2d(c1, out_ch, 1)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self._resize(e0, 13))
        e2 = self.enc2(self._resize(e1, 8))
        e3 = self.enc3(self._resize(e2, 3))
        b = self.bott(self._resize(e3, 1))

        d3 = self.dec3(torch.cat([self._resize(self.up3(b), 3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._resize(self.up2(d3), 8), e2], dim=1))
        d1 = self.dec1(torch.cat([self._resize(self.up1(d2), 13), e1], dim=1))
        d0 = self.dec0(torch.cat([self._resize(self.up0(d1), 19), e0], dim=1))
        return self.head(d0)

class MultiKernelBlock(nn.Module):
    def __init__(self, in_ch=5, out_per_var=4, kernel_size=3, kernel_sizes=None):
        super().__init__()
        self.in_ch = int(in_ch)
        self.kernel_sizes = None

        # Backward-compatible path (single depthwise conv that outputs multiple maps per channel)
        if kernel_sizes is None:
            self.out_per_var = int(out_per_var)
            self.out_ch = int(self.in_ch * self.out_per_var)
            # groups=in_ch ensures independent channel processing (Depthwise Conv)
            self.depth_conv = nn.Conv2d(
                self.in_ch, self.out_ch, kernel_size, padding=kernel_size//2,
                groups=self.in_ch, bias=False
            )
            self.bn = nn.BatchNorm2d(self.out_ch)
        else:
            # True multi-kernel path: one depthwise conv per kernel size, each producing 1 map per channel
            ks = [int(k) for k in kernel_sizes]
            if len(ks) == 0:
                ks = [int(kernel_size)]
            self.kernel_sizes = ks
            self.out_per_var = int(len(ks))
            self.out_ch = int(self.in_ch * self.out_per_var)
            self.convs = nn.ModuleList([
                nn.Conv2d(
                    self.in_ch,
                    self.in_ch,
                    kernel_size=int(k),
                    padding=int(k)//2,
                    groups=self.in_ch,
                    bias=False
                ) for k in ks
            ])
            self.bn = nn.BatchNorm2d(self.out_ch)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'depth_conv'):
            y = self.depth_conv(x)
        else:
            ys = [conv(x) for conv in self.convs]
            y = torch.cat(ys, dim=1)
        return self.act(self.bn(y))

class CNN_UNet_LSTM(nn.Module):
    def __init__(self, in_ch=5, out_per_var=4, hidden_dim=256, use_se=False, bottleneck_1x1=False, mkb_kernel_sizes=None):
        super().__init__()
        self.mkb = MultiKernelBlock(in_ch=in_ch, out_per_var=out_per_var, kernel_sizes=mkb_kernel_sizes)
        self.unet = UNetOdd(in_ch=in_ch*out_per_var, out_ch=1, use_se=use_se, bottleneck_1x1=bottleneck_1x1)
        self.flatten = nn.Flatten(1)
        self.lstm = nn.LSTM(19*19, hidden_dim, num_layers=2, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_dim, 19*19)

    def forward(self, x):
        B, T, C, H, W = x.shape
        spatial_features_seq = []

        # 1. Process Spatial Features for every frame
        for t in range(T):
            frame = x[:, t, :, :, :]
            f = self.mkb(frame)
            ysp = self.unet(f)
            spatial_features_seq.append(self.flatten(ysp))

        cnn_out_seq = torch.stack(spatial_features_seq, dim=1) # Shape: [B, T, Features]

        # 2. LSTM Processing
        # output shape: [B, T, Hidden]
        lstm_out, _ = self.lstm(cnn_out_seq)

        # --- CORRECTION START ---
        # Instead of taking only the last step (lstm_out[:, -1, :]),
        # we process ALL time steps to match the Target Sequence shape.

        # Flatten Batch and Time dimensions together to apply Linear layer
        # Shape becomes: [B * T, Hidden]
        lstm_out_reshaped = lstm_out.contiguous().view(B * T, -1)

        # Apply FC layer
        y_hat_flat = self.fc(lstm_out_reshaped) # Shape: [B*T, 19*19]

        # Reshape back to Sequence: [B, T, 1, H, W]
        y_hat = y_hat_flat.view(B, T, 1, H, W)
        # --- CORRECTION END ---

        # For visualization consistency, we can still extract the last spatial feature
        last_spatial_out = spatial_features_seq[-1].view(B, 1, H, W)

        return y_hat, last_spatial_out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        h_t = torch.zeros((b, self.hidden_dim, h, w), device=x.device, dtype=x.dtype)
        c_t = torch.zeros((b, self.hidden_dim, h, w), device=x.device, dtype=x.dtype)
        outs = []
        for k in range(t):
            h_t, c_t = self.cell(x[:, k], (h_t, c_t))
            outs.append(h_t)
        return torch.stack(outs, dim=1), (h_t, c_t)


class ConvLSTMForecast(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, kernel_size=3, out_ch=1):
        super().__init__()
        self.convlstm = ConvLSTM(input_dim=in_ch, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.head = nn.Conv2d(hidden_dim, out_ch, kernel_size=1)

    def forward(self, x):
        feats, state = self.convlstm(x)
        b, t, c, h, w = feats.shape
        y = self.head(feats.view(b * t, c, h, w)).view(b, t, -1, h, w)
        return y, state


def _param_bytes(model):
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def _fmt_bytes(n):
    n = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def benchmark_mkcnn_vs_convlstm(in_ch, sequence_length=6, batch_size=8, hidden_dim=64,
                               out_per_var=2, use_se=False, bottleneck_1x1=False,
                               iters=50, warmup=10, device=None, output_dir='output_plots'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mk = CNN_UNet_LSTM(
        in_ch=in_ch,
        out_per_var=out_per_var,
        hidden_dim=hidden_dim,
        use_se=use_se,
        bottleneck_1x1=bottleneck_1x1
    ).to(device)
    cl = ConvLSTMForecast(in_ch=in_ch, hidden_dim=hidden_dim, kernel_size=3, out_ch=1).to(device)
    mk.eval()
    cl.eval()

    x = torch.randn(batch_size, sequence_length, in_ch, 19, 19, device=device)

    def _time_forward(model):
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(int(warmup)):
                y, _ = model(x)
                _ = y.mean().item()
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(int(iters)):
                y, _ = model(x)
                _ = y.mean().item()
            if device == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        elapsed = max(1e-9, (t1 - t0))
        samples_per_s = (float(iters) * float(batch_size)) / elapsed
        peak = None
        if device == 'cuda':
            peak = int(torch.cuda.max_memory_allocated())
        return float(samples_per_s), peak

    mk_tput, mk_peak = _time_forward(mk)
    cl_tput, cl_peak = _time_forward(cl)

    rows = [
        {
            'Model': 'MKCNN-UNet-LSTM',
            'Params': int(sum(p.numel() for p in mk.parameters())),
            'ParamMemory': _fmt_bytes(_param_bytes(mk)),
            'PeakCudaMem': (_fmt_bytes(mk_peak) if mk_peak is not None else 'N/A'),
            'Throughput_samples_per_s': f"{mk_tput:.2f}"
        },
        {
            'Model': 'ConvLSTM',
            'Params': int(sum(p.numel() for p in cl.parameters())),
            'ParamMemory': _fmt_bytes(_param_bytes(cl)),
            'PeakCudaMem': (_fmt_bytes(cl_peak) if cl_peak is not None else 'N/A'),
            'Throughput_samples_per_s': f"{cl_tput:.2f}"
        }
    ]

    headers = ['Model', 'Params', 'ParamMemory', 'PeakCudaMem', 'Throughput_samples_per_s']
    colw = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    sep = ' | '
    line = '-+-'.join('-' * colw[h] for h in headers)

    print("\n=== MKCNN vs ConvLSTM Benchmark ===")
    print(f"Device: {device}")
    print(f"Input: batch={batch_size}, seq={sequence_length}, ch={in_ch}, H=W=19")
    print(f"Iters: {iters} (warmup {warmup})")
    print(sep.join(h.ljust(colw[h]) for h in headers))
    print(line)
    for r in rows:
        print(sep.join(str(r[h]).ljust(colw[h]) for h in headers))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'benchmark_mkcnn_vs_convlstm.csv'
    try:
        import csv
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved: {csv_path}")
    except Exception as e:
        print(f"Warning: could not save benchmark CSV ({e})")

    return rows



# 5. Evaluation Metrics

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())

def calculate_rmse(y_true, y_pred):
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_metrics_per_horizon(target_seq, pred_seq, ssim_data_range=None):
    """Compute metrics for each horizon.

    Expects tensors on CPU of shape [N, T, C, H, W].
    Returns dict of lists with length T.
    """
    if target_seq is None or pred_seq is None:
        return {
            'MSE': [],
            'RMSE': [],
            'SSIM': [],
        }

    if target_seq.ndim != 5 or pred_seq.ndim != 5:
        raise ValueError(f"Expected [N,T,C,H,W] tensors, got target_seq={tuple(target_seq.shape)} pred_seq={tuple(pred_seq.shape)}")

    if target_seq.shape[:2] != pred_seq.shape[:2]:
        raise ValueError(f"Mismatched seq shapes: target_seq={tuple(target_seq.shape)} pred_seq={tuple(pred_seq.shape)}")

    t = int(target_seq.shape[1])
    out = {'MSE': [], 'RMSE': [], 'SSIM': []}
    for h in range(t):
        yt = target_seq[:, h]
        yp = pred_seq[:, h]
        mse_v = float(calculate_mse(yt, yp))
        out['MSE'].append(mse_v)
        out['RMSE'].append(float(np.sqrt(mse_v)))
        out['SSIM'].append(float(calculate_ssim(yt, yp, ssim_data_range)))
    return out


def plot_metrics_vs_horizon(metrics_by_h, output_dir='output_plots', prefix=''):
    """Save MSE/RMSE/SSIM vs horizon plots."""
    if metrics_by_h is None:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot(values, title, ylabel, filename, color):
        if values is None or len(values) == 0:
            return
        xs = np.arange(1, len(values) + 1)
        ys = np.asarray(values, dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(3.8, 4.0))
        ax.plot(xs, ys, color=color, linewidth=1.7, marker='o', markersize=6.5,
                markeredgecolor='black', markeredgewidth=0.3)
        ax.set_xlabel('Horizon (steps ahead)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis='both', labelsize=7)
        ax.set_xticks(xs)
        plt.tight_layout(pad=0.6)
        out_path = out_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    p = f"{prefix}_" if prefix else ''
    _plot(metrics_by_h.get('MSE', []), f'{p}MSE vs Horizon', 'MSE', f'{p}mse_vs_horizon.png', '#1f77b4')
    _plot(metrics_by_h.get('RMSE', []), f'{p}RMSE vs Horizon', 'RMSE', f'{p}rmse_vs_horizon.png', '#2ca02c')
    _plot(metrics_by_h.get('SSIM', []), f'{p}SSIM vs Horizon', 'SSIM', f'{p}ssim_vs_horizon.png', '#ff7f0e')


def _evaluate_simple_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targs = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            pred_seq, _ = model(inputs)
            pred_last = pred_seq[:, -1].detach().cpu()
            targ_last = targets[:, -1].detach().cpu()
            all_preds.append(pred_last)
            all_targs.append(targ_last)
    if len(all_preds) == 0:
        return {'MAE': float('nan'), 'MSE': float('nan'), 'RMSE': float('nan'), 'SSIM': float('nan')}
    all_preds = torch.cat(all_preds, dim=0)
    all_targs = torch.cat(all_targs, dim=0)
    mse_v = calculate_mse(all_targs, all_preds)
    mae_v = calculate_mae(all_targs, all_preds)
    rmse_v = float(np.sqrt(mse_v))
    ssim_v = calculate_ssim(all_targs, all_preds)
    return {'MAE': float(mae_v), 'MSE': float(mse_v), 'RMSE': float(rmse_v), 'SSIM': float(ssim_v)}


def plot_metric_vs_epochs_multi_horizon(histories_by_horizon, metric_key, output_dir='output_plots', filename='metric_vs_epochs.png', title=None, ylabel=None):
    if histories_by_horizon is None or len(histories_by_horizon) == 0:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2))
    for horizon, hist in sorted(histories_by_horizon.items(), key=lambda kv: int(kv[0])):
        if hist is None:
            continue
        ys = hist.get(metric_key, None)
        if ys is None or len(ys) == 0:
            continue
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, np.asarray(ys, dtype=float), linewidth=1.8, label=f'H={horizon}')

    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel((ylabel if ylabel else metric_key), fontsize=9)
    ax.set_title((title if title else f'{metric_key} vs Epochs'), fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout(pad=0.7)
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def _apply_aerosol_mask(inputs, aerosol_mask):
    if aerosol_mask is None:
        return inputs
    m = np.asarray(aerosol_mask, dtype=np.float32).reshape(-1)
    if m.size != 3:
        return inputs
    mask_t = torch.tensor(m, device=inputs.device, dtype=inputs.dtype).view(1, 1, 3, 1, 1)
    inputs = inputs.clone()
    inputs[:, :, 0:3, :, :] = inputs[:, :, 0:3, :, :] * mask_t
    return inputs


def plot_test_metrics_history(history, output_dir='output_plots'):
    if history is None:
        return
    epochs = history.get('test_eval_epochs', [])
    if epochs is None or len(epochs) == 0:
        return
    maes = history.get('test_mae', [])
    mses = history.get('test_mse', [])
    rmses = history.get('test_rmse', [])

    def _smooth(y, window=3):
        y = np.asarray(y, dtype=float)
        if y.size < 3 or window <= 1:
            return y
        w = min(int(window), int(y.size))
        if w < 2:
            return y
        kernel = np.ones(w, dtype=float) / float(w)
        pad_left = w // 2
        pad_right = w - 1 - pad_left
        ypad = np.pad(y, (pad_left, pad_right), mode='edge')
        return np.convolve(ypad, kernel, mode='valid')

    def _plot_single(epochs_local, y, y_s, title, ylabel, out_path, color):
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.8))
        epochs_local = np.asarray(epochs_local, dtype=float)
        y = np.asarray(y, dtype=float)

        ax.plot(
            epochs_local, y,
            color=color,
            linewidth=1.6,
            marker='o',
            markersize=7.5,
            markeredgecolor='black',
            markeredgewidth=0.3,
            label='Local'
        )
        if y_s is not None and len(y_s) == len(epochs_local) and len(epochs_local) >= 3:
            ax.plot(epochs_local, y_s, color=color, linewidth=1.2, linestyle=':', alpha=0.95, label='Global')

        # Avoid ultra-tight axis scaling when there are few points
        xmin = float(np.min(epochs_local))
        xmax = float(np.max(epochs_local))
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        ax.set_xlim(xmin - 0.5, xmax + 0.5)

        if np.isfinite(y).any():
            ymin = float(np.nanmin(y))
            ymax = float(np.nanmax(y))
            if ymin == ymax:
                ymin -= 1e-6
                ymax += 1e-6
            margin = 0.10 * (ymax - ymin)
            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis='both', labelsize=7)
        ax.legend(fontsize=7, frameon=True)
        plt.tight_layout(pad=0.6)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 5.0))
    mses_s = _smooth(mses, window=3)
    maes_s = _smooth(maes, window=3)
    rmses_s = _smooth(rmses, window=3)

    c_mse = '#1f77b4'
    c_mae = '#ff7f0e'
    c_rmse = '#2ca02c'
    ax.plot(epochs, mses_s, label='Test MSE', linewidth=1.6, color=c_mse)
    ax.plot(epochs, maes_s, label='Test MAE', linewidth=1.6, color=c_mae)
    ax.plot(epochs, rmses_s, label='Test RMSE', linewidth=1.6, color=c_rmse)
    ax.scatter(epochs, mses, s=26, color=c_mse, alpha=0.9)
    ax.scatter(epochs, maes, s=26, color=c_mae, alpha=0.9)
    ax.scatter(epochs, rmses, s=26, color=c_rmse, alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Metric', fontsize=9)
    ax.set_title('Test metrics vs epoch', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    out_path = out_dir / 'test_metrics_timeseries.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.close(fig)

    _plot_single(epochs, mses, mses_s, 'Test MSE vs Epoch', 'MSE', out_dir / 'test_mse_vs_epoch.png', c_mse)
    _plot_single(epochs, maes, maes_s, 'Test MAE vs Epoch', 'MAE', out_dir / 'test_mae_timeseries.png', c_mae)
    _plot_single(epochs, rmses, rmses_s, 'Test RMSE vs Epoch', 'RMSE', out_dir / 'test_rmse_timeseries.png', c_rmse)

def calculate_mape(y_true, y_pred, epsilon=1e-8):
    y_true_np, y_pred_np = y_true.cpu().numpy(), y_pred.cpu().numpy()
    denominator = np.maximum(np.abs(y_true_np), epsilon)
    return np.mean(np.abs((y_true_np - y_pred_np) / denominator)) * 100

def calculate_ssim(y_true, y_pred, data_range=None):
    y_true_np, y_pred_np = y_true.cpu().numpy(), y_pred.cpu().numpy()
    ssim_scores = []

    # If data_range is missing, infer it (Dynamic Range)
    if data_range is None:
        data_range = np.max(y_true_np) - np.min(y_true_np)

    for i in range(y_true_np.shape[0]):
        for c in range(y_true_np.shape[1]):
            try:
                # Use a smaller window size (3 or 5) for 19x19 images
                score = ssim(y_true_np[i, c], y_pred_np[i, c],
                             data_range=data_range, win_size=3)
                ssim_scores.append(score)
            except ValueError as e:
                # Print error once to debug, then silence
                if len(ssim_scores) == 0: print(f"SSIM Error: {e}")
                ssim_scores.append(0.0)
    return np.mean(ssim_scores)

def evaluate_model(model, test_loader, device, ssim_data_range=None):
    """Evaluate single model on test set - uses last frame of sequence"""
    model.eval()
    all_predictions, all_targets, inference_times = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            start_time = time.time()
            pred_seq, _ = model(inputs)
            inference_times.append(time.time() - start_time)

            # Extract last frame for evaluation
            all_predictions.append(pred_seq[:, -1, :, :, :].cpu())
            all_targets.append(targets[:, -1, :, :, :].cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    results = {
        'MAE': calculate_mae(all_targets, all_predictions),
        'MSE': calculate_mse(all_targets, all_predictions),
        'RMSE': calculate_rmse(all_targets, all_predictions),
        'MAPE': calculate_mape(all_targets, all_predictions),
        'SSIM': calculate_ssim(all_targets, all_predictions, ssim_data_range),
        'Avg_Inference_Time(s)': np.mean(inference_times)
    }

    return results




# 6. Training Function with Physics-Informed Loss


def train_model(model, train_loader, val_loader, device,num_epochs=50, lr=0.001, patience=10,
                lambda_huber=1.0, lambda_temp=1.5, lambda_grad=0.5, lambda_ssim=0.0, ssim_window_size=7, time_weight_gamma=0.0, use_mse_magnitude=False, huber_delta=2.0, ssim_all_timesteps=False, lr_schedule='plateau', noise_std=0.0, verbose=True,
                test_loader=None, test_eval_every=5, output_dir='output_plots', aerosol_mask=None,
                checkpoint_name=None, track_val_metrics_each_epoch=True):
    # Define model
    model = model.to(device)

    # Define Criterion
    criterion = PhysicsInformedLoss(
        lambda_huber=lambda_huber,
        lambda_temp=lambda_temp,
        lambda_grad=lambda_grad,
        lambda_ssim=lambda_ssim,
        ssim_window_size=ssim_window_size,
        time_weight_gamma=time_weight_gamma,
        use_mse_magnitude=use_mse_magnitude,
        huber_delta=huber_delta,
        ssim_all_timesteps=ssim_all_timesteps
    )

    # Define Optimizer    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Define scheduler
    if lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    history = {
        'train_losses': [],
        'val_losses': [],
        'train_mse': [],
        'train_temp': [],
        "train_grad": [],
        'val_mse_metric': [],
        'val_rmse_metric': [],
        'val_ssim_metric': [],
        'test_eval_epochs': [],
        'test_mae': [],
        'test_mse': [],
        'test_rmse': [],
        'test_ssim': []
    }
    best_val_loss = float('inf')
    patience_counter = 0

    if verbose:
        print(f"\nTraining MKCNN_UNet_LSTM with PhysicsInformedLoss")
        print(f"Lambda MSE: {lambda_huber}, Lambda Temp: {lambda_temp}, Lambda Grad: {lambda_grad}")
        print("="*70)

    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_huber_accum = 0.0
        train_temp_accum = 0.0
        train_grad_accum = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = _apply_aerosol_mask(inputs, aerosol_mask)
            if noise_std and noise_std > 0.0:
                inputs = inputs + (torch.randn_like(inputs) * float(noise_std))
            optimizer.zero_grad()

            # Forward pass returns sequence [B, T, 1, H, W]
            pred_seq, _ = model(inputs)

            # Calculate loss
            loss, huber_component, temp_component, grad_component = criterion(pred_seq, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_huber_accum += huber_component
            train_temp_accum += temp_component
            train_grad_accum += grad_component

        avg_train_loss = train_loss / len(train_loader)
        avg_train_huber = train_huber_accum / len(train_loader)
        avg_train_temp = train_temp_accum / len(train_loader)
        avg_train_grad = train_grad_accum / len(train_loader)

        history['train_losses'].append(avg_train_loss)
        history['train_mse'].append(avg_train_huber)
        history['train_temp'].append(avg_train_temp)
        history['train_grad'].append(avg_train_grad)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = _apply_aerosol_mask(inputs, aerosol_mask)
                pred_seq, _ = model(inputs)
                loss, _, _, _ = criterion(pred_seq, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_losses'].append(avg_val_loss)

        if track_val_metrics_each_epoch:
            metrics_val = _evaluate_simple_metrics(model, val_loader, device)
            history['val_mse_metric'].append(metrics_val.get('MSE', float('nan')))
            history['val_rmse_metric'].append(metrics_val.get('RMSE', float('nan')))
            history['val_ssim_metric'].append(metrics_val.get('SSIM', float('nan')))

        if lr_schedule == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            mkb_ks = None
            try:
                mkb_ks = getattr(getattr(model, 'mkb', None), 'kernel_sizes', None)
            except Exception:
                mkb_ks = None
            if checkpoint_name is not None and str(checkpoint_name).strip() != '':
                ckpt_name = str(checkpoint_name)
            else:
                if mkb_ks is not None and len(mkb_ks) > 0:
                    ks_str = '_'.join(str(int(k)) for k in mkb_ks)
                    ckpt_name = f'best_mkcnn_unet_lstm_physics_mkb_{ks_str}.pth'
                else:
                    ckpt_name = 'best_mkcnn_unet_lstm_physics.pth'
            out_dir = Path(output_dir) if output_dir else Path('.')
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / ckpt_name
            torch.save(best_model_state, ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f} (MSE: {avg_train_huber:.4f}, Temp: {avg_train_temp:.4f}, Grad: {avg_train_grad} | "
                  f"Val Loss: {avg_val_loss:.4f}")

        if test_loader is not None and test_eval_every and test_eval_every > 0 and ((epoch + 1) % int(test_eval_every) == 0):
            with torch.no_grad():
                masked_batches = []
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    xb = _apply_aerosol_mask(xb, aerosol_mask)
                    masked_batches.append((xb, yb))

            if len(masked_batches) > 0:
                class _TmpDL:
                    def __init__(self, batches):
                        self.batches = batches
                    def __iter__(self):
                        for b in self.batches:
                            yield b
                metrics = _evaluate_simple_metrics(model, _TmpDL(masked_batches), device)
            else:
                metrics = {'MAE': float('nan'), 'MSE': float('nan'), 'RMSE': float('nan'), 'SSIM': float('nan')}
            history['test_eval_epochs'].append(epoch + 1)
            history['test_mae'].append(metrics['MAE'])
            history['test_mse'].append(metrics['MSE'])
            history['test_rmse'].append(metrics['RMSE'])
            history['test_ssim'].append(metrics.get('SSIM', float('nan')))
            if verbose:
                print(f"Test @ epoch {epoch+1}: MAE={metrics['MAE']:.4f} MSE={metrics['MSE']:.4f} RMSE={metrics['RMSE']:.4f} SSIM={metrics.get('SSIM', float('nan')):.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is None:
        best_model_state = model.state_dict()
    model.load_state_dict(best_model_state)
    history['best_val_loss'] = best_val_loss
    history['epochs_trained'] = epoch + 1

    if verbose:
        print("="*70)
        print(f"Training completed. Best Val Loss: {best_val_loss:.4f}")

    plot_test_metrics_history(history, output_dir=output_dir)

    return model, history


# 7. Visualization Functions

def plot_training_history(history):
    """Plot training and validation loss curves with components"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    epochs = np.arange(1, len(history['train_losses']) + 1)

    def _smooth(y, window=3):
        y = np.asarray(y, dtype=float)
        if y.size < 3 or window <= 1:
            return y
        w = min(int(window), int(y.size))
        if w < 2:
            return y
        kernel = np.ones(w, dtype=float) / float(w)
        pad_left = w // 2
        pad_right = w - 1 - pad_left
        ypad = np.pad(y, (pad_left, pad_right), mode='edge')
        return np.convolve(ypad, kernel, mode='valid')

    train_loss_s = _smooth(history['train_losses'], window=3)
    val_loss_s = _smooth(history['val_losses'], window=3)

    # Total Loss
    axes[0].plot(epochs, train_loss_s, color='#1f77b4', linewidth=1.8, label='Train')
    axes[0].plot(epochs, val_loss_s, color='#d62728', linewidth=1.8, label='Val')
    axes[0].scatter(epochs, history['train_losses'], color='#1f77b4', s=8, alpha=0.35)
    axes[0].scatter(epochs, history['val_losses'], color='#d62728', s=8, alpha=0.35)
    axes[0].set_xlabel('Epoch', fontsize=9)
    axes[0].set_ylabel('Loss', fontsize=9)
    axes[0].set_title('Total Loss', fontsize=10, fontweight='bold')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].tick_params(axis='both', labelsize=8)

    # Loss Components
    mse_s = _smooth(history.get('train_mse', []), window=3)
    temp_s = _smooth(history.get('train_temp', []), window=3)
    grad_s = _smooth(history.get('train_grad', []), window=3)
    if len(mse_s) == len(epochs):
        axes[1].plot(epochs, mse_s, color='#2ca02c', linewidth=1.6, label='Magnitude')
    if len(temp_s) == len(epochs):
        axes[1].plot(epochs, temp_s, color='#9467bd', linewidth=1.6, label='Temporal')
    if len(grad_s) == len(epochs):
        axes[1].plot(epochs, grad_s, color='#ff7f0e', linewidth=1.6, label='Gradient')
    axes[1].set_xlabel('Epoch', fontsize=9)
    axes[1].set_ylabel('Component', fontsize=9)
    axes[1].set_title('Loss Components (Train)', fontsize=10, fontweight='bold')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].tick_params(axis='both', labelsize=8)

    plt.tight_layout(pad=1.0)
    plt.savefig('training_history_physics.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved to 'training_history_physics.png'")
    plt.close(fig)


def plot_predictions(model, test_loader, device, num_samples_to_plot=3,
                    shapefile_path='/content/drive/MyDrive/India_Boundary_Files/India_State_Boundary.shp',
                    mask_to_shape=True,
                    mu_y=None,
                    sd_y=None,
                    mu_x=None,
                    sd_x=None,
                    output_dir='output_plots'):
    """
    Visualize predictions with Shapefile masking and individual colorbars.
    """
    print("\n--- Generating Prediction Visualizations ---")

    out_dir = Path(output_dir) / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and Prepare Shapefile
    gdf = None
    try:
        if Path(shapefile_path).exists():
            gdf = gpd.read_file(shapefile_path)
            try:
                if gdf is not None and getattr(gdf, 'crs', None) is not None and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
            except Exception as e:
                print(f"Warning: could not reproject shapefile to EPSG:4326 ({e}).")
            # Clip shapefile to your grid extent (70-81.25 E, 28-37 N)
            extent_box = box(70.0, 28.0, 81.25, 37.0)
            gdf = gdf.clip(extent_box)
            if gdf is None or len(gdf) == 0:
                print("Warning: Shapefile clipped to grid extent but resulted in empty geometry. Plotting without map.")
                gdf = None
            else:
                print("Shapefile loaded and clipped to grid extent.")
                try:
                    print(f"Shapefile CRS: {gdf.crs}")
                    print(f"Shapefile bounds: {gdf.total_bounds}")
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: Shapefile error ({e}). Plotting without map.")

    def build_mask_19x19(gdf_local, extent_local):
        """Return a (19,19) boolean mask where True indicates inside geometry."""
        if gdf_local is None or len(gdf_local) == 0:
            return None
        geom = gdf_local.unary_union
        if geom is None or geom.is_empty:
            return None
        prepared = prep(geom)
        lon_min, lon_max, lat_min, lat_max = extent_local
        lons = np.linspace(lon_min, lon_max, 19)
        lats = np.linspace(lat_max, lat_min, 19)  # top-to-bottom to match origin='upper'
        mask = np.zeros((19, 19), dtype=bool)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # covers() includes boundary points; contains() excludes boundary and can create gaps.
                try:
                    mask[i, j] = prepared.covers(Point(float(lon), float(lat)))
                except Exception:
                    mask[i, j] = prepared.contains(Point(float(lon), float(lat)))
        return mask

    model.eval()
    samples_plotted = 0
    extent = [70.0, 81.25, 28.0, 37.0] # [Lon Min, Lon Max, Lat Min, Lat Max]
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.linspace(lon_min, lon_max, 19)
    lats = np.linspace(lat_max, lat_min, 19)  # top-to-bottom to match origin='upper'
    Lon, Lat = np.meshgrid(lons, lats)
    shape_mask = build_mask_19x19(gdf, extent) if mask_to_shape else None
    if mask_to_shape and shape_mask is None:
        print("Warning: mask_to_shape=True but no valid shape mask could be built (missing/empty shapefile geometry).")
    if shape_mask is not None:
        try:
            inside = int(shape_mask.sum())
            print(f"Shape mask coverage: {inside}/361 cells inside geometry")
        except Exception:
            pass

    with torch.no_grad():
        for inputs, targets in test_loader:
            if samples_plotted >= num_samples_to_plot: break

            inputs, targets = inputs.to(device), targets.to(device)
            pred_seq, last_spatial_out = model(inputs) # Forward pass

            for i in range(inputs.shape[0]):
                if samples_plotted >= num_samples_to_plot: break

                # Extract Last Timestep Data
                # Inputs: (T, 5, H, W) -> Take T=-1
                bc_aod = inputs[i, -1, 0].cpu().numpy()
                su_aod = inputs[i, -1, 1].cpu().numpy()
                du_aod = inputs[i, -1, 2].cpu().numpy()

                if mu_x is not None and sd_x is not None:
                    try:
                        bc_aod = (bc_aod * float(sd_x[0])) + float(mu_x[0])
                        su_aod = (su_aod * float(sd_x[1])) + float(mu_x[1])
                        du_aod = (du_aod * float(sd_x[2])) + float(mu_x[2])
                    except Exception:
                        pass

                # Target & Prediction
                true_temp = targets[i, -1, 0].cpu().numpy()
                pred_temp = pred_seq[i, -1, 0].cpu().numpy()

                if mu_y is not None and sd_y is not None:
                    true_temp = (true_temp * float(sd_y)) + float(mu_y)
                    pred_temp = (pred_temp * float(sd_y)) + float(mu_y)

                true_temp_m = np.where(shape_mask, true_temp, np.nan) if shape_mask is not None else true_temp
                pred_temp_m = np.where(shape_mask, pred_temp, np.nan) if shape_mask is not None else pred_temp
                temp_stack = np.stack([true_temp_m, pred_temp_m], axis=0)
                if np.isfinite(temp_stack).any():
                    temp_vmin = float(np.nanmin(temp_stack))
                    temp_vmax = float(np.nanmax(temp_stack))
                else:
                    temp_vmin, temp_vmax = 0.0, 1.0
                if temp_vmin == temp_vmax:
                    temp_vmin -= 1e-6
                    temp_vmax += 1e-6

                # --- Plotting ---
                fig, axes = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)

                plot_configs = [
                    (bc_aod, 'Input: BC AOD', 'RdBu_r'),
                    (su_aod, 'Input: SU AOD', 'RdBu_r'),
                    (du_aod, 'Input: DU AOD', 'RdBu_r'),
                    (true_temp, 'Ground Truth (Temp)', 'RdBu_r'),
                    (pred_temp, 'Prediction (Temp)', 'RdBu_r')
                ]

                for ax, (data, title, cmap) in zip(axes, plot_configs):
                    if shape_mask is not None:
                        data = np.asarray(data, dtype=float)
                        data = np.ma.masked_where(~shape_mask, data)
                    cmap_obj = plt.get_cmap(cmap)
                    try:
                        cmap_obj = cmap_obj.copy()
                    except Exception:
                        pass
                    try:
                        midpoint = cmap_obj(0.5)
                        bg = (1.0, 1.0, 1.0, 1.0)
                        cmap_obj.set_bad(color=bg, alpha=1.0)
                    except Exception:
                        pass

                    # Match background to colormap midpoint (also used for masked-out regions)
                    try:
                        ax.set_facecolor(bg)
                    except Exception:
                        pass

                    ax.grid(True, which='both', color='0.85', linewidth=0.6, alpha=0.8)

                    if isinstance(data, np.ma.MaskedArray):
                        data_vals = np.asarray(data.data)
                        finite = (~np.asarray(data.mask)) & np.isfinite(data_vals)
                    else:
                        data_vals = np.asarray(data)
                        finite = np.isfinite(data_vals)
                    if 'Temp' in title:
                        vmin, vmax = temp_vmin, temp_vmax
                    else:
                        if finite.any():
                            vmin = float(np.nanmin(data_vals[finite]))
                            vmax = float(np.nanmax(data_vals[finite]))
                        else:
                            vmin, vmax = 0.0, 1.0

                    if vmin == vmax:
                        vmin -= 1e-6
                        vmax += 1e-6

                    levels = np.linspace(vmin, vmax, 15)

                    # 1. Background: filled contours
                    cf = ax.contourf(Lon, Lat, data, levels=levels, cmap=cmap_obj, extend='both', alpha=0.55)

                    # 2. Triangle markers at grid points
                    ax.scatter(
                        Lon[finite], Lat[finite],
                        c=data_vals[finite],
                        cmap=cmap_obj,
                        vmin=vmin, vmax=vmax,
                        marker='^',
                        s=240,
                        linewidths=0.0,
                        zorder=2
                    )

                    # 2. Overlay Shapefile
                    if gdf is not None:
                        gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, alpha=0.95, zorder=3)

                    # 3. Aesthetics
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.tick_params(labelsize=8)
                    ax.set_xlim(lon_min, lon_max)
                    ax.set_ylim(lat_min, lat_max)

                    # 4. Individual Colorbar
                    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_obj)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
                    cbar.ax.tick_params(labelsize=8)

                plt.suptitle(f'Test Sample {samples_plotted + 1}', fontsize=16, y=1.05)
                out_path = out_dir / f'prediction_sample_{samples_plotted+1}_physics.png'
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {out_path}")
                plt.close(fig)

                try:
                    # Build a single spatial map BEFORE UNet (UNet input) from the last timestep.
                    frame_last = inputs[i, -1].detach()
                    mkb_before = model.mkb(frame_last.unsqueeze(0))
                    before_feats = mkb_before[0].detach().cpu().numpy()  # [C, H, W]
                    unet_map = last_spatial_out[i, 0].detach().cpu().numpy()

                    if shape_mask is not None:
                        before_feats = np.where(shape_mask[None, :, :], before_feats, np.nan)
                    if shape_mask is not None:
                        unet_map = np.where(shape_mask, unet_map, np.nan)

                    # Use a shared color scale for before/after to make comparison meaningful.
                    n_show = int(min(12, before_feats.shape[0]))
                    show_feats = before_feats[:n_show]
                    stack_ba = np.concatenate([show_feats, unet_map[None, :, :]], axis=0)
                    finite_u = np.isfinite(stack_ba)
                    if finite_u.any():
                        umin = float(np.nanmin(stack_ba))
                        umax = float(np.nanmax(stack_ba))
                    else:
                        umin, umax = 0.0, 1.0
                    if umin == umax:
                        umin -= 1e-6
                        umax += 1e-6

                    ulevels = np.linspace(umin, umax, 15)

                    # Side-by-side BEFORE vs AFTER UNet
                    import matplotlib.gridspec as gridspec
                    ncols = 4
                    nrows = int(np.ceil(n_show / float(ncols)))
                    fig_u = plt.figure(figsize=(10.6, 3.9 + 1.55 * max(0, nrows - 1)), constrained_layout=True)
                    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols + 1, figure=fig_u, width_ratios=[1]*ncols + [1.2])
                    cmap_u = plt.get_cmap('RdBu_r')
                    try:
                        cmap_u = cmap_u.copy()
                    except Exception:
                        pass
                    bg_u = (1.0, 1.0, 1.0, 1.0)
                    try:
                        cmap_u.set_bad(color=bg_u, alpha=1.0)
                    except Exception:
                        pass

                    axes_list = []
                    for idx in range(n_show):
                        r = int(idx // ncols)
                        c = int(idx % ncols)
                        ax_u = fig_u.add_subplot(gs[r, c])
                        axes_list.append(ax_u)
                        data_u = show_feats[idx]
                        title_u = f'Before UNet ch{idx}'
                        try:
                            ax_u.set_facecolor(bg_u)
                        except Exception:
                            pass
                        ax_u.grid(True, which='both', color='0.88', linewidth=0.5, alpha=0.75)
                        finite_local = np.isfinite(data_u)
                        ax_u.contourf(Lon, Lat, data_u, levels=ulevels, cmap=cmap_u, extend='both', alpha=0.55)
                        ax_u.scatter(
                            Lon[finite_local], Lat[finite_local],
                            c=data_u[finite_local],
                            cmap=cmap_u,
                            vmin=umin, vmax=umax,
                            marker='^',
                            s=120,
                            linewidths=0.0,
                            zorder=2
                        )
                        if gdf is not None:
                            gdf.boundary.plot(ax=ax_u, color='black', linewidth=1.2, alpha=0.9, zorder=3)
                        ax_u.set_title(title_u, fontsize=9)
                        ax_u.tick_params(labelsize=7)
                        ax_u.set_xlim(lon_min, lon_max)
                        ax_u.set_ylim(lat_min, lat_max)

                    ax_after = fig_u.add_subplot(gs[:, -1])
                    axes_list.append(ax_after)
                    try:
                        ax_after.set_facecolor(bg_u)
                    except Exception:
                        pass
                    ax_after.grid(True, which='both', color='0.85', linewidth=0.6, alpha=0.8)
                    finite_after = np.isfinite(unet_map)
                    ax_after.contourf(Lon, Lat, unet_map, levels=ulevels, cmap=cmap_u, extend='both', alpha=0.55)
                    ax_after.scatter(
                        Lon[finite_after], Lat[finite_after],
                        c=unet_map[finite_after],
                        cmap=cmap_u,
                        vmin=umin, vmax=umax,
                        marker='^',
                        s=220,
                        linewidths=0.0,
                        zorder=2
                    )
                    if gdf is not None:
                        gdf.boundary.plot(ax=ax_after, color='black', linewidth=1.8, alpha=0.95, zorder=3)
                    ax_after.set_title('After UNet', fontsize=11, fontweight='bold')
                    ax_after.tick_params(labelsize=8)
                    ax_after.set_xlim(lon_min, lon_max)
                    ax_after.set_ylim(lat_min, lat_max)

                    sm_u = ScalarMappable(norm=Normalize(vmin=umin, vmax=umax), cmap=cmap_u)
                    sm_u.set_array([])
                    cbar_u = fig_u.colorbar(sm_u, ax=axes_list, orientation='horizontal', pad=0.06, fraction=0.06)
                    cbar_u.ax.tick_params(labelsize=8)

                    out_u = out_dir / f'unet_before_after_sample_{samples_plotted+1}.png'
                    plt.savefig(out_u, dpi=300, bbox_inches='tight')
                    print(f"Saved: {out_u}")
                    plt.close(fig_u)
                except Exception as e:
                    print(f"Warning: could not save UNet spatial plot ({e})")

                samples_plotted += 1


def plot_mkcnn_spatial_outputs(model, test_loader, device, num_samples_to_plot=3,
                              shapefile_path="NWH_States_Shapefile-20260120T110552Z-3-001/NWH_States_Shapefile/NWH_states.shp",
                              mask_to_shape=True,
                              output_dir='output_plots',
                              mu_x=None,
                              sd_x=None):
    print("\n--- Generating MKCNN Spatial Visualizations ---")

    out_dir = Path(output_dir) / 'mkcnn_features'
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = None
    try:
        if Path(shapefile_path).exists():
            gdf = gpd.read_file(shapefile_path)
            try:
                if gdf is not None and getattr(gdf, 'crs', None) is not None and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
            except Exception as e:
                print(f"Warning: could not reproject shapefile to EPSG:4326 ({e}).")
            extent_box = box(70.0, 28.0, 81.25, 37.0)
            gdf = gdf.clip(extent_box)
            if gdf is None or len(gdf) == 0:
                print("Warning: Shapefile clipped to grid extent but resulted in empty geometry. Plotting without map.")
                gdf = None
            else:
                print("Shapefile loaded and clipped to grid extent.")
    except Exception as e:
        print(f"Warning: Shapefile error ({e}). Plotting without map.")

    def build_mask_19x19(gdf_local, extent_local):
        if gdf_local is None or len(gdf_local) == 0:
            return None
        geom = gdf_local.unary_union
        if geom is None or geom.is_empty:
            return None
        prepared = prep(geom)
        lon_min, lon_max, lat_min, lat_max = extent_local
        lons = np.linspace(lon_min, lon_max, 19)
        lats = np.linspace(lat_max, lat_min, 19)
        mask = np.zeros((19, 19), dtype=bool)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                mask[i, j] = prepared.contains(Point(float(lon), float(lat)))
        return mask

    model.eval()
    samples_plotted = 0
    extent = [70.0, 81.25, 28.0, 37.0]
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.linspace(lon_min, lon_max, 19)
    lats = np.linspace(lat_max, lat_min, 19)
    Lon, Lat = np.meshgrid(lons, lats)
    shape_mask = build_mask_19x19(gdf, extent) if mask_to_shape else None

    with torch.no_grad():
        for inputs, targets in test_loader:
            if samples_plotted >= num_samples_to_plot:
                break

            inputs = inputs.to(device)

            for i in range(inputs.shape[0]):
                if samples_plotted >= num_samples_to_plot:
                    break

                frame = inputs[i, -1]
                mkb_feat = model.mkb(frame.unsqueeze(0))
                mkb_feat_np = mkb_feat[0].detach().cpu().numpy()

                out_per_var = int(getattr(model.mkb, 'out_per_var', 1))
                in_ch = int(getattr(model.mkb, 'in_ch', mkb_feat_np.shape[0]))
                kernel_sizes = getattr(model.mkb, 'kernel_sizes', None)

                def _mkb_slice(var_idx):
                    # Returns: [out_per_var, H, W] corresponding to this variable across kernels/features
                    var_idx = int(var_idx)
                    if kernel_sizes is None:
                        # Old depth_conv layout: [var0_f1..var0_fK, var1_f1..var1_fK, ...]
                        start = var_idx * out_per_var
                        end = start + out_per_var
                        start = max(0, min(start, mkb_feat_np.shape[0]))
                        end = max(0, min(end, mkb_feat_np.shape[0]))
                        if end <= start:
                            return mkb_feat_np[0:1]
                        return mkb_feat_np[start:end]
                    # True multi-kernel layout: concat over kernels, each conv outputs [in_ch, H, W]
                    maps = []
                    for j in range(out_per_var):
                        ch = (j * in_ch) + var_idx
                        ch = max(0, min(int(ch), mkb_feat_np.shape[0] - 1))
                        maps.append(mkb_feat_np[ch])
                    return np.stack(maps, axis=0)

                bc_aod = inputs[i, -1, 0].detach().cpu().numpy()
                su_aod = inputs[i, -1, 1].detach().cpu().numpy()
                du_aod = inputs[i, -1, 2].detach().cpu().numpy()

                if mu_x is not None and sd_x is not None:
                    try:
                        bc_aod = (bc_aod * float(sd_x[0])) + float(mu_x[0])
                        su_aod = (su_aod * float(sd_x[1])) + float(mu_x[1])
                        du_aod = (du_aod * float(sd_x[2])) + float(mu_x[2])
                    except Exception:
                        pass

                vars_to_plot = [
                    ('BC', bc_aod, 0),
                    ('SU', su_aod, 1),
                    ('DU', du_aod, 2)
                ]
                ncols = int(1 + max(1, out_per_var))
                fig, axes = plt.subplots(3, ncols, figsize=(4.2 * ncols, 9.5), constrained_layout=True)

                for r, (vname, vraw, vidx) in enumerate(vars_to_plot):
                    feat_slice = _mkb_slice(vidx)
                    for c in range(ncols):
                        ax = axes[r, c]
                        if c == 0:
                            data = vraw
                            title = f"Input: {vname} AOD (raw)"
                            cmap = 'RdBu_r'
                        else:
                            k = c - 1
                            if k < feat_slice.shape[0]:
                                data = feat_slice[k]
                            else:
                                data = feat_slice[0]
                            if kernel_sizes is not None and k < len(kernel_sizes):
                                title = f"After MKB: {vname} k={int(kernel_sizes[k])}"
                            else:
                                title = f"After MKB: {vname} kernel {k+1}"
                            cmap = 'RdBu_r'

                        if shape_mask is not None:
                            data = np.where(shape_mask, data, np.nan)

                        cmap_obj = plt.get_cmap(cmap)
                        try:
                            cmap_obj = cmap_obj.copy()
                        except Exception:
                            pass
                        bg = (1.0, 1.0, 1.0, 1.0)
                        try:
                            cmap_obj.set_bad(color=bg, alpha=1.0)
                        except Exception:
                            pass

                        try:
                            ax.set_facecolor(bg)
                        except Exception:
                            pass
                        ax.grid(True, which='both', color='0.85', linewidth=0.6, alpha=0.8)

                        finite = np.isfinite(data)
                        if finite.any():
                            vmin = float(np.nanmin(data))
                            vmax = float(np.nanmax(data))
                        else:
                            vmin, vmax = 0.0, 1.0
                        if vmin == vmax:
                            vmin -= 1e-6
                            vmax += 1e-6
                        levels = np.linspace(vmin, vmax, 15)

                        ax.contourf(Lon, Lat, data, levels=levels, cmap=cmap_obj, extend='both', alpha=0.55)
                        ax.scatter(
                            Lon[finite], Lat[finite],
                            c=data[finite],
                            cmap=cmap_obj,
                            vmin=vmin, vmax=vmax,
                            marker='^',
                            s=240,
                            linewidths=0.0,
                            zorder=2
                        )

                        if gdf is not None:
                            gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, alpha=0.95, zorder=3)

                        ax.set_title(title, fontsize=12, fontweight='bold')
                        ax.tick_params(labelsize=8)
                        ax.set_xlim(lon_min, lon_max)
                        ax.set_ylim(lat_min, lat_max)

                        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_obj)
                        sm.set_array([])
                        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
                        cbar.ax.tick_params(labelsize=8)

                plt.suptitle(f'MKB per-variable kernel features (Sample {samples_plotted + 1})', fontsize=16, y=1.02)
                out_path = out_dir / f'mkcnn_spatial_kernels_sample_{samples_plotted+1}.png'
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {out_path}")
                plt.close(fig)

                samples_plotted += 1


# 8. Main model pipeline execution function


def run_model_pipeline(data_root_path='/content/drive/MyDrive/Research Result/',
                       batch_size=8, num_epochs=100, lr=0.001,
                       lambda_huber=1.0, lambda_temp=1.5, lambda_grad=0.5,
                       sequence_length=5,
                       target_delay=1,
                       out_per_var=2,
                       hidden_dim=64,
                       patience=15,
                       seed=42,
                       include_t2m_input=False,
                       lambda_ssim=0.0,
                       ssim_window_size=7,
                       time_weight_gamma=0.0,
                       predict_delta=False,
                       use_mse_magnitude=False,
                       huber_delta=2.0,
                       ssim_all_timesteps=False,
                       lr_schedule='plateau',
                       use_se=False,
                       bottleneck_1x1=False,
                       shuffle_train=False,
                       noise_std=0.0,
                       mkb_kernel_sizes=None,
                       aerosol_mask=None,
                       shapefile_path="NWH_States_Shapefile-20260120T110552Z-3-001/NWH_States_Shapefile/NWH_states.shp",
                       mask_to_shape=True,
                       test_eval_every=5,
                       output_dir='output_plots'):
    """
    Run the complete training and evaluation pipeline with PhysicsInformedLoss.
    """
    print("="*70)
    print("MKCNN-UNet-LSTM MODEL TRAINING PIPELINE")
    print("Using Physics-Informed Loss Function")
    print(f"Sequence Length: {sequence_length}")
    print("="*70)

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup data loaders
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        full_dataset = GridAerosolDataset(
            data_root_path,
            sequence_length=sequence_length,
            target_delay=target_delay,
            include_t2m_input=include_t2m_input,
            predict_delta=predict_delta
        )
        total_size = len(full_dataset)
        train_count = int(total_size * 0.8)
        val_count = max(1, int(total_size * 0.1))
        indices = list(range(total_size))
        train_indices = indices[:train_count]
        val_indices = indices[train_count : train_count + val_count]
        test_indices = indices[train_count + val_count:]
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)

        print(f"Total samples: {total_size}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"\nERROR: Failed to set up data loaders: {e}")
        print("Please check your `data_root_path` and ensure CSV files are present.")
        return None

    # Initialize model
    base_ds = train_dl.dataset.dataset if isinstance(train_dl.dataset, Subset) else train_dl.dataset
    print(f"sd_y: {float(base_ds.sd_y.mean()):.4f}")
    print(f"mu_y: {float(base_ds.mu_y.mean()):.4f}")

    if mkb_kernel_sizes is not None and len(mkb_kernel_sizes) > 0:
        out_per_var = int(len(mkb_kernel_sizes))

    model = CNN_UNet_LSTM(
        in_ch=base_ds.in_ch,
        out_per_var=out_per_var,
        hidden_dim=hidden_dim,
        use_se=use_se,
        bottleneck_1x1=bottleneck_1x1,
        mkb_kernel_sizes=mkb_kernel_sizes
    )

    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")

    # Train model
    trained_model, history = train_model(
        model, train_dl, val_dl, device, num_epochs, lr,
        patience=patience,
        lambda_huber=lambda_huber, lambda_temp=lambda_temp, lambda_grad=lambda_grad,
        lambda_ssim=lambda_ssim, ssim_window_size=ssim_window_size, time_weight_gamma=time_weight_gamma,
        use_mse_magnitude=use_mse_magnitude,
        huber_delta=huber_delta,
        ssim_all_timesteps=ssim_all_timesteps,
        lr_schedule=lr_schedule,
        noise_std=noise_std,
        test_loader=test_dl,
        test_eval_every=test_eval_every,
        output_dir=output_dir,
        aerosol_mask=aerosol_mask,
        checkpoint_name=f"best_model_h{int(target_delay)}_seq{int(sequence_length)}.pth",
        track_val_metrics_each_epoch=True
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)



    # Metrics are computed in normalized space. For real-world metrics, unnormalize using dataset stats.
    # Retrieve stats from underlying dataset.
    base_ds = train_dl.dataset.dataset if isinstance(train_dl.dataset, Subset) else train_dl.dataset
    mu_y = torch.from_numpy(base_ds.mu_y).float()
    sd_y = torch.from_numpy(base_ds.sd_y).float()

    trained_model.eval()
    all_predictions, all_targets = [], []
    all_predictions_seq, all_targets_seq = [], []
    all_persistence = []
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            pred_seq, _ = trained_model(inputs)

            pred_seq_cpu = pred_seq.detach().cpu()
            targ_seq_cpu = targets.detach().cpu()

            pred_last = pred_seq_cpu[:, -1]
            target_last = targ_seq_cpu[:, -1]

            # Persistence baseline: predict next = last observed T2m (only available if include_t2m_input)
            if include_t2m_input:
                # t2m input channel index = 3, already normalized with mu_y/sd_y
                persist_norm = inputs[:, -1, 3:4].cpu()
                all_persistence.append(persist_norm)

            # If predicting deltas, reconstruct absolute T using persistence + predicted delta
            if predict_delta:
                if not include_t2m_input:
                    raise ValueError('predict_delta=True requires include_t2m_input=True to reconstruct absolute temperature')
                pred_last = persist_norm + pred_last
                target_last = persist_norm + target_last

                # Apply the same reconstruction for the whole predicted horizon sequence
                pred_seq_cpu = pred_seq_cpu + persist_norm.unsqueeze(1)
                targ_seq_cpu = targ_seq_cpu + persist_norm.unsqueeze(1)

            all_predictions.append(pred_last)
            all_targets.append(target_last)
            all_predictions_seq.append(pred_seq_cpu)
            all_targets_seq.append(targ_seq_cpu)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if len(all_predictions_seq) > 0:
        all_predictions_seq = torch.cat(all_predictions_seq, dim=0)
        all_targets_seq = torch.cat(all_targets_seq, dim=0)
    else:
        all_predictions_seq, all_targets_seq = None, None

    if include_t2m_input and len(all_persistence) > 0:
        all_persistence = torch.cat(all_persistence, dim=0)
    else:
        all_persistence = None

    # Metrics in normalized space (useful when targeting thresholds like MSE < 0.3)
    results_norm = {
        'MAE_norm': calculate_mae(all_targets, all_predictions),
        'MSE_norm': calculate_mse(all_targets, all_predictions),
        'RMSE_norm': calculate_rmse(all_targets, all_predictions),
        'SSIM_norm': calculate_ssim(all_targets, all_predictions),
    }

    if all_targets_seq is not None and all_predictions_seq is not None:
        per_h_norm = calculate_metrics_per_horizon(all_targets_seq, all_predictions_seq)
        results_norm.update({
            'MSE_by_horizon_norm': per_h_norm.get('MSE', []),
            'RMSE_by_horizon_norm': per_h_norm.get('RMSE', []),
            'SSIM_by_horizon_norm': per_h_norm.get('SSIM', []),
        })
        plot_metrics_vs_horizon(per_h_norm, output_dir=output_dir, prefix='norm')

    if include_t2m_input and all_persistence is not None:
        results_norm.update({
            'Persistence_MAE_norm': calculate_mae(all_targets, all_persistence),
            'Persistence_MSE_norm': calculate_mse(all_targets, all_persistence),
            'Persistence_RMSE_norm': calculate_rmse(all_targets, all_persistence),
            'Persistence_SSIM_norm': calculate_ssim(all_targets, all_persistence),
        })

    # Unnormalize for final metrics
    all_predictions = (all_predictions * sd_y) + mu_y
    all_targets = (all_targets * sd_y) + mu_y

    if all_predictions_seq is not None and all_targets_seq is not None:
        all_predictions_seq = (all_predictions_seq * sd_y.view(1, 1, -1, 1, 1)) + mu_y.view(1, 1, -1, 1, 1)
        all_targets_seq = (all_targets_seq * sd_y.view(1, 1, -1, 1, 1)) + mu_y.view(1, 1, -1, 1, 1)

    if include_t2m_input and all_persistence is not None:
        all_persistence = (all_persistence * sd_y) + mu_y

    results = {
        'MAE': calculate_mae(all_targets, all_predictions),
        'MSE': calculate_mse(all_targets, all_predictions),
        'RMSE': calculate_rmse(all_targets, all_predictions),
        'MAPE': calculate_mape(all_targets, all_predictions),
        'SSIM': calculate_ssim(all_targets, all_predictions),
    }

    if all_targets_seq is not None and all_predictions_seq is not None:
        per_h = calculate_metrics_per_horizon(all_targets_seq, all_predictions_seq)
        results.update({
            'MSE_by_horizon': per_h.get('MSE', []),
            'RMSE_by_horizon': per_h.get('RMSE', []),
            'SSIM_by_horizon': per_h.get('SSIM', []),
        })
        plot_metrics_vs_horizon(per_h, output_dir=output_dir, prefix='')

    if include_t2m_input and all_persistence is not None:
        results.update({
            'Persistence_MAE': calculate_mae(all_targets, all_persistence),
            'Persistence_MSE': calculate_mse(all_targets, all_persistence),
            'Persistence_RMSE': calculate_rmse(all_targets, all_persistence),
            'Persistence_SSIM': calculate_ssim(all_targets, all_persistence),
        })

    # Merge normalized metrics after unnormalized metrics
    results.update(results_norm)

    print("\nTest Set Results:")
    print("-"*70)
    for metric, value in results.items():
        print(f"{metric:<25}: {value:.4f}")
    print("-"*70)


    # Visualize predictions
    plot_predictions(
        trained_model,
        test_dl,
        device,
        num_samples_to_plot=3,
        shapefile_path=shapefile_path,
        mask_to_shape=mask_to_shape,
        mu_y=float(base_ds.mu_y.mean()),
        sd_y=float(base_ds.sd_y.mean()),
        mu_x=np.array(base_ds.mu).reshape(-1),
        sd_x=np.array(base_ds.sd).reshape(-1),
        output_dir=output_dir
    )

    return trained_model, history, results


def run_multi_horizon_pipeline(data_root_path='/content/drive/MyDrive/Research Result/',
                              horizons=(2, 3, 4, 5, 6),
                              batch_size=8, num_epochs=100, lr=0.001,
                              lambda_huber=1.0, lambda_temp=1.5, lambda_grad=0.5,
                              sequence_length=5,
                              out_per_var=2,
                              hidden_dim=64,
                              patience=15,
                              seed=42,
                              include_t2m_input=False,
                              lambda_ssim=0.0,
                              ssim_window_size=7,
                              time_weight_gamma=0.0,
                              predict_delta=False,
                              use_mse_magnitude=False,
                              huber_delta=2.0,
                              ssim_all_timesteps=False,
                              lr_schedule='plateau',
                              use_se=False,
                              bottleneck_1x1=False,
                              shuffle_train=False,
                              noise_std=0.0,
                              mkb_kernel_sizes=None,
                              shapefile_path="NWH_States_Shapefile-20260120T110552Z-3-001/NWH_States_Shapefile/NWH_states.shp",
                              mask_to_shape=True,
                              test_eval_every=5,
                              output_dir='output_plots'):
    histories_by_h = {}

    for h in horizons:
        h_int = int(h)
        h_out_dir = str(Path(output_dir) / f"h{h_int}_seq{int(sequence_length)}")
        out = run_model_pipeline(
            data_root_path=data_root_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lambda_huber=lambda_huber,
            lambda_temp=lambda_temp,
            lambda_grad=lambda_grad,
            sequence_length=sequence_length,
            target_delay=h_int,
            out_per_var=out_per_var,
            hidden_dim=hidden_dim,
            patience=patience,
            seed=seed,
            include_t2m_input=include_t2m_input,
            lambda_ssim=lambda_ssim,
            ssim_window_size=ssim_window_size,
            time_weight_gamma=time_weight_gamma,
            predict_delta=predict_delta,
            use_mse_magnitude=use_mse_magnitude,
            huber_delta=huber_delta,
            ssim_all_timesteps=ssim_all_timesteps,
            lr_schedule=lr_schedule,
            use_se=use_se,
            bottleneck_1x1=bottleneck_1x1,
            shuffle_train=shuffle_train,
            noise_std=noise_std,
            mkb_kernel_sizes=mkb_kernel_sizes,
            shapefile_path=shapefile_path,
            mask_to_shape=mask_to_shape,
            test_eval_every=test_eval_every,
            output_dir=h_out_dir
        )
        if out is None:
            raise RuntimeError(f"run_model_pipeline returned None for horizon={h_int}")
        _, hist, _ = out
        histories_by_h[h_int] = hist

    plot_metric_vs_epochs_multi_horizon(
        histories_by_h,
        metric_key='val_rmse_metric',
        output_dir=output_dir,
        filename=f"rmse_vs_epochs_multi_horizon_seq{int(sequence_length)}.png",
        title=f"RMSE vs Epochs (Seq={int(sequence_length)})",
        ylabel='RMSE'
    )
    plot_metric_vs_epochs_multi_horizon(
        histories_by_h,
        metric_key='val_mse_metric',
        output_dir=output_dir,
        filename=f"mse_vs_epochs_multi_horizon_seq{int(sequence_length)}.png",
        title=f"MSE vs Epochs (Seq={int(sequence_length)})",
        ylabel='MSE'
    )
    plot_metric_vs_epochs_multi_horizon(
        histories_by_h,
        metric_key='val_ssim_metric',
        output_dir=output_dir,
        filename=f"ssim_vs_epochs_multi_horizon_seq{int(sequence_length)}.png",
        title=f"SSIM vs Epochs (Seq={int(sequence_length)})",
        ylabel='SSIM'
    )

    return histories_by_h


def run_input_ablation(data_root_path,
                       include_t2m_input=True,
                       mkb_kernel_sizes=None,
                       batch_size=8,
                       num_epochs=40,
                       sequence_length=6,
                       target_delay=1,
                       test_eval_every=5,
                       output_dir='output_plots',
                       seed=42,
                       hidden_dim=96,
                       lr=5e-4,
                       lambda_huber=1.0,
                       lambda_temp=0.7,
                       lambda_grad=0.3,
                       use_se=False,
                       bottleneck_1x1=False,
                       lr_schedule='cosine'):
    combos = [
        ('NO_AEROSOL', [0, 0, 0]),
        ('BC', [1, 0, 0]),
        ('SU', [0, 1, 0]),
        ('DU', [0, 0, 1]),
        ('BC+SU', [1, 1, 0]),
        ('BC+DU', [1, 0, 1]),
        ('SU+DU', [0, 1, 1]),
        ('BC+SU+DU', [1, 1, 1]),
    ]

    out_root = Path(output_dir) / 'ablation'
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    histories = {}
    for name, mask in combos:
        run_out = out_root / name.replace('+', '_')
        run_out.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 70)
        print(f"ABLATION: {name} (mask={mask})")
        print("=" * 70)
        trained_model, history, results = run_model_pipeline(
            data_root_path=data_root_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            lambda_huber=lambda_huber,
            lambda_temp=lambda_temp,
            lambda_grad=lambda_grad,
            sequence_length=sequence_length,
            target_delay=target_delay,
            out_per_var=3,
            hidden_dim=hidden_dim,
            patience=15,
            seed=seed,
            include_t2m_input=include_t2m_input,
            lambda_ssim=0.1,
            ssim_window_size=7,
            time_weight_gamma=1.5,
            predict_delta=False,
            use_mse_magnitude=False,
            huber_delta=0.5,
            ssim_all_timesteps=True,
            lr_schedule=lr_schedule,
            use_se=use_se,
            bottleneck_1x1=bottleneck_1x1,
            shuffle_train=False,
            noise_std=0.0,
            mkb_kernel_sizes=mkb_kernel_sizes,
            aerosol_mask=mask,
            shapefile_path="NWH_States_Shapefile-20260120T110552Z-3-001/NWH_States_Shapefile/NWH_states.shp",
            mask_to_shape=True,
            test_eval_every=test_eval_every,
            output_dir=str(run_out)
        )

        histories[name] = history
        epochs = history.get('test_eval_epochs', [])
        mses = history.get('test_mse', [])
        rmses = history.get('test_rmse', [])
        ssims = history.get('test_ssim', [])
        for e, mse, rmse, ssim in zip(epochs, mses, rmses, ssims):
            all_rows.append({
                'combo': name,
                'epoch': int(e),
                'mse': float(mse),
                'rmse': float(rmse),
                'ssim': float(ssim),
            })

    csv_path = out_root / 'ablation_timeseries.csv'
    if len(all_rows) == 0:
        print(
            f"Warning: no ablation time-series rows were recorded; "
            f"not overwriting {csv_path}. "
            f"(Tip: set --num_epochs >= --test_eval_every)"
        )
        return histories

    try:
        import csv
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['combo', 'epoch', 'mse', 'rmse', 'ssim'])
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"Saved: {csv_path}")
    except Exception as e:
        print(f"Warning: could not save ablation CSV ({e})")

    def _plot_metric(metric_key, ylabel, out_name):
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 4.2))
        for name, _ in combos:
            if name == 'NO_AEROSOL':
                continue
            h = histories.get(name, {})
            xs = h.get('test_eval_epochs', [])
            ys = h.get(metric_key, [])
            if xs is None or ys is None or len(xs) == 0:
                continue
            ax.plot(xs, ys, marker='o', linewidth=0.9, markersize=3.0, label=name)

        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f'{ylabel} vs Epoch (Ablation)', fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.22)
        ax.legend(
            fontsize=6,
            ncols=2,
            frameon=True,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.8
        )
        ax.tick_params(axis='both', labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.55)
        ax.tick_params(axis='x', width=0.55, length=3)
        out_path = out_root / out_name
        plt.tight_layout(pad=0.6)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    _plot_metric('test_mse', 'MSE', 'ablation_mse_timeseries.png')
    _plot_metric('test_rmse', 'RMSE', 'ablation_rmse_timeseries.png')
    _plot_metric('test_ssim', 'SSIM', 'ablation_ssim_timeseries.png')

    return histories


def plot_ablation_timeseries_from_csv(output_dir='output_plots'):
    out_root = Path(output_dir) / 'ablation'
    csv_path = out_root / 'ablation_timeseries.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Ablation CSV not found: {csv_path}")

    import csv
    data = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            combo = str(row.get('combo', '')).strip()
            if combo == '' or combo == 'NO_AEROSOL':
                continue
            try:
                e = int(float(row.get('epoch', '0')))
                mse = float(row.get('mse', 'nan'))
                rmse = float(row.get('rmse', 'nan'))
                ssim = float(row.get('ssim', 'nan'))
            except Exception:
                continue
            data.setdefault(combo, {'epoch': [], 'mse': [], 'rmse': [], 'ssim': []})
            data[combo]['epoch'].append(e)
            data[combo]['mse'].append(mse)
            data[combo]['rmse'].append(rmse)
            data[combo]['ssim'].append(ssim)

    if len(data) == 0:
        raise RuntimeError(f"No ablation rows found in {csv_path}")

    def _plot(metric, ylabel, out_name):
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 4.2))
        for combo in sorted(data.keys()):
            xs = data[combo]['epoch']
            ys = data[combo][metric]
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, marker='o', linewidth=0.9, markersize=3.0, label=combo)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f'{ylabel} vs Epoch (Ablation)', fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.55, alpha=0.22)
        ax.legend(
            fontsize=6,
            ncols=2,
            frameon=True,
            framealpha=0.9,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=0.8
        )
        ax.tick_params(axis='both', labelsize=7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.55)
        ax.tick_params(axis='x', width=0.55, length=3)
        out_path = out_root / out_name
        plt.tight_layout(pad=0.6)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    _plot('mse', 'MSE', 'ablation_mse_timeseries.png')
    _plot('rmse', 'RMSE', 'ablation_rmse_timeseries.png')
    _plot('ssim', 'SSIM', 'ablation_ssim_timeseries.png')



def run_plot_only(data_root_path,
                  batch_size=8,
                  sequence_length=6,
                  target_delay=1,
                  out_per_var=2,
                  hidden_dim=64,
                  seed=42,
                  include_t2m_input=True,
                  use_se=False,
                  bottleneck_1x1=False,
                  checkpoint_path='best_mkcnn_unet_lstm_physics.pth',
                  load_strict=False,
                  infer_from_checkpoint=True,
                  mkb_kernel_sizes=None,
                  shapefile_path="NWH_States_Shapefile-20260120T110552Z-3-001/NWH_States_Shapefile/NWH_states.shp",
                  mask_to_shape=True,
                  output_dir='output_plots'):
    """Load a trained checkpoint and only run plotting/evaluation without retraining."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    loaded = torch.load(str(ckpt_path), map_location='cpu')
    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
        state = loaded['model_state_dict']
    elif isinstance(loaded, dict) and 'state_dict' in loaded:
        state = loaded['state_dict']
    else:
        state = loaded

    full_dataset = GridAerosolDataset(
        data_root_path,
        sequence_length=sequence_length,
        target_delay=target_delay,
        include_t2m_input=include_t2m_input,
        predict_delta=False
    )
    total_size = len(full_dataset)
    train_count = int(total_size * 0.8)
    val_count = max(1, int(total_size * 0.1))
    indices = list(range(total_size))
    train_indices = indices[:train_count]
    val_indices = indices[train_count: train_count + val_count]
    test_indices = indices[train_count + val_count:]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    base_ds = train_dl.dataset.dataset if isinstance(train_dl.dataset, Subset) else train_dl.dataset
    print(f"sd_y: {float(base_ds.sd_y.mean()):.4f}")
    print(f"mu_y: {float(base_ds.mu_y.mean()):.4f}")

    if infer_from_checkpoint:
        try:
            if isinstance(state, dict):
                if 'mkb.depth_conv.weight' in state:
                    mkb_out = int(state['mkb.depth_conv.weight'].shape[0])
                    inferred_out_per_var = int(round(mkb_out / float(base_ds.in_ch)))
                    if inferred_out_per_var != out_per_var:
                        print(f"Overriding out_per_var from checkpoint: {out_per_var} -> {inferred_out_per_var}")
                    out_per_var = inferred_out_per_var
                elif any(k.startswith('mkb.convs.') and k.endswith('.weight') for k in state.keys()):
                    # Infer multi-kernel sizes from the conv weights
                    conv_keys = [k for k in state.keys() if k.startswith('mkb.convs.') and k.endswith('.weight')]
                    conv_indices = sorted({int(k.split('.')[2]) for k in conv_keys if k.split('.')[2].isdigit()})
                    inferred_ks = []
                    for idx in conv_indices:
                        w_key = f"mkb.convs.{idx}.weight"
                        if w_key in state and isinstance(state[w_key], torch.Tensor) and state[w_key].ndim >= 4:
                            inferred_ks.append(int(state[w_key].shape[2]))
                    if len(inferred_ks) > 0:
                        if mkb_kernel_sizes is None:
                            mkb_kernel_sizes = inferred_ks
                            print(f"Inferred mkb_kernel_sizes from checkpoint: {mkb_kernel_sizes}")
                        out_per_var = int(len(mkb_kernel_sizes))

            if isinstance(state, dict) and 'lstm.weight_ih_l0' in state:
                inferred_hidden_dim = int(state['lstm.weight_ih_l0'].shape[0] // 4)
                if inferred_hidden_dim != hidden_dim:
                    print(f"Overriding hidden_dim from checkpoint: {hidden_dim} -> {inferred_hidden_dim}")
                hidden_dim = inferred_hidden_dim

            inferred_use_se = isinstance(state, dict) and any('.se.fc.' in k for k in state.keys())
            if inferred_use_se != use_se:
                print(f"Overriding use_se from checkpoint: {use_se} -> {inferred_use_se}")
            use_se = inferred_use_se

            inferred_bottleneck_1x1 = isinstance(state, dict) and any('unet.bott.0.weight' in k for k in state.keys())
            if inferred_bottleneck_1x1 != bottleneck_1x1:
                print(f"Overriding bottleneck_1x1 from checkpoint: {bottleneck_1x1} -> {inferred_bottleneck_1x1}")
            bottleneck_1x1 = inferred_bottleneck_1x1
        except Exception as e:
            print(f"Warning: could not infer hyperparameters from checkpoint: {e}")

    if mkb_kernel_sizes is not None and len(mkb_kernel_sizes) > 0:
        out_per_var = int(len(mkb_kernel_sizes))

    model = CNN_UNet_LSTM(
        in_ch=base_ds.in_ch,
        out_per_var=out_per_var,
        hidden_dim=hidden_dim,
        use_se=use_se,
        bottleneck_1x1=bottleneck_1x1,
        mkb_kernel_sizes=mkb_kernel_sizes
    ).to(device)

    state_device = state
    if isinstance(state, dict):
        state_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state_device, strict=load_strict)
    if len(missing) > 0:
        print(f"Warning: missing keys when loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if len(unexpected) > 0:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    print("\nLoaded checkpoint successfully. Generating plots...")

    mu_y = torch.from_numpy(base_ds.mu_y).float()
    sd_y = torch.from_numpy(base_ds.sd_y).float()

    model.eval()
    all_predictions_seq, all_targets_seq = [], []
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            pred_seq, _ = model(inputs)
            all_predictions_seq.append(pred_seq.detach().cpu())
            all_targets_seq.append(targets.detach().cpu())

    if len(all_predictions_seq) > 0:
        all_predictions_seq = torch.cat(all_predictions_seq, dim=0)
        all_targets_seq = torch.cat(all_targets_seq, dim=0)

        per_h_norm = calculate_metrics_per_horizon(all_targets_seq, all_predictions_seq)
        plot_metrics_vs_horizon(per_h_norm, output_dir=output_dir, prefix='norm')

        all_predictions_seq = (all_predictions_seq * sd_y.view(1, 1, -1, 1, 1)) + mu_y.view(1, 1, -1, 1, 1)
        all_targets_seq = (all_targets_seq * sd_y.view(1, 1, -1, 1, 1)) + mu_y.view(1, 1, -1, 1, 1)
        per_h = calculate_metrics_per_horizon(all_targets_seq, all_predictions_seq)
        plot_metrics_vs_horizon(per_h, output_dir=output_dir, prefix='')

    plot_predictions(
        model,
        test_dl,
        device,
        num_samples_to_plot=3,
        shapefile_path=shapefile_path,
        mask_to_shape=mask_to_shape,
        mu_y=float(base_ds.mu_y.mean()),
        sd_y=float(base_ds.sd_y.mean()),
        mu_x=np.array(base_ds.mu).reshape(-1),
        sd_x=np.array(base_ds.sd).reshape(-1),
        output_dir=output_dir
    )

    plot_mkcnn_spatial_outputs(
        model,
        test_dl,
        device,
        num_samples_to_plot=3,
        shapefile_path=shapefile_path,
        mask_to_shape=mask_to_shape,
        output_dir=output_dir,
        mu_x=np.array(base_ds.mu).reshape(-1),
        sd_x=np.array(base_ds.sd).reshape(-1)
    )

    return model


def tune_hyperparams(data_root_path,
                     batch_size=8,
                     sequence_length=6,
                     seed=42):
    """Lightweight tuning on a few sensible configs; selects best by val loss."""
    configs = [
        {'out_per_var': 2, 'hidden_dim': 64, 'lr': 5e-4, 'lambda_temp': 0.7, 'lambda_grad': 0.3},
        {'out_per_var': 3, 'hidden_dim': 96, 'lr': 5e-4, 'lambda_temp': 0.5, 'lambda_grad': 0.2},
        {'out_per_var': 4, 'hidden_dim': 128, 'lr': 3e-4, 'lambda_temp': 0.4, 'lambda_grad': 0.15},
    ]

    best = None
    for i, cfg in enumerate(configs, start=1):
        print("\n" + "="*70)
        print(f"TUNING RUN {i}/{len(configs)}: {cfg}")
        print("="*70)
        model, history, results = run_model_pipeline(
            data_root_path=data_root_path,
            batch_size=batch_size,
            num_epochs=80,
            lr=cfg['lr'],
            lambda_huber=1.0,
            lambda_temp=cfg['lambda_temp'],
            lambda_grad=cfg['lambda_grad'],
            sequence_length=sequence_length,
            out_per_var=cfg['out_per_var'],
            hidden_dim=cfg['hidden_dim'],
            patience=12,
            seed=seed,
            include_t2m_input=True,
            lambda_ssim=0.1,
            ssim_window_size=7,
            time_weight_gamma=1.5,
            huber_delta=0.5,
            ssim_all_timesteps=True,
            lr_schedule='cosine',
            use_se=True,
            bottleneck_1x1=True
        )
        if history is None:
            continue

        score = history.get('best_val_loss', float('inf'))
        print(f"Config best_val_loss: {score:.6f}")

        if best is None or score < best['best_val_loss']:
            best = {
                'cfg': cfg,
                'best_val_loss': score,
                'results': results,
            }

    return best


# 9. Main Function

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default='/content/drive/MyDrive/Research Result/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lambda_huber', type=float, default=1.0)
    parser.add_argument('--lambda_temp', type=float, default=0.7)
    parser.add_argument('--lambda_grad', type=float, default=0.3)
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--target_delay', type=int, default=1)
    parser.add_argument('--out_per_var', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=96)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--include_t2m_input', action='store_true')
    parser.add_argument('--lambda_ssim', type=float, default=0.1)
    parser.add_argument('--ssim_window_size', type=int, default=7)
    parser.add_argument('--time_weight_gamma', type=float, default=1.5)
    parser.add_argument('--predict_delta', action='store_true')
    parser.add_argument('--use_mse_magnitude', action='store_true')
    parser.add_argument('--huber_delta', type=float, default=0.5)
    parser.add_argument('--ssim_all_timesteps', action='store_true')
    parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['plateau', 'cosine'])
    parser.add_argument('--use_se', action='store_true')
    parser.add_argument('--bottleneck_1x1', action='store_true')
    parser.add_argument('--shuffle_train', action='store_true')
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--mkb_kernel_sizes', type=str, default='')
    parser.add_argument('--shapefile_path', type=str, default="NWH_States_Shapefile-20260120T110552Z-3-001/NWH_States_Shapefile/NWH_states.shp")
    parser.add_argument('--mask_to_shape', action='store_true')
    parser.add_argument('--plot_only', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default='best_mkcnn_unet_lstm_physics.pth')
    parser.add_argument('--load_strict', action='store_true', default=False)
    parser.add_argument('--no_infer_checkpoint', action='store_true')
    parser.add_argument('--output_dir', type=str, default='output_plots')
    parser.add_argument('--test_eval_every', type=int, default=5)
    parser.add_argument('--multi_horizon', action='store_true')
    parser.add_argument('--horizons', type=str, default='2,3,4,5,6')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--bench_iters', type=int, default=50)
    parser.add_argument('--bench_warmup', type=int, default=10)
    parser.add_argument('--bench_batch_size', type=int, default=8)
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--ablation_plot_only', action='store_true')
    args = parser.parse_args()

    mkb_kernel_sizes = None
    if isinstance(args.mkb_kernel_sizes, str) and args.mkb_kernel_sizes.strip() != '':
        try:
            mkb_kernel_sizes = [int(s) for s in args.mkb_kernel_sizes.split(',') if s.strip() != '']
        except Exception:
            mkb_kernel_sizes = None

    if args.benchmark:
        tmp_ds = GridAerosolDataset(
            args.data_root_path,
            sequence_length=args.sequence_length,
            target_delay=args.target_delay,
            include_t2m_input=args.include_t2m_input,
            predict_delta=args.predict_delta
        )
        benchmark_mkcnn_vs_convlstm(
            in_ch=int(tmp_ds.in_ch),
            sequence_length=int(args.sequence_length),
            batch_size=int(args.bench_batch_size),
            hidden_dim=int(args.hidden_dim),
            out_per_var=int(args.out_per_var),
            use_se=bool(args.use_se),
            bottleneck_1x1=bool(args.bottleneck_1x1),
            iters=int(args.bench_iters),
            warmup=int(args.bench_warmup),
            output_dir=args.output_dir
        )
        raise SystemExit(0)

    if args.ablation_plot_only:
        plot_ablation_timeseries_from_csv(output_dir=args.output_dir)
        raise SystemExit(0)

    if args.tune:
        best = tune_hyperparams(
            data_root_path=args.data_root_path,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            seed=args.seed
        )
        print("\n" + "="*70)
        print("TUNING SUMMARY")
        print("="*70)
        print(best)

    if args.ablation:
        run_input_ablation(
            data_root_path=args.data_root_path,
            include_t2m_input=args.include_t2m_input,
            mkb_kernel_sizes=mkb_kernel_sizes,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            sequence_length=args.sequence_length,
            target_delay=args.target_delay,
            test_eval_every=args.test_eval_every,
            output_dir=args.output_dir,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            lambda_huber=args.lambda_huber,
            lambda_temp=args.lambda_temp,
            lambda_grad=args.lambda_grad,
            use_se=args.use_se,
            bottleneck_1x1=args.bottleneck_1x1,
            lr_schedule=args.lr_schedule
        )
        raise SystemExit(0)

    if args.plot_only:
        run_plot_only(
            data_root_path=args.data_root_path,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            target_delay=args.target_delay,
            out_per_var=args.out_per_var,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            include_t2m_input=args.include_t2m_input,
            use_se=args.use_se,
            bottleneck_1x1=args.bottleneck_1x1,
            checkpoint_path=args.checkpoint_path,
            load_strict=args.load_strict,
            infer_from_checkpoint=not args.no_infer_checkpoint,
            mkb_kernel_sizes=mkb_kernel_sizes,
            shapefile_path=args.shapefile_path,
            mask_to_shape=args.mask_to_shape,
            output_dir=args.output_dir
        )
    else:
        if args.multi_horizon:
            horizons = None
            if isinstance(args.horizons, str) and args.horizons.strip() != '':
                try:
                    horizons = [int(s) for s in args.horizons.split(',') if s.strip() != '']
                except Exception:
                    horizons = None
            if horizons is None or len(horizons) == 0:
                horizons = [2, 3, 4, 5, 6]
            run_multi_horizon_pipeline(
                data_root_path=args.data_root_path,
                horizons=tuple(horizons),
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                lambda_huber=args.lambda_huber,
                lambda_temp=args.lambda_temp,
                lambda_grad=args.lambda_grad,
                sequence_length=args.sequence_length,
                out_per_var=args.out_per_var,
                hidden_dim=args.hidden_dim,
                patience=args.patience,
                seed=args.seed,
                include_t2m_input=args.include_t2m_input,
                lambda_ssim=args.lambda_ssim,
                ssim_window_size=args.ssim_window_size,
                time_weight_gamma=args.time_weight_gamma,
                predict_delta=args.predict_delta,
                use_mse_magnitude=args.use_mse_magnitude,
                huber_delta=args.huber_delta,
                ssim_all_timesteps=args.ssim_all_timesteps,
                lr_schedule=args.lr_schedule,
                use_se=args.use_se,
                bottleneck_1x1=args.bottleneck_1x1,
                shuffle_train=args.shuffle_train,
                noise_std=args.noise_std,
                mkb_kernel_sizes=mkb_kernel_sizes,
                shapefile_path=args.shapefile_path,
                mask_to_shape=args.mask_to_shape,
                test_eval_every=args.test_eval_every,
                output_dir=args.output_dir
            )
        else:
            out = run_model_pipeline(
                data_root_path=args.data_root_path,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                lr=args.lr,
                lambda_huber=args.lambda_huber,
                lambda_temp=args.lambda_temp,
                lambda_grad=args.lambda_grad,
                sequence_length=args.sequence_length,
                target_delay=args.target_delay,
                out_per_var=args.out_per_var,
                hidden_dim=args.hidden_dim,
                patience=args.patience,
                seed=args.seed,
                include_t2m_input=args.include_t2m_input,
                lambda_ssim=args.lambda_ssim,
                ssim_window_size=args.ssim_window_size,
                time_weight_gamma=args.time_weight_gamma,
                predict_delta=args.predict_delta,
                use_mse_magnitude=args.use_mse_magnitude,
                huber_delta=args.huber_delta,
                ssim_all_timesteps=args.ssim_all_timesteps,
                lr_schedule=args.lr_schedule,
                use_se=args.use_se,
                bottleneck_1x1=args.bottleneck_1x1,
                shuffle_train=args.shuffle_train,
                noise_std=args.noise_std,
                mkb_kernel_sizes=mkb_kernel_sizes,
                shapefile_path=args.shapefile_path,
                mask_to_shape=args.mask_to_shape,
                test_eval_every=args.test_eval_every,
                output_dir=args.output_dir
            )

            if out is None:
                raise SystemExit(1)

            trained_model, history, results = out
