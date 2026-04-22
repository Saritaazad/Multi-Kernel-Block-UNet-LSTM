

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import time
import csv
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as sk_ssim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import geopandas as gpd
from shapely.geometry import box, Point
from shapely.prepared import prep

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# 0.  Experiment matrix
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = [
    # (label,          model_key,     extra_input, loss_fn)
    ("ConvLSTM_No_Huber",       "ConvLSTM",    False, "Huber"),
    ("ConvLSTM_Yes_Huber",      "ConvLSTM",    True,  "Huber"),
    ("ConvLSTM_Yes_Physics",    "ConvLSTM",    True,  "Physics"),
    ("DA-ConvLSTM_No_Huber",    "DA-ConvLSTM", False, "Huber"),
    ("DA-ConvLSTM_Yes_Huber",   "DA-ConvLSTM", True,  "Huber"),
    ("DA-ConvLSTM_Yes_Physics", "DA-ConvLSTM", True,  "Physics"),
    ("Ours_No_Huber",           "Ours",        False, "Huber"),
    ("Ours_Yes_Huber",          "Ours",        True,  "Huber"),
    ("Ours_Yes_Physics",        "Ours",        True,  "Physics"),
]

# Loss-weight presets
LOSS_PRESETS = {
    "Huber":   dict(lambda_huber=1.0, lambda_temp=0.0, lambda_grad=0.0, lambda_ssim=0.0),
    "Physics": dict(lambda_huber=1.0, lambda_temp=0.7, lambda_grad=0.3, lambda_ssim=0.1),
}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Dataset
# ──────────────────────────────────────────────────────────────────────────────

class GridAerosolDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, target_delay=1,
                 train_split_idx=418, include_t2m_input=False, predict_delta=False):
        self.sequence_length = sequence_length
        self.target_delay = target_delay
        self.include_t2m_input = include_t2m_input
        self.predict_delta = predict_delta
        self.vars = ['BC_AOD', 'SU_AOD', 'DU_AOD_pm25']

        root = Path(root_dir)
        data = {}
        for v in self.vars + ['T2m']:
            fpath = root / f'{v}_time_series.csv'
            if not fpath.exists():
                raise FileNotFoundError(f"Missing file: {fpath}")
            arr = pd.read_csv(fpath, header=0).values
            if arr.shape[0] != 361 and arr.shape[1] == 361:
                arr = arr.T
            if arr.shape[0] != 361:
                raise ValueError(f"Unexpected shape for {v}: {arr.shape}")
            data[v] = arr

        print("Generating Sine/Cosine Month Channels & Processing Frames...")
        all_X_frames, all_Y_frames = [], []
        total_time_steps = data['T2m'].shape[1]

        for t in range(2, total_time_steps):
            x_stack = [data[v][:, t].reshape(19, 19)[::-1, :] for v in self.vars]
            if self.include_t2m_input:
                x_stack.append(data['T2m'][:, t - 1].reshape(19, 19)[::-1, :])
            month_angle = 2 * np.pi * ((t % 12) / 12.0)
            x_stack.append(np.full((19, 19), np.sin(month_angle), dtype=np.float32))
            x_stack.append(np.full((19, 19), np.cos(month_angle), dtype=np.float32))
            all_X_frames.append(np.stack(x_stack, axis=0))
            temp = data['T2m'][:, t].reshape(19, 19)[::-1, :].reshape(1, 19, 19)
            all_Y_frames.append(temp)

        all_X_frames = np.stack(all_X_frames).astype(np.float32)
        all_Y_frames = np.stack(all_Y_frames).astype(np.float32)
        self.in_ch = all_X_frames.shape[1]

        train_stats_data = all_X_frames[:train_split_idx, 0:3, :, :]
        self.mu = train_stats_data.mean(axis=(0, 2, 3), keepdims=True)
        self.sd = train_stats_data.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        all_X_frames[:, 0:3, :, :] = (all_X_frames[:, 0:3, :, :] - self.mu) / self.sd

        train_y = all_Y_frames[:train_split_idx]
        self.mu_y = train_y.mean(axis=(0, 2, 3), keepdims=True)
        self.sd_y = train_y.std(axis=(0, 2, 3), keepdims=True) + 1e-6

        if self.predict_delta:
            delta = all_Y_frames.copy()
            delta[1:] = all_Y_frames[1:] - all_Y_frames[:-1]
            delta[0] = 0.0
            all_Y_frames = delta

        all_Y_frames = (all_Y_frames - self.mu_y) / self.sd_y

        if self.include_t2m_input:
            t2m_ch = 3
            all_X_frames[:, t2m_ch:t2m_ch+1, :, :] = \
                (all_X_frames[:, t2m_ch:t2m_ch+1, :, :] - self.mu_y) / self.sd_y

        self.X_data, self.Y_data = [], []
        num_total_frames = all_X_frames.shape[0]
        max_idx = num_total_frames - self.target_delay

        for i in range(self.sequence_length, max_idx):
            x_seq = all_X_frames[i - self.sequence_length: i]
            y_seq = all_Y_frames[i - self.sequence_length + self.target_delay: i + self.target_delay]
            if x_seq.shape[0] == self.sequence_length and y_seq.shape[0] == self.sequence_length:
                self.X_data.append(x_seq)
                self.Y_data.append(y_seq)

        self.X_data = np.stack(self.X_data).astype(np.float32)
        self.Y_data = np.stack(self.Y_data).astype(np.float32)
        print(f"Dataset: Input {self.X_data.shape}, Target {self.Y_data.shape}")

    def __len__(self):  return len(self.X_data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X_data[idx]), torch.from_numpy(self.Y_data[idx])


def make_loaders(data_root, seq_len, target_delay, include_t2m, batch_size, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ds = GridAerosolDataset(data_root, sequence_length=seq_len,
                            target_delay=target_delay, include_t2m_input=include_t2m)
    n = len(ds)
    tr = int(n * 0.8); va = max(1, int(n * 0.1))
    idx = list(range(n))
    tr_dl = DataLoader(Subset(ds, idx[:tr]),            batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(Subset(ds, idx[tr:tr+va]),       batch_size=batch_size, shuffle=False)
    te_dl = DataLoader(Subset(ds, idx[tr+va:]),         batch_size=batch_size, shuffle=False)
    print(f"  Train={tr}  Val={va}  Test={n-tr-va}")
    return tr_dl, va_dl, te_dl, ds


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Models  (ConvLSTM / DA-ConvLSTM / MKCNN-UNet-LSTM)
# ──────────────────────────────────────────────────────────────────────────────

# ── ConvLSTM ──────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size // 2)
    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.hidden_dim = hidden_dim
    def forward(self, x):
        b, t, c, h, w = x.shape
        ht = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
        ct = torch.zeros_like(ht)
        outs = []
        for k in range(t):
            ht, ct = self.cell(x[:, k], (ht, ct))
            outs.append(ht)
        return torch.stack(outs, 1), (ht, ct)

class ConvLSTMForecast(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, kernel_size=3, out_ch=1):
        super().__init__()
        self.convlstm = ConvLSTM(in_ch, hidden_dim, kernel_size)
        self.head = nn.Conv2d(hidden_dim, out_ch, 1)
    def forward(self, x):
        feats, state = self.convlstm(x)
        b, t, c, h, w = feats.shape
        y = self.head(feats.view(b*t, c, h, w)).view(b, t, -1, h, w)
        return y, state

# ── DA-ConvLSTM ───────────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.q = nn.Conv2d(in_dim, max(1, in_dim // 8), 1)
        self.k = nn.Conv2d(in_dim, max(1, in_dim // 8), 1)
        self.v = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b, c, h, w = x.size()
        q = self.q(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.k(x).view(b, -1, h*w)
        att = F.softmax(torch.bmm(q, k), -1)
        v = self.v(x).view(b, -1, h*w)
        out = torch.bmm(v, att.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x

class ChannelAttention(nn.Module):
    def __init__(self, in_dim, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_dim, max(1, in_dim//r), 1),
                                 nn.ReLU(), nn.Conv2d(max(1, in_dim//r), in_dim, 1))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))

class DAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size // 2)
        self.sa = SpatialAttention(hidden_dim)
        self.ca = ChannelAttention(hidden_dim)
    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        h_next = self.sa(h_next)
        h_next = self.ca(h_next)
        return h_next, c_next

class DAConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = DAConvLSTMCell(input_dim, hidden_dim, kernel_size)
    def forward(self, x):
        b, t, _, h, w = x.size()
        ht = torch.zeros(b, self.hidden_dim, h, w, device=x.device, dtype=x.dtype)
        ct = torch.zeros_like(ht)
        outs = []
        for k in range(t):
            ht, ct = self.cell(x[:, k], (ht, ct))
            outs.append(ht)
        return torch.stack(outs, 1), (ht, ct)

class DAConvLSTMForecast(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, kernel_size=3, out_ch=1):
        super().__init__()
        self.convlstm = DAConvLSTM(in_ch, hidden_dim, kernel_size)
        self.head = nn.Conv2d(hidden_dim, out_ch, 1)
    def forward(self, x):
        feats, state = self.convlstm(x)
        b, t, c, h, w = feats.shape
        y = self.head(feats.view(b*t, c, h, w)).view(b, t, -1, h, w)
        return y, state

# ── MKCNN-UNet-LSTM ("Ours") ──────────────────────────────────────────────────
class DoubleConv(nn.Sequential):
    def __init__(self, ic, oc):
        super().__init__(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                         nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        h = max(1, ch // r)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(ch, h), nn.ReLU(True),
                                 nn.Linear(h, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).view(x.shape[0], -1, 1, 1)

class DoubleConvSE(nn.Module):
    def __init__(self, ic, oc, use_se=False):
        super().__init__()
        self.conv = DoubleConv(ic, oc)
        self.se = SEBlock(oc) if use_se else nn.Identity()
    def forward(self, x): return self.se(self.conv(x))

class UNetOdd(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=16, use_se=False, bottleneck_1x1=False):
        super().__init__()
        c1,c2,c3,c4,bt = base,base*2,base*4,base*8,base*16
        self.enc0,self.enc1 = DoubleConvSE(in_ch,c1,use_se),DoubleConvSE(c1,c2,use_se)
        self.enc2,self.enc3 = DoubleConvSE(c2,c3,use_se),DoubleConvSE(c3,c4,use_se)
        self.bott = (nn.Sequential(nn.Conv2d(c4,bt,1),nn.BatchNorm2d(bt),nn.ReLU(True),
                                   nn.Conv2d(bt,bt,1),nn.BatchNorm2d(bt),nn.ReLU(True))
                     if bottleneck_1x1 else DoubleConvSE(c4,bt,use_se))
        self.up3,self.dec3 = nn.Conv2d(bt,c4,1),DoubleConvSE(c4*2,c3,use_se)
        self.up2,self.dec2 = nn.Conv2d(c3,c3,1),DoubleConvSE(c3*2,c2,use_se)
        self.up1,self.dec1 = nn.Conv2d(c2,c2,1),DoubleConvSE(c2*2,c1,use_se)
        self.up0,self.dec0 = nn.Conv2d(c1,c1,1),DoubleConvSE(c1*2,c1,use_se)
        self.head = nn.Conv2d(c1,out_ch,1)
    @staticmethod
    def _r(x,s): return F.interpolate(x,(s,s),mode='bilinear',align_corners=False)
    def forward(self,x):
        e0=self.enc0(x); e1=self.enc1(self._r(e0,13))
        e2=self.enc2(self._r(e1,8)); e3=self.enc3(self._r(e2,3))
        b=self.bott(self._r(e3,1))
        d3=self.dec3(torch.cat([self._r(self.up3(b),3),e3],1))
        d2=self.dec2(torch.cat([self._r(self.up2(d3),8),e2],1))
        d1=self.dec1(torch.cat([self._r(self.up1(d2),13),e1],1))
        d0=self.dec0(torch.cat([self._r(self.up0(d1),19),e0],1))
        return self.head(d0)

class MultiKernelBlock(nn.Module):
    def __init__(self, in_ch=5, out_per_var=4, kernel_size=3, kernel_sizes=None):
        super().__init__()
        self.in_ch=int(in_ch); self.kernel_sizes=None
        if kernel_sizes is None:
            self.out_per_var=int(out_per_var)
            self.out_ch=self.in_ch*self.out_per_var
            self.depth_conv=nn.Conv2d(self.in_ch,self.out_ch,kernel_size,
                                       padding=kernel_size//2,groups=self.in_ch,bias=False)
            self.bn=nn.BatchNorm2d(self.out_ch)
        else:
            ks=[int(k) for k in kernel_sizes] or [int(kernel_size)]
            self.kernel_sizes=ks; self.out_per_var=len(ks); self.out_ch=self.in_ch*self.out_per_var
            self.convs=nn.ModuleList([nn.Conv2d(self.in_ch,self.in_ch,k,padding=k//2,
                                                 groups=self.in_ch,bias=False) for k in ks])
            self.bn=nn.BatchNorm2d(self.out_ch)
        self.act=nn.ReLU(True)
    def forward(self,x):
        y=(self.depth_conv(x) if hasattr(self,'depth_conv')
           else torch.cat([c(x) for c in self.convs],1))
        return self.act(self.bn(y))

class CNN_UNet_LSTM(nn.Module):
    def __init__(self, in_ch=5, out_per_var=4, hidden_dim=256,
                 use_se=False, bottleneck_1x1=False, mkb_kernel_sizes=None):
        super().__init__()
        self.mkb=MultiKernelBlock(in_ch,out_per_var,kernel_sizes=mkb_kernel_sizes)
        self.unet=UNetOdd(in_ch*self.mkb.out_per_var,out_ch=1,use_se=use_se,bottleneck_1x1=bottleneck_1x1)
        self.flatten=nn.Flatten(1)
        self.lstm=nn.LSTM(19*19,hidden_dim,num_layers=2,batch_first=True,dropout=0.4)
        self.fc=nn.Linear(hidden_dim,19*19)
    def forward(self,x):
        B,T,C,H,W=x.shape
        sp=[]
        for t in range(T):
            sp.append(self.flatten(self.unet(self.mkb(x[:,t]))))
        lstm_out,_=self.lstm(torch.stack(sp,1))
        y=self.fc(lstm_out.contiguous().view(B*T,-1)).view(B,T,1,H,W)
        return y, sp[-1].view(B,1,H,W)


def build_model(model_key, in_ch, hidden_dim):
    if model_key == "ConvLSTM":
        return ConvLSTMForecast(in_ch=in_ch, hidden_dim=hidden_dim)
    elif model_key == "DA-ConvLSTM":
        return DAConvLSTMForecast(in_ch=in_ch, hidden_dim=hidden_dim)
    else:  # Ours
        return CNN_UNet_LSTM(in_ch=in_ch, out_per_var=3, hidden_dim=hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Physics-informed loss
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_huber=1.0, lambda_temp=0.5,
                 lambda_grad=0.1, lambda_ssim=0.0, huber_delta=2.0):
        super().__init__()
        self.w1=lambda_huber; self.w2=lambda_temp
        self.w3=lambda_grad;  self.w4=lambda_ssim
        self.huber=nn.HuberLoss(delta=huber_delta)
        self.mse=nn.MSELoss(); self.l1=nn.L1Loss()

    def _grad(self, p, t):
        return (self.l1(torch.abs(p[:,:,1:,:]-p[:,:,:-1,:]),
                        torch.abs(t[:,:,1:,:]-t[:,:,:-1,:])) +
                self.l1(torch.abs(p[:,:,:,1:]-p[:,:,:,:-1]),
                        torch.abs(t[:,:,:,1:]-t[:,:,:,:-1])))

    def forward(self, pred, tgt):
        l_h = self.huber(pred, tgt)
        l_t = (self.mse(pred[:,1:]-pred[:,:-1], tgt[:,1:]-tgt[:,:-1])
               if pred.shape[1]>1 else torch.tensor(0., device=pred.device))
        b,t,c,h,w = pred.shape
        l_g = self._grad(pred.view(b*t,c,h,w), tgt.view(b*t,c,h,w))
        loss = self.w1*l_h + self.w2*l_t + self.w3*l_g
        return loss, l_h.item(), l_t.item(), l_g.item()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _np(t): return t.cpu().numpy().flatten()

def calc_mae(yt, yp):  return float(mean_absolute_error(_np(yt), _np(yp)))
def calc_mse(yt, yp):  return float(mean_squared_error(_np(yt), _np(yp)))
def calc_rmse(yt, yp): return float(np.sqrt(calc_mse(yt, yp)))

def calc_ssim(yt, yp, data_range=None):
    a, b = yt.cpu().numpy(), yp.cpu().numpy()
    dr = float(a.max()-a.min()) if data_range is None else float(data_range)
    dr = dr if dr > 0 else 1.0
    scores = []
    for i in range(a.shape[0]):
        for c in range(a.shape[1]):
            try: scores.append(sk_ssim(a[i,c], b[i,c], data_range=dr, win_size=3))
            except: scores.append(0.0)
    return float(np.mean(scores))

def eval_last_frame(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            ps, _ = model(xb)
            preds.append(ps[:,-1].cpu()); tgts.append(yb[:,-1].cpu())
    p = torch.cat(preds); t = torch.cat(tgts)
    return dict(MAE=calc_mae(t,p), MSE=calc_mse(t,p),
                RMSE=calc_rmse(t,p), SSIM=calc_ssim(t,p))


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_config(model, tr_dl, va_dl, te_dl, device, cfg, args):
    """Train a single config; return history dict with per-epoch test metrics."""
    lw = LOSS_PRESETS[cfg["loss_fn"]]
    criterion = PhysicsInformedLoss(huber_delta=0.5, **lw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.num_epochs))

    history = dict(test_eval_epochs=[], test_mse=[], test_rmse=[],
                   test_mae=[], test_ssim=[],
                   train_losses=[], val_losses=[])
    best_val, patience_cnt = float('inf'), 0

    for epoch in range(1, args.num_epochs + 1):
        # ── train
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            ps, _ = model(xb)
            loss, *_ = criterion(ps, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        scheduler.step()
        history['train_losses'].append(tr_loss / len(tr_dl))

        # ── val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                ps, _ = model(xb)
                l, *_ = criterion(ps, yb)
                va_loss += l.item()
        avg_va = va_loss / len(va_dl)
        history['val_losses'].append(avg_va)

        # ── best / patience
        if avg_va < best_val:
            best_val = avg_va; patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"    Early stop at epoch {epoch}")
                break

        # ── test eval
        if epoch % args.test_eval_every == 0:
            m = eval_last_frame(model, te_dl, device)
            history['test_eval_epochs'].append(epoch)
            history['test_mse'].append(m['MSE'])
            history['test_rmse'].append(m['RMSE'])
            history['test_mae'].append(m['MAE'])
            history['test_ssim'].append(m['SSIM'])
            print(f"    Epoch {epoch:3d}: val={avg_va:.4f} | "
                  f"test MAE={m['MAE']:.4f} MSE={m['MSE']:.4f} "
                  f"RMSE={m['RMSE']:.4f} SSIM={m['SSIM']:.4f}")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Comparison plots  (one line per config, styled like the ablation image)
# ──────────────────────────────────────────────────────────────────────────────

# Color / marker scheme — consistent across all plots
_PALETTE = [
    ('#1f77b4', 'o'),  # ConvLSTM No Huber
    ('#aec7e8', 's'),  # ConvLSTM Yes Huber
    ('#ffbb78', '^'),  # ConvLSTM Yes Physics
    ('#d62728', 'o'),  # DA-ConvLSTM No Huber
    ('#ff9896', 's'),  # DA-ConvLSTM Yes Huber
    ('#9467bd', '^'),  # DA-ConvLSTM Yes Physics
    ('#2ca02c', 'o'),  # Ours No Huber
    ('#98df8a', 's'),  # Ours Yes Huber
    ('#ff7f0e', '^'),  # Ours Yes Physics
]

_LABELS = [
    "ConvLSTM  No  Huber",
    "ConvLSTM  Yes Huber",
    "ConvLSTM  Yes Physics",
    "DA-ConvLSTM  No  Huber",
    "DA-ConvLSTM  Yes Huber",
    "DA-ConvLSTM  Yes Physics",
    "Ours  No  Huber",
    "Ours  Yes Huber",
    "Ours  Yes Physics",
]


def _plot_metric_comparison(histories, metric_key, ylabel, title, out_path):
    """histories: list of dicts (one per config, same order as EXPERIMENT_CONFIGS)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (hist, (color, marker), label) in enumerate(
            zip(histories, _PALETTE, _LABELS)):
        xs = hist.get('test_eval_epochs', [])
        ys = hist.get(metric_key, [])
        if not xs: continue
        ax.plot(xs, ys, color=color, marker=marker, linewidth=1.6,
                markersize=5, label=label, alpha=0.92)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(fontsize=7.5, ncols=3, frameon=True, framealpha=0.9,
              loc='upper right', title='Config', title_fontsize=8)
    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_comparison_plots(all_histories, seq_len, out_dir):
    """Save MSE / RMSE / SSIM comparison plots for one sequence length."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in [('test_mse',  'MSE'),
                            ('test_rmse', 'RMSE'),
                            ('test_ssim', 'SSIM')]:
        _plot_metric_comparison(
            all_histories, metric, ylabel,
            title=f'{ylabel} vs Epoch — All Configs  (Seq={seq_len})',
            out_path=out_dir / f'{metric}_vs_epoch_seq{seq_len}.png'
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Results table
# ──────────────────────────────────────────────────────────────────────────────

def save_results_table(rows, out_path):
    """rows: list of dicts with at least Model / Extra_IP / Loss / MAE / MSE / RMSE / SSIM."""
    if not rows: return
    fieldnames = list(rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Saved results table: {out_path}")

    # Pretty-print
    df = pd.DataFrame(rows)
    print("\n" + "="*90)
    print(df.to_string(index=False))
    print("="*90)


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Main experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all_experiments(args, seq_len):
    """Run 9 configs for one sequence length; return summary rows + all histories."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"SEQUENCE LENGTH = {seq_len}")
    print(f"{'='*80}")

    summary_rows = []
    all_histories = []

    for exp_i, (label, model_key, extra_input, loss_fn) in enumerate(EXPERIMENT_CONFIGS):
        print(f"\n── Experiment {exp_i+1}/9 ─ {label} ──")

        # Build data loaders (re-created per config because extra_input differs)
        tr_dl, va_dl, te_dl, ds = make_loaders(
            args.data_root_path, seq_len, args.target_delay,
            extra_input, args.batch_size, args.seed
        )

        in_ch = ds.in_ch
        model = build_model(model_key, in_ch, args.hidden_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {model_key}  in_ch={in_ch}  params={n_params:,}")

        cfg = dict(loss_fn=loss_fn)
        try:
            history = train_one_config(model, tr_dl, va_dl, te_dl, device, cfg, args)
        except Exception as e:
            print(f"  ERROR: {e}")
            history = dict(test_eval_epochs=[], test_mse=[], test_rmse=[],
                           test_mae=[], test_ssim=[], train_losses=[], val_losses=[])

        all_histories.append(history)

        # Final test metrics (last recorded eval)
        final_mae  = history['test_mae'][-1]  if history['test_mae']  else float('nan')
        final_mse  = history['test_mse'][-1]  if history['test_mse']  else float('nan')
        final_rmse = history['test_rmse'][-1] if history['test_rmse'] else float('nan')
        final_ssim = history['test_ssim'][-1] if history['test_ssim'] else float('nan')

        row = {
            'Seq_Len':      seq_len,
            'Model':        model_key,
            'Extra_Input':  'Yes' if extra_input else 'No',
            'Loss':         loss_fn,
            'MAE':          round(final_mae,  6),
            'MSE':          round(final_mse,  6),
            'RMSE':         round(final_rmse, 6),
            'SSIM':         round(final_ssim, 6),
        }
        summary_rows.append(row)

    return summary_rows, all_histories


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root_path',  type=str, default='/content/drive/MyDrive/Research Result/')
    p.add_argument('--output_dir',      type=str, default='experiment_out')
    p.add_argument('--seq_lengths',     type=str, default='3,6',
                   help='Comma-separated sequence lengths, e.g. 3,6')
    p.add_argument('--batch_size',      type=int, default=8)
    p.add_argument('--num_epochs',      type=int, default=40)
    p.add_argument('--lr',              type=float, default=5e-4)
    p.add_argument('--hidden_dim',      type=int, default=96)
    p.add_argument('--target_delay',    type=int, default=1)
    p.add_argument('--patience',        type=int, default=15)
    p.add_argument('--seed',            type=int, default=42)
    p.add_argument('--test_eval_every', type=int, default=5)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seq_lengths = [int(s) for s in args.seq_lengths.split(',') if s.strip()]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for seq_len in seq_lengths:
        seq_out = out_root / f'seq{seq_len}'
        seq_out.mkdir(parents=True, exist_ok=True)

        summary_rows, all_histories = run_all_experiments(args, seq_len)
        all_summary_rows.extend(summary_rows)

        # Per-sequence CSV
        save_results_table(summary_rows, seq_out / f'results_seq{seq_len}.csv')

        # Comparison plots
        save_comparison_plots(all_histories, seq_len, seq_out)

    # Combined CSV (all sequence lengths together)
    save_results_table(all_summary_rows, out_root / 'results_all_sequences.csv')

    print(f"\nAll done. Outputs written to: {out_root.resolve()}")