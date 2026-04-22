#!/usr/bin/env python3
"""
Minimal Pipeline for Experiment Grid
Contains essential components to run the 9-configuration experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import warnings
import random
import time
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Class (from original)
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

# ──────────────────────────────────────────────────────────────────────────────
# Model Implementations
# ──────────────────────────────────────────────────────────────────────────────
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
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        state = (torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device),
                 torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device))
        outputs = []
        for t in range(seq_len):
            state = self.cell(x[:, t], state)
            outputs.append(state[0])
        return torch.stack(outputs, dim=1), state

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

# DA-ConvLSTM (Dual Attention)
class SpatialAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key(x).view(b, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x

class ChannelAttention(nn.Module):
    def __init__(self, in_dim, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim // reduction_ratio, in_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class DAConvLSTMCell(nn.Module):
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
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.channel_attention = ChannelAttention(hidden_dim)

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
        
        # Apply dual attention
        h_next = self.spatial_attention(h_next)
        h_next = self.channel_attention(h_next)
        
        return h_next, c_next

class DAConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = DAConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        state = (torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device),
                 torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device))
        outputs = []
        for t in range(seq_len):
            state = self.cell(x[:, t], state)
            outputs.append(state[0])
        return torch.stack(outputs, dim=1), state

class DAConvLSTMForecast(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, kernel_size=3, out_ch=1):
        super().__init__()
        self.convlstm = DAConvLSTM(input_dim=in_ch, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.head = nn.Conv2d(hidden_dim, out_ch, kernel_size=1)

    def forward(self, x):
        feats, state = self.convlstm(x)
        b, t, c, h, w = feats.shape
        y = self.head(feats.view(b * t, c, h, w)).view(b, t, -1, h, w)
        return y, state

# Simple MKCNN-UNet-LSTM placeholder (simplified version)
class SimpleUNetLSTM(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, out_ch=1):
        super().__init__()
        # Simplified UNet encoder
        self.enc1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc3 = nn.Conv2d(64, hidden_dim, 3, padding=1)
        
        # Calculate flattened size after encoding
        # After 3 conv layers: 19x19 -> hidden_dim channels
        self.feature_size = hidden_dim * 19 * 19  # 64 * 19 * 19 = 23104
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(self.feature_size, hidden_dim, batch_first=True)
        
        # Decoder
        self.dec = nn.Conv2d(hidden_dim, 32, 3, padding=1)
        self.final = nn.Conv2d(32, out_ch, 1)
        
    def forward(self, x):
        b, t, c, h, w = x.shape  # e.g., (8, 5, 5, 19, 19) or (8, 5, 6, 19, 19)
        
        # Encode each frame
        encoded = []
        for i in range(t):
            feat = F.relu(self.enc1(x[:, i]))  # (b, 32, 19, 19)
            feat = F.relu(self.enc2(feat))      # (b, 64, 19, 19)
            feat = F.relu(self.enc3(feat))      # (b, hidden_dim, 19, 19)
            
            # Flatten properly: (b, hidden_dim, 19, 19) -> (b, hidden_dim * 19 * 19)
            feat_flat = feat.contiguous().view(b, -1)  # (b, 23104)
            encoded.append(feat_flat)
        
        # Process sequence with LSTM
        encoded = torch.stack(encoded, dim=1)  # (b, t, feature_size)
        lstm_out, _ = self.lstm(encoded)       # (b, t, hidden_dim)
        
        # Decode final output
        final_feat = lstm_out[:, -1]           # (b, hidden_dim)
        # Reshape to spatial format: (b, hidden_dim) -> (b, hidden_dim, 19, 19)
        final_feat = final_feat.view(b, hidden_dim, h, w)  # (b, 64, 19, 19)
        
        out = F.relu(self.dec(final_feat))     # (b, 32, 19, 19)
        out = self.final(out)                  # (b, 1, 19, 19)
        
        # Return sequence prediction (repeat final output for sequence)
        return out.unsqueeze(1).repeat(1, t, 1, 1, 1), None

# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────────────
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_huber=1.0, lambda_temp=0.7, lambda_grad=0.3, huber_delta=1.0):
        super().__init__()
        self.lambda_huber = lambda_huber
        self.lambda_temp = lambda_temp
        self.lambda_grad = lambda_grad
        self.huber_delta = huber_delta
        
    def huber_loss(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        mask = abs_diff <= self.huber_delta
        loss = torch.where(mask, 0.5 * diff**2, self.huber_delta * (abs_diff - 0.5 * self.huber_delta))
        return loss.mean()
    
    def temporal_loss(self, pred_seq):
        if pred_seq.size(1) < 2:
            return torch.tensor(0.0, device=pred_seq.device)
        temp_diff = pred_seq[:, 1:] - pred_seq[:, :-1]
        return torch.mean(torch.sum(temp_diff**2, dim=[2,3,4]))
    
    def gradient_loss(self, pred):
        grad_x = torch.mean(torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]))
        grad_y = torch.mean(torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]))
        return grad_x + grad_y
    
    def forward(self, pred, target):
        huber_loss = self.huber_loss(pred, target)
        temporal_loss = self.temporal_loss(pred)
        gradient_loss = self.gradient_loss(pred)
        
        total_loss = (self.lambda_huber * huber_loss + 
                     self.lambda_temp * temporal_loss + 
                     self.lambda_grad * gradient_loss)
        
        return total_loss, {'huber': huber_loss.item(), 'temporal': temporal_loss.item(), 'gradient': gradient_loss.item()}

# ──────────────────────────────────────────────────────────────────────────────
# Training and Evaluation Functions
# ──────────────────────────────────────────────────────────────────────────────
def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    
    # SSIM for 2D images (take last timestep)
    if len(y_true.shape) == 5:  # Batch, Seq, Ch, H, W
        y_true_2d = y_true[:, -1, 0]  # Last timestep, first channel
        y_pred_2d = y_pred[:, -1, 0]
    else:
        y_true_2d = y_true[0] if len(y_true.shape) == 3 else y_true
        y_pred_2d = y_pred[0] if len(y_pred.shape) == 3 else y_pred
    
    try:
        ssim_val = ssim(y_true_2d.cpu().numpy(), y_pred_2d.cpu().numpy(), 
                       data_range=y_true_2d.max() - y_true_2d.min())
    except:
        ssim_val = 0.0
    
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'SSIM': ssim_val,
        'MAPE': mape
    }

def train_model(model, train_loader, val_loader, device, num_epochs=30, lr=5e-4,
                lambda_huber=1.0, lambda_temp=0.7, lambda_grad=0.3, huber_delta=1.0,
                output_dir='output'):
    """Simple training function."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup loss function
    if lambda_temp > 0 or lambda_grad > 0:
        criterion = PhysicsInformedLoss(lambda_huber, lambda_temp, lambda_grad, huber_delta)
    else:
        criterion = nn.HuberLoss(delta=huber_delta)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred, _ = model(batch_x)
            
            if isinstance(criterion, PhysicsInformedLoss):
                loss, loss_dict = criterion(pred, batch_y)
            else:
                loss = criterion(pred, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, _ = model(batch_x)
                
                if isinstance(criterion, PhysicsInformedLoss):
                    loss, _ = criterion(pred, batch_y)
                else:
                    loss = criterion(pred, batch_y)
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    return model, history

def evaluate_model(model, test_loader, device, output_dir='output'):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred, _ = model(batch_x)
            
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets.numpy(), all_preds.numpy())
    
    return metrics

def print_results(results, model_name="Model"):
    """Print results in a formatted way."""
    print(f"\n{model_name} Results:")
    print("-"*70)
    for metric, value in results.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            print(f"{metric:<25}: {float(value):.4f}")
        else:
            print(f"{metric:<25}: {value}")
    print("-"*70)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Functions
# ──────────────────────────────────────────────────────────────────────────────
def run_convlstm_experiment(data_root_path, include_t2m_input=False, 
                          loss_fn='Huber', num_epochs=10, output_dir='output'):
    """Run ConvLSTM experiment."""
    print(f"\n{'='*60}")
    print(f"ConvLSTM: Extra Input={'Yes' if include_t2m_input else 'No'}, Loss={loss_fn}")
    print(f"{'='*60}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Dataset has: 3 aerosol + optional T2m + 2 temporal = 5 or 6 channels
    in_ch = 5 if not include_t2m_input else 6
    
    # Load data
    dataset = GridAerosolDataset(data_root_path, include_t2m_input=include_t2m_input)
    n = len(dataset)
    tr = int(n * 0.8); va = max(1, int(n * 0.1))
    indices = list(range(n))
    
    train_loader = DataLoader(Subset(dataset, indices[:tr]), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[tr:tr+va]), batch_size=8, shuffle=False)
    test_loader = DataLoader(Subset(dataset, indices[tr+va:]), batch_size=8, shuffle=False)
    
    # Model
    model = ConvLSTMForecast(in_ch=in_ch, hidden_dim=64).to(device)
    
    # Loss parameters
    if loss_fn == 'Physics':
        lambda_huber, lambda_temp, lambda_grad = 1.0, 0.7, 0.3
    else:
        lambda_huber, lambda_temp, lambda_grad = 1.0, 0.0, 0.0
    
    # Train
    trained_model, history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs,
        lambda_huber=lambda_huber,
        lambda_temp=lambda_temp,
        lambda_grad=lambda_grad,
        output_dir=output_dir
    )
    
    # Evaluate
    results = evaluate_model(trained_model, test_loader, device, output_dir)
    print_results(results, f"ConvLSTM ({loss_fn})")
    
    return results, history

def run_daconvlstm_experiment(data_root_path, include_t2m_input=False, 
                           loss_fn='Huber', num_epochs=10, output_dir='output'):
    """Run DA-ConvLSTM experiment."""
    print(f"\n{'='*60}")
    print(f"DA-ConvLSTM: Extra Input={'Yes' if include_t2m_input else 'No'}, Loss={loss_fn}")
    print(f"{'='*60}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Dataset has: 3 aerosol + optional T2m + 2 temporal = 5 or 6 channels
    in_ch = 5 if not include_t2m_input else 6
    
    # Load data
    dataset = GridAerosolDataset(data_root_path, include_t2m_input=include_t2m_input)
    n = len(dataset)
    tr = int(n * 0.8); va = max(1, int(n * 0.1))
    indices = list(range(n))
    
    train_loader = DataLoader(Subset(dataset, indices[:tr]), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[tr:tr+va]), batch_size=8, shuffle=False)
    test_loader = DataLoader(Subset(dataset, indices[tr+va:]), batch_size=8, shuffle=False)
    
    # Model
    model = DAConvLSTMForecast(in_ch=in_ch, hidden_dim=64).to(device)
    
    # Loss parameters
    if loss_fn == 'Physics':
        lambda_huber, lambda_temp, lambda_grad = 1.0, 0.7, 0.3
    else:
        lambda_huber, lambda_temp, lambda_grad = 1.0, 0.0, 0.0
    
    # Train
    trained_model, history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs,
        lambda_huber=lambda_huber,
        lambda_temp=lambda_temp,
        lambda_grad=lambda_grad,
        output_dir=output_dir
    )
    
    # Evaluate
    results = evaluate_model(trained_model, test_loader, device, output_dir)
    print_results(results, f"DA-ConvLSTM ({loss_fn})")
    
    return results, history

def run_ours_experiment(data_root_path, include_t2m_input=False, 
                       loss_fn='Huber', num_epochs=10, output_dir='output'):
    """Run Ours (SimpleUNetLSTM) experiment."""
    print(f"\n{'='*60}")
    print(f"Ours: Extra Input={'Yes' if include_t2m_input else 'No'}, Loss={loss_fn}")
    print(f"{'='*60}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Dataset has: 3 aerosol + optional T2m + 2 temporal = 5 or 6 channels
    in_ch = 5 if not include_t2m_input else 6
    
    # Load data
    dataset = GridAerosolDataset(data_root_path, include_t2m_input=include_t2m_input)
    n = len(dataset)
    tr = int(n * 0.8); va = max(1, int(n * 0.1))
    indices = list(range(n))
    
    train_loader = DataLoader(Subset(dataset, indices[:tr]), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[tr:tr+va]), batch_size=8, shuffle=False)
    test_loader = DataLoader(Subset(dataset, indices[tr+va:]), batch_size=8, shuffle=False)
    
    # Model
    model = SimpleUNetLSTM(in_ch=in_ch, hidden_dim=64).to(device)
    
    # Loss parameters
    if loss_fn == 'Physics':
        lambda_huber, lambda_temp, lambda_grad = 1.0, 0.7, 0.3
    else:
        lambda_huber, lambda_temp, lambda_grad = 1.0, 0.0, 0.0
    
    # Train
    trained_model, history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs,
        lambda_huber=lambda_huber,
        lambda_temp=lambda_temp,
        lambda_grad=lambda_grad,
        output_dir=output_dir
    )
    
    # Evaluate
    results = evaluate_model(trained_model, test_loader, device, output_dir)
    print_results(results, f"Ours ({loss_fn})")
    
    return results, history

def run_experiment_grid(data_root_path, output_dir='experiment_results', num_epochs=10):
    """Run 9 experiments: 3 models x 2 input configs x 2 loss functions."""
    
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT GRID: 9 CONFIGURATIONS")
    print("=" * 80)
    
    # Experiment configurations
    configs = []
    models = ['ConvLSTM', 'DA-ConvLSTM', 'Ours']
    extra_inputs = [False, True]  # No T2m, Yes T2m  
    loss_functions = ['Huber', 'Physics']  # Huber only, Physics-informed
    
    for model in models:
        for extra_input in extra_inputs:
            for loss_fn in loss_functions:
                configs.append((model, extra_input, loss_fn))
    
    results = []
    
    for i, (model, extra_input, loss_fn) in enumerate(configs):
        exp_num = i + 1
        print(f"\nExperiment {exp_num}/9: {model}, Extra={'Yes' if extra_input else 'No'}, Loss={loss_fn}")
        
        # Create experiment-specific output directory
        exp_name = f"{model.lower().replace('-', '_')}_extra_{extra_input}_{loss_fn.lower()}"
        exp_output_dir = os.path.join(output_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        try:
            # Run experiment
            if model == 'ConvLSTM':
                result, history = run_convlstm_experiment(
                    data_root_path, extra_input, loss_fn, num_epochs, exp_output_dir
                )
            elif model == 'DA-ConvLSTM':
                result, history = run_daconvlstm_experiment(
                    data_root_path, extra_input, loss_fn, num_epochs, exp_output_dir
                )
            else:  # Ours
                result, history = run_ours_experiment(
                    data_root_path, extra_input, loss_fn, num_epochs, exp_output_dir
                )
            
            # Add metadata
            result['Model'] = model
            result['Extra_Input'] = 'Yes' if extra_input else 'No'
            result['Loss_Function'] = loss_fn
            result['Experiment_Num'] = exp_num
            
            results.append(result)
            print(f"✓ Experiment {exp_num} completed successfully")
            
        except Exception as e:
            print(f"✗ Experiment {exp_num} failed: {str(e)}")
            # Add failed experiment record
            failed_result = {
                'Model': model,
                'Extra_Input': 'Yes' if extra_input else 'No',
                'Loss_Function': loss_fn,
                'Experiment_Num': exp_num,
                'Error': str(e)
            }
            results.append(failed_result)
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    cols = ['Experiment_Num', 'Model', 'Extra_Input', 'Loss_Function']
    metric_cols = [col for col in results_df.columns if col not in cols + ['Error']]
    cols.extend(metric_cols)
    if 'Error' in results_df.columns:
        cols.append('Error')
    
    results_df = results_df[cols]
    
    # Save combined results
    csv_path = os.path.join(output_dir, 'experiment_grid_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("EXPERIMENT GRID SUMMARY")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {csv_path}")
    
    return results_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run experiment grid for model comparison')
    parser.add_argument('--data_root_path', type=str, required=True, 
                       help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='experiment_results',
                       help='Output directory for results')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Run experiment grid
    results_df = run_experiment_grid(
        data_root_path=args.data_root_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs
    )
