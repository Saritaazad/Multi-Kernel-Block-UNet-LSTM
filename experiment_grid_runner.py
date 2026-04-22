#!/usr/bin/env python3
"""
Experiment Grid Runner for Multi-Horizon Model Comparison
Runs 9 configurations: 3 models x 2 input configs x 2 loss functions
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
warnings.filterwarnings('ignore')

# Import the original pipeline
try:
    from MKB_UNet_LSTM_2 import (
        run_model_pipeline, run_convlstm_pipeline,
        GridAerosolDataset, train_model, evaluate_model
    )
except ImportError:
    print("Error: Cannot import from original pipeline file")
    print("The original file may have been corrupted. Please restore MKB_UNet_LSTM(2).py")
    exit(1)

# DA-ConvLSTM Implementation (Dual Attention)
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

def print_results(results, model_name="Model"):
    """Print results in a formatted way."""
    print(f"\n{model_name} Results:")
    print("-"*70)
    for metric, value in results.items():
        try:
            if isinstance(value, (int, float, np.floating, np.integer)):
                print(f"{metric:<25}: {float(value):.4f}")
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                print(f"{metric:<25}: {float(value.item()):.4f}")
            else:
                print(f"{metric:<25}: {value}")
        except Exception:
            print(f"{metric:<25}: {value}")
    print("-"*70)

def run_daconvlstm_pipeline(data_root_path,
                           include_t2m_input=True,
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
                           lambda_ssim=0.1,
                           ssim_window_size=5,
                           time_weight_gamma=0.95,
                           predict_delta=True,
                           use_mse_magnitude=False,
                           huber_delta=0.5,
                           ssim_all_timesteps=False,
                           lr_schedule='cosine'):
    """Run DA-ConvLSTM forecasting pipeline with sequence evaluation."""
    print("\n" + "=" * 70)
    print("DUAL ATTENTION ConvLSTM PIPELINE")
    print("=" * 70)
    
    # Determine input channels
    in_ch = 4 if include_t2m_input else 3
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data using original dataset class
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
    val_indices = indices[train_count: train_count + val_count]
    test_indices = indices[train_count + val_count:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize DA-ConvLSTM model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DAConvLSTMForecast(in_ch=in_ch, hidden_dim=hidden_dim, out_ch=1).to(device)
    
    # Train model using original train_model function
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        lr_schedule=lr_schedule,
        lambda_huber=lambda_huber,
        lambda_temp=lambda_temp,
        lambda_grad=lambda_grad,
        lambda_ssim=lambda_ssim,
        ssim_window_size=ssim_window_size,
        time_weight_gamma=time_weight_gamma,
        use_mse_magnitude=use_mse_magnitude,
        huber_delta=huber_delta,
        ssim_all_timesteps=ssim_all_timesteps,
        test_loader=test_loader,
        test_eval_every=test_eval_every,
        output_dir=output_dir,
        track_val_metrics_each_epoch=True,
        track_val_seq_metrics_each_epoch=True,
        val_seq_metrics_max_horizon=5
    )
    
    # Evaluate model using original evaluate_model function
    results = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        sequence_length=sequence_length,
        target_delay=target_delay,
        predict_delta=predict_delta,
        use_mse_magnitude=use_mse_magnitude,
        output_dir=output_dir
    )
    
    # Print results
    print_results(results, model_name="DA-ConvLSTM")
    
    # Save results CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(output_dir, 'daconvlstm_results.csv'), index=False)
    
    return results, history

def run_experiment_grid(data_root_path,
                       output_dir='experiment_results',
                       batch_size=8,
                       num_epochs=30,
                       sequence_length=6,
                       target_delay=1,
                       test_eval_every=5,
                       seed=42,
                       hidden_dim=96,
                       lr=5e-4):
    """Run 9 experiments: 3 models x 2 input configs x 2 loss functions."""
    
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENT GRID: 9 CONFIGURATIONS")
    print("=" * 80)
    
    # Experiment configurations
    models = ['ConvLSTM', 'DA-ConvLSTM', 'Ours']
    extra_inputs = [False, True]  # No T2m, Yes T2m
    loss_functions = ['Huber', 'Physics']  # Huber only, Physics-informed
    
    results = []
    total_experiments = len(models) * len(extra_inputs) * len(loss_functions)
    
    for i, model in enumerate(models):
        for j, extra_input in enumerate(extra_inputs):
            for k, loss_fn in enumerate(loss_functions):
                exp_num = i * len(extra_inputs) * len(loss_functions) + j * len(loss_functions) + k + 1
                
                print(f"\n{'='*60}")
                print(f"EXPERIMENT {exp_num}/{total_experiments}")
                print(f"Model: {model}, Extra Input: {'Yes' if extra_input else 'No'}, Loss: {loss_fn}")
                print(f"{'='*60}")
                
                # Create experiment-specific output directory
                exp_name = f"{model.lower().replace('-', '_')}_extra_{extra_input}_{loss_fn.lower()}"
                exp_output_dir = os.path.join(output_dir, exp_name)
                os.makedirs(exp_output_dir, exist_ok=True)
                
                try:
                    # Set loss function parameters
                    if loss_fn == 'Huber':
                        lambda_huber = 1.0
                        lambda_temp = 0.0
                        lambda_grad = 0.0
                        lambda_ssim = 0.0
                    else:  # Physics
                        lambda_huber = 1.0
                        lambda_temp = 0.7
                        lambda_grad = 0.3
                        lambda_ssim = 0.1
                    
                    # Run experiment based on model
                    if model == 'ConvLSTM':
                        # Use original ConvLSTM pipeline but fix return values
                        trained_model, history, conv_results = run_convlstm_pipeline(
                            data_root_path=data_root_path,
                            include_t2m_input=extra_input,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            sequence_length=sequence_length,
                            target_delay=target_delay,
                            test_eval_every=test_eval_every,
                            output_dir=exp_output_dir,
                            seed=seed,
                            hidden_dim=hidden_dim,
                            lr=lr,
                            lambda_huber=lambda_huber,
                            lambda_temp=lambda_temp,
                            lambda_grad=lambda_grad,
                            lambda_ssim=lambda_ssim
                        )
                        result = conv_results
                    elif model == 'DA-ConvLSTM':
                        result, history = run_daconvlstm_pipeline(
                            data_root_path=data_root_path,
                            include_t2m_input=extra_input,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            sequence_length=sequence_length,
                            target_delay=target_delay,
                            test_eval_every=test_eval_every,
                            output_dir=exp_output_dir,
                            seed=seed,
                            hidden_dim=hidden_dim,
                            lr=lr,
                            lambda_huber=lambda_huber,
                            lambda_temp=lambda_temp,
                            lambda_grad=lambda_grad,
                            lambda_ssim=lambda_ssim
                        )
                    else:  # Ours (MKCNN-UNet-LSTM)
                        trained_model, history, mk_results = run_model_pipeline(
                            data_root_path=data_root_path,
                            include_t2m_input=extra_input,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            sequence_length=sequence_length,
                            target_delay=target_delay,
                            test_eval_every=test_eval_every,
                            output_dir=exp_output_dir,
                            seed=seed,
                            hidden_dim=hidden_dim,
                            lr=lr,
                            lambda_huber=lambda_huber,
                            lambda_temp=lambda_temp,
                            lambda_grad=lambda_grad,
                            lambda_ssim=lambda_ssim
                        )
                        result = mk_results
                    
                    # Add experiment metadata
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
    # Add metric columns if they exist
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
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=6,
                       help='Sequence length for forecasting')
    parser.add_argument('--target_delay', type=int, default=1,
                       help='Target delay for forecasting')
    parser.add_argument('--hidden_dim', type=int, default=96,
                       help='Hidden dimension for models')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--test_eval_every', type=int, default=5,
                       help='Test evaluation frequency')
    
    args = parser.parse_args()
    
    # Run experiment grid
    results_df = run_experiment_grid(
        data_root_path=args.data_root_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        sequence_length=args.sequence_length,
        target_delay=args.target_delay,
        test_eval_every=args.test_eval_every,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        lr=args.lr
    )
