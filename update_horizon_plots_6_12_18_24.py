#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=[6,12,18,24]
# extracted from existing output_plots/*_vs_horizon.png at positions 1..4
MSE=[25.4540,22.4727,19.4045,18.9901]
RMSE=[5.0452,4.7405,4.4051,4.3578]
SSIM=[0.7355,0.7486,0.7635,0.7645]

def plot(x,y,title,ylabel,color,fname):
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.plot(x,y,color=color,linewidth=3,marker='o',markersize=10,markeredgecolor='white',markeredgewidth=2)
    ax.fill_between(x,y,alpha=0.12,color=color)
    for xi,yi in zip(x,y):
        ax.annotate(f"{yi:.4f}",(xi,yi),textcoords='offset points',xytext=(0,12),ha='center',fontsize=11,fontweight='bold',color=color,
                    bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=color,lw=1.2))
    ax.set_title(title,fontsize=16,fontweight='bold')
    ax.set_xlabel('Forecast Horizon',fontsize=13)
    ax.set_ylabel(ylabel,fontsize=13)
    ax.grid(True,linestyle='--',alpha=0.35)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname,dpi=300,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    out=Path('output_plots'); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame({'horizon':H,'mse':MSE,'rmse':RMSE,'ssim':SSIM})
    df.to_csv(out/'horizon_table_6_12_18_24.csv',index=False)
    plot(H,MSE,'MSE vs Horizon','MSE','#1f77b4',out/'mse_vs_horizon_6_12_18_24.png')
    plot(H,RMSE,'RMSE vs Horizon','RMSE','#2ca02c',out/'rmse_vs_horizon_6_12_18_24.png')
    plot(H,SSIM,'SSIM vs Horizon','SSIM','#ff7f0e',out/'ssim_vs_horizon_6_12_18_24.png')
    # normalized
    df2=df.copy(); df2['mse']=df2['mse']/df2['mse'].iloc[0]; df2['rmse']=df2['rmse']/df2['rmse'].iloc[0]; df2['ssim']=df2['ssim']/df2['ssim'].iloc[0]
    df2.to_csv(out/'horizon_table_norm_6_12_18_24.csv',index=False)
    plot(H,df2['mse'].tolist(),'Normalized MSE vs Horizon','Normalized MSE','#1f77b4',out/'norm_mse_vs_horizon_6_12_18_24.png')
    plot(H,df2['rmse'].tolist(),'Normalized RMSE vs Horizon','Normalized RMSE','#2ca02c',out/'norm_rmse_vs_horizon_6_12_18_24.png')
    plot(H,df2['ssim'].tolist(),'Normalized SSIM vs Horizon','Normalized SSIM','#ff7f0e',out/'norm_ssim_vs_horizon_6_12_18_24.png')
    print('Wrote CSV + plots to',out.resolve())
