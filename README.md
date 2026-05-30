# Physics-Informed MKCNN-UNet-LSTM Spatiotemporal Pipeline

This repository/directory contains the training and evaluation pipeline for a novel **Physics-Informed Multi-Kernel CNN-UNet-LSTM** model developed by Gaurav. The model is tailored for the spatiotemporal prediction of variables (such as Temperature `T2m`), influenced by physical features such as Black Carbon (BC), Sulfate (SU), and Dust (DU) Aerosol Optical Depths (AOD).

## Model Architecture

The `CNN_UNet_LSTM` leverages several interconnected blocks:
- **Time-Series Sequences:** Transforms raw spatial grid inputs into temporal windows, preserving autoregressive signals and encoding temporal features (sine/cosine of months). 
- **`MultiKernelBlock (MKB)`:** Operates via independent Depthwise convolutions across multiple kernel scales to extract multi-scale spatial representations of physical drivers (like aerosol data).
- **`UNetOdd`:** An overarching spatial feature extractor incorporating Squeeze-and-Excitation (SE) blocks, bottleneck pathways, and encoder-decoder loops for high-resolution 19x19 gridded spatial dynamics.
- **`ConvLSTMForecast` / `LSTM`:** Ensures robust temporal mapping. Feature maps are flattened directly into an LSTM processing node to encode continuous inertia, predicting up to various temporal horizons.

## Physics-Informed Loss Function

To ensure that the predictions conform to physical realities and temporal consistency, a novel `PhysicsInformedLoss` function is integrated:
1. **Magnitude/Huber Loss:** Controls the error on overall magnitude variations using regression norms.
2. **Temporal Consistency (Physics Inertia):** Applies an MSE penalization to structural leaps ($T_{t} - T_{t-1}$) tracking model stability.
3. **Spatial Gradient (Sharpness):** Implements an L1 regularizer on adjacent pixel gradients (both DX and DY) to preserve morphological sharpness in forecasted grids.
4. **Structural Similarity (SSIM):** Leverages a differentiable SSIM mapping module dynamically applied to evaluating image consistency.

## Usage

### 1. Requirements

Dependencies used by this model pipeline are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

The contents of `requirements.txt` are as follows:
```text
torch
numpy
scikit-learn
scikit-image
pandas
matplotlib
seaborn
geopandas
shapely
```

### 2. Training the Model

The easiest way to execute the entire training and visualization pipeline is by running:
```bash
python MKB_UNet_LSTM\(2\).py --data_root_path "/path/to/csv/data" --num_epochs 100 --batch_size 8
```

Argument Examples:
- `--include_t2m_input`: Enable testing of continuous autoregression (lagged spatial predictors).
- `--ablation`: Run input ablation automatically against the variables (leaves out NO_AEROSOL, SU, BC, etc., and calculates relative loss contributions).
- `--tune`: Start hyperparameter tuning spanning different architectures and kernel sizes.
- `--benchmark`: Automatically benchmark throughput vs a simpler `ConvLSTM`.

### 3. Using Shapefiles

The geographic bounds and masking of the output predictions are constrained using shapefiles (`.shp`). By default, the model comes with a standard North-West Himalayas / India shapefile.

- **`--shapefile_path`**: Specifies the absolute or relative path to the `.shp` file to use.
  - Default: `"NWH_States_Shapefile-20260120T110552Z-3-001/NWH_States_Shapefile/NWH_states.shp"`
  - Example: `--shapefile_path "/path/to/custom_shapefile.shp"`

The pipeline uses `geopandas` to load this shapefile which overlays realistic geographic boundaries over the plotted `(19, 19)` prediction grids.

### 4. Plotting and Visualizations

The pipeline emits a comprehensive suite of plots during training and evaluation. By default, all visual outputs are saved into the `--output_dir` (which defaults to `output_plots/`).

**Plotting Flags:**
- **`--output_dir`**: Directory where all plots will be saved. Default: `output_plots/`
- **`--plot_only`**: Use this flag to strictly generate output plots based on a loaded checkpoint, without initiating a training run.

**Saved Output Structure:**
When the model executes, it will sequentially save these artifacts into the output directory:
- **`training_history_physics.png`**: Breakdown of overall validation loss metrics mapped continuously to Physics-Informed bounds over all epochs.
- **`mkcnn_features/`**: Spatiotemporal visualization of individual kernels masking physical spaces. 
- **`predictions/`**: Visual layout featuring Input variables, Ground Truth Maps, and Output Predictions rendered explicitly bounded against designated Geospatial representations via the loaded shapefile.

## Dataset Assumption

The `GridAerosolDataset` inherently expects input data mapped to exactly `(19, 19)` scaled spatial coordinates, aligned from a transposed standard shapefile clip over North-West Himalayas / India (`[70.0 - 81.25 E, 28.0 - 37.0 N]`). Inputs needed (`[var]_time_series.csv`):
- `BC_AOD_time_series.csv`
- `SU_AOD_time_series.csv`
- `DU_AOD_pm25_time_series.csv`
- `T2m_time_series.csv`
