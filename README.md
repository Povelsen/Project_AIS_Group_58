# AIS Trajectory Prediction with Transformer

This project implements a generative Transformer model (`AISGPTFormer`) for predicting vessel trajectories using AIS data. It includes a complete pipeline from data acquisition and preprocessing to model training and evaluation.

## Prerequisites

Ensure you have a Python environment set up with the following dependencies:

*   `torch`
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `cartopy`
*   `tqdm`
*   `scikit-learn`
*   `requests`

You can install them via pip:

```bash
pip install torch numpy pandas matplotlib cartopy tqdm scikit-learn requests
```

## 1. Data Preparation

### Step 1: Download AIS Data
Use the `download_ais_to_folder.py` script to download raw AIS data from the Danish Maritime Authority.

```bash
python download_ais_to_folder.py
```
**Inputs:**
*   **Year**: e.g., `2025`
*   **Month**: e.g., `02`
*   **Start Day**: e.g., `12`
*   **End Day**: e.g., `18`

The data will be downloaded and extracted to `raw_ais_data/`.

### Step 2: Preprocess Data
Run `build_ct_dataset.py` to clean the data, build trajectories, and generate training/validation/test sets.

```bash
python build_ct_dataset.py
```
This script performs:
*   **Filtering**: Selects Cargo ships, filters by speed (SOG < 30 knots), and removes invalid navigational statuses.
*   **Resampling**: Resamples trajectories to a fixed time interval (default: 10 mins).
*   **Splitting**: Splits trajectories based on time gaps (> 2 hours) and duration constraints.
*   **Feature Engineering**: Generates auxiliary features like distance to coast and nearest port.
*   **Formatting**: Converts data into a **PyTorch-ready format** (pickled lists of dictionaries containing numpy arrays for Lat, Lon, SOG, COG, etc.).
*   **Dataset Split**: Saves `ct_dma_train.pkl`, `ct_dma_valid.pkl`, and `ct_dma_test.pkl` in `cleaned_ais_data/`.

## 2. Configuration

The model configuration is handled via Python files in the `config_files/` directory.

*   **Default Config**: `config_files/config_trAISformer_default.py`
*   **Experiment Config**: `config_files/config_file_with_FLAGS.py`

To switch between configurations, edit the import statements in `trAISformer.py`:

```python
# Use default
from config_files.config_trAISformer_default import Config

# OR Use experiment config
# from config_files.config_file_with_FLAGS import Config
```

### Experiment Configurations

You can switch between different model configurations by changing the `experiment` variable in `config_files/config_file_with_FLAGS.py`.

| Experiment Name | Model Size | Context | Description |
| :--- | :--- | :--- | :--- |
| **`baseline`** | **Medium**<br>8 Layers, 8 Heads<br>896 Dim | 3 Hours | The standard model. Good starting point. |
| **`smaller_model`** | **Small**<br>4 Layers, 4 Heads<br>~512 Dim | 3 Hours | Faster to train, uses less memory. Good for debugging or weak hardware. |
| **`best_combined`** | **Medium**<br>8 Layers, 8 Heads<br>896 Dim | 3 Hours | Optimized baseline with Early Stopping and Higher Dropout (0.3) to prevent overfitting. |
| **`long_run`** | **Medium**<br>8 Layers, 8 Heads<br>896 Dim | 3 Hours | Designed for long training. Uses **Data Augmentation** (Noise + Jitter) and Lower LR (1e-4) to generalize better. |
| **`long_run_curriculum`**<br>*(Recommended)* | **Large**<br>12 Layers, 16 Heads<br>896 Dim | **6 Hours** | **State-of-the-art config.**<br>• **Larger Model**: Smarter, learns complex patterns.<br>• **Longer Memory**: Sees 6h history instead of 3h.<br>• **Curriculum Learning**: Learns 1h -> 3h -> 17h predictions incrementally.<br>• **Robust**: Uses Data Augmentation + Early Stopping. |

**How to run a specific model:**
1.  Open `config_files/config_file_with_FLAGS.py`.
2.  Change line 17: `experiment = "long_run_curriculum"` (or your choice).
3.  Run `python trAISformer.py`.

## 3. Training & Evaluation

To train the model and evaluate it on the test set, run:

```bash
python trAISformer.py
```

### Training Process
1.  **Training**: If `cf.retrain` is `True` in the config, it trains the `TrAISformer` model.
2.  **Logging**: Saves checkpoints and logs to the directory specified in `cf.savedir` (default: `results/<experiment_name>`).
3.  **Visual Monitoring**:
    *   **Per-Epoch Visualization**: At the end of each epoch, the model generates a map visualization (`epoch_XXX.jpg`) showing predictions on a sample batch from the test set. This allows you to visually monitor the model's learning progress.
    *   **Loss Curve**: A `loss_curve.png` is saved showing the training and validation loss over time.

### Evaluation & Metrics
After training, the script loads the best checkpoint and evaluates it on the test set.

**Prediction Error Metric:**
The primary metric is the **Mean Minimum Haversine Distance** (Best-of-N).
*   **Generative Nature**: Since the future is uncertain, the model generates `N` (default: 16) possible future trajectories for each input.
*   **Best-of-N**: We compare all `N` generated paths to the *true* ground truth path.
*   **Error Calculation**: The error is the average distance (in km) between the *best* matching generated path and the true path.
*   **Interpretation**: A lower error means the model successfully assigned high probability to the true outcome, effectively capturing the possibility space of vessel movements.

**Outputs:**
*   `prediction_error.png`: A plot showing the mean prediction error (km) over the prediction horizon (time).
*   `errors.npy`: The raw error values for further analysis.
*   `trajectories.npy`: Contains the full ground truth and predicted trajectories for visualization.

## 4. Visualization

To visualize the model's predictions on a map, use `plot_cartopy_example.py`.

```bash
python plot_cartopy_example.py
```

**Output:**
*   `results/<experiment_name>/cartopy_example.png`: A high-quality map showing:
    *   **White**: Observed trajectory (input).
    *   **Green**: True future trajectory (ground truth).
    *   **Red**: Predicted future trajectory.
    *   **Ports**: Locations of ports (if enabled).

## Project Structure

*   `download_ais_to_folder.py`: Data downloader.
*   `build_ct_dataset.py`: Data preprocessing and feature engineering.
*   `trAISformer.py`: Main script for training and evaluation.
*   `trainers.py`: Training loop and sampling logic.
*   `models.py`: Transformer model architecture.
*   `datasets.py`: PyTorch Dataset implementations.
*   `plot_cartopy_example.py`: Visualization script.
