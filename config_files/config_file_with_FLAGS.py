"""Configuration flags to run the main script.
"""

import os
import pickle
import torch


class Config():
    retrain = True
    tb_log = False
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # =========================================================================
    # EXPERIMENT PRESETS - Set this to quickly switch between experiments
    # =========================================================================
    experiment = "long_run_curriculum"  # Options: "baseline", "early_stopping", "smaller_model", 
                             # "high_dropout", "lower_lr", "data_augmentation", "combined", "best_combined", "long_run", "long_run_curriculum"
    
    # Base configuration
    max_epochs = 20
    batch_size = 64
    n_samples = 16
    
    init_seqlen = 18
    max_seqlen = 120
    min_seqlen = 36
    
    dataset_name = "ct_dma"
    
    use_port_features = True
    use_coast_features = True
    predict_ports = True  # Enable port prediction head
    port_prediction_weight = 0.1  # Weight for port prediction loss

    if dataset_name == "ct_dma":
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        # Add embeddings for auxiliary features
        if use_port_features:
            # CHANGED: Increased from 50 to 512 to cover max ID 508
            port_size = 512  
            n_port_embd = 64
        else:
            port_size = 0
            n_port_embd = 0
            
        if use_coast_features:
            coast_size = 50  # Discretize coast distance
            n_coast_embd = 64
        else:
            coast_size = 0
            n_coast_embd = 0


        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128

        # Update embedding dimension
        n_embd = (n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd + 
                  n_port_embd + n_coast_embd)
        
        # Update full size for output
        full_size = (lat_size + lon_size + sog_size + cog_size + 
                     port_size + coast_size)
    
        lat_min = 50.0  # Was 55.5
        lat_max = 60.0  # Was 58.0
        lon_min = 0.0   # Was 10.3
        lon_max = 20.0  # Was 13.0

    # Model and sampling flags
    mode = "pos"
    sample_mode = "pos_vicinity"
    top_k = 10
    r_vicinity = 40
    
    # Blur flags
    blur = False
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0
    
    # Data flags
    datadir = "/work3/s214381/AIS_data/cleaned_ais_data/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"
    
    # Model parameters (will be overridden by experiments)
    n_head = 8
    n_layer = 8
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
    
    # Dropout parameters (will be overridden by experiments)
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # Optimization parameters (will be overridden by experiments)
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = True
    warmup_tokens = 1e6
    final_tokens = 1.75e8
    num_workers = 4
    
    # Early stopping parameters
    early_stopping = False
    patience = 10  # Number of epochs to wait for improvement
    min_delta = 1e-4  # Minimum change to qualify as improvement
    
    # Data augmentation flags
    data_augmentation = False
    aug_noise_std = 0.01  # Gaussian noise standard deviation
    aug_temporal_jitter = True  # Random time shifts
    aug_rotation = False  # Rotate trajectories
    
    # =========================================================================
    # APPLY EXPERIMENT CONFIGURATIONS
    # =========================================================================
    if experiment == "baseline":
        # Keep default settings
        exp_suffix = "baseline"
    
    elif experiment == "early_stopping":
        early_stopping = True
        patience = 10
        max_epochs = 100  # Can train longer with early stopping
        exp_suffix = "early_stop"
    
    elif experiment == "smaller_model":
        n_head = 4  # Reduced from 8
        n_layer = 4  # Reduced from 8
        n_lat_embd = 128  # Reduced from 256
        n_lon_embd = 128  # Reduced from 256
        n_sog_embd = 64   # Reduced from 128
        n_cog_embd = 64   # Reduced from 128
        n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
        if use_port_features:
            n_embd += n_port_embd
        if use_coast_features:
            n_embd += n_coast_embd
        exp_suffix = "smaller_model"
    
    elif experiment == "high_dropout":
        embd_pdrop = 0.3   # Increased from 0.1
        resid_pdrop = 0.3  # Increased from 0.1
        attn_pdrop = 0.3   # Increased from 0.1
        exp_suffix = "high_dropout"
    
    elif experiment == "lower_lr":
        learning_rate = 3e-4  # Reduced from 6e-4
        exp_suffix = "lower_lr"
    
    elif experiment == "data_augmentation":
        data_augmentation = True
        aug_noise_std = 0.005
        aug_temporal_jitter = True
        exp_suffix = "data_aug"
    
    elif experiment == "combined":
        # Combine multiple strategies
        early_stopping = True
        patience = 10
        max_epochs = 100
        
        n_head = 6
        n_layer = 6
        n_lat_embd = 192
        n_lon_embd = 192
        n_sog_embd = 96
        n_cog_embd = 96
        n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
        if use_port_features:
            n_embd += n_port_embd
        if use_coast_features:
            n_embd += n_coast_embd
        
        embd_pdrop = 0.2
        resid_pdrop = 0.2
        attn_pdrop = 0.2
        
        learning_rate = 4e-4
        
        data_augmentation = True
        aug_noise_std = 0.005
        
        exp_suffix = "combined"

    elif experiment == "best_combined":
    # ==== Best combination: smaller model + higher dropout + early stopping ====

        # Early stopping
        early_stopping = True
        patience = 10
        max_epochs = 100     # allow more, auto-stop when stable
        
        # Smaller model
        n_head = 8           # Reduced
        n_layer = 8          # Reduced

        n_lat_embd = 256     # Reduced
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
        n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
        if use_port_features:
            n_embd += n_port_embd
        if use_coast_features:
            n_embd += n_coast_embd

        # Higher dropout to reduce overfitting
        embd_pdrop = 0.3
        resid_pdrop = 0.3
        attn_pdrop = 0.3

        # Slightly safer learning rate
        learning_rate = 3e-4

        # Keep augmentation off (best stability), but toggle if needed
        data_augmentation = False

        exp_suffix = "best_combined"

    elif experiment == "long_run":
        # ==== Long Run: Augmentation + Lower LR + Patience to prevent early stopping ====
        
        # 1. Enable Data Augmentation (Crucial for generalization)
        data_augmentation = True
        aug_noise_std = 0.005
        aug_temporal_jitter = True
        
        # 2. Lower Learning Rate (Slower, more careful convergence)
        learning_rate = 1e-4  # Reduced from 3e-4
        
        # 3. Increase Patience (Allow recovery from plateaus)
        early_stopping = True
        patience = 20         # Increased from 10
        max_epochs = 200      # Allow plenty of time
        
        # 4. Keep the stable "best_combined" architecture
        n_head = 8
        n_layer = 8
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
        n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
        if use_port_features:
            n_embd += n_port_embd
        if use_coast_features:
            n_embd += n_coast_embd
            
        # High dropout from best_combined
        embd_pdrop = 0.3
        resid_pdrop = 0.3
        attn_pdrop = 0.3
        
        exp_suffix = "long_run"

    elif experiment == "long_run_curriculum":
        # ==== Curriculum Learning + Stabilized Model (Reduced Size) ====
        
        # 1. Architecture (Reduced from 12 layers/16 heads to prevent overfitting)
        n_head = 8           # Reduced from 16 to 8 (896 / 8 = 112 dim per head)
        n_layer = 8          # Reduced from 12 to 8 (matches "best_combined")
        init_seqlen = 36     # Keep context length
        
        # 2. Curriculum Learning
        # List of tuples: (epoch_limit, max_seqlen_limit)
        curriculum_learning = True
        curriculum_schedule = [(20, 42), (40, 54), (150, 120)]
        
        # 3. Inherit Long Run settings
        data_augmentation = True
        aug_noise_std = 0.005
        aug_temporal_jitter = True
        learning_rate = 1e-4
        early_stopping = True
        patience = 50         
        max_epochs = 150      
        
        # Architecture (Standard)
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
        n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd
        if use_port_features:
            n_embd += n_port_embd
        if use_coast_features:
            n_embd += n_coast_embd
            
        # Dropout (Keep low for now, increase if overfitting persists)
        embd_pdrop = 0.1
        resid_pdrop = 0.1
        attn_pdrop = 0.1
        
        exp_suffix = "long_run_curriculum_v2" 

    
    # =========================================================================
    # SAVE PATHS
    # =========================================================================
    filename = f"{dataset_name}_{mode}_bs{batch_size}_lr{learning_rate}"
    savedir = f"./results/{exp_suffix}/{filename}/"
    ckpt_path = os.path.join(savedir, "model.pt")
    
    def __str__(self):
        """Print current experiment configuration"""
        return f"""
Experiment: {self.experiment}
Max Epochs: {self.max_epochs}
Batch Size: {self.batch_size}
Learning Rate: {self.learning_rate}
Model Size: {self.n_layer} layers, {self.n_head} heads, {self.n_embd} embedding dim
Dropout: {self.embd_pdrop}
Early Stopping: {self.early_stopping} (patience={self.patience if self.early_stopping else 'N/A'})
Data Augmentation: {self.data_augmentation}
Save Directory: {self.savedir}
"""