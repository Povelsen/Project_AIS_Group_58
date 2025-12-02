"""Pytorch implementation
A generative transformer for AIS trajectory prediction
"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
#For the default model, use the below:
# from config_files.config_trAISformer_default import Config

#To adjust the experiments, use below:
from config_files.config_file_with_FLAGS import Config


cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2

if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    print(f"Device: {device}")
    print(cf)

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        print("Creating pytorch dataset...")
    
        # FIXED: Use the enhanced dataset with port/coast features
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        elif cf.use_port_features or cf.use_coast_features:
            # Use enhanced dataset when auxiliary features are enabled
            aisdatasets[phase] = datasets.AISDataset_Enhanced(
                Data[phase],
                max_seqlen=cf.max_seqlen + 1,
                use_port_features=cf.use_port_features,
                use_coast_features=cf.use_coast_features,
                predict_ports=cf.predict_ports,
                data_augmentation=cf.data_augmentation if phase == "train" else False,
                aug_noise_std=cf.aug_noise_std,
                aug_temporal_jitter=cf.aug_temporal_jitter,
                device=cf.device)
        else:
            # Use standard dataset when features are disabled
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                 max_seqlen=cf.max_seqlen + 1,
                                                 device=cf.device)
    
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    model = models.TrAISformer(cf, partition_model=None)

    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Load the best model
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = cf.max_seqlen

    model.eval()
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    saved_trajs = []
    with torch.no_grad():
        for it, batch_data in pbar:
            if len(batch_data) == 6:
                seqs, masks, seqlens, mmsis, time_starts, port_labels = batch_data
            else:
                seqs, masks, seqlens, mmsis, time_starts = batch_data
                port_labels = None
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            for i_sample in range(cf.n_samples):
                preds, port_probs = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs[:,:,:4] * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds[:,:,:4] * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]

            best_err, best_idx = error_ens.min(dim=-1)  # (batch, seq)
        
            # Convert coords back to latitude/longitude
            pred_all = preds[:, :, :2]  # (batch, seq, lat/lon)
            true_all = inputs[:, :, :2]  # original ground truth
            
            # Convert back to degrees
            pred_all = (pred_all * v_ranges[:2] + v_roi_min[:2])
            true_all = (true_all * v_ranges[:2] + v_roi_min[:2])
            
            pred_all = pred_all.detach().cpu().numpy()
            true_all = true_all.detach().cpu().numpy()
            
            # Save FULL trajectories including initial context
            for b in range(batchsize):
                saved_trajs.append({
                    "true": true_all[b, :, :],                      # FULL ground truth trajectory
                    "pred_init": true_all[b, :cf.init_seqlen, :],   # Initial observed part (ground truth)
                    "pred_future": pred_all[b, cf.init_seqlen:, :], # Future predictions
                    "init_seqlen": cf.init_seqlen,                  # Store for visualization
                    "pred_port": torch.argmax(port_probs[b, -1]).item() if port_probs is not None else -1,
                    "true_port": port_labels[b, 0].item() if port_labels is not None else -1
                })
        
            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()
    np.save(os.path.join(cf.savedir, "errors.npy"), pred_errors)


   # =============================== PLOT ===============================
    plt.figure(figsize=(9, 6), dpi=150)

    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors, color="blue")

    err0 = pred_errors[0]
    plt.plot(0, err0, "o", color="black")
    plt.plot([0, 0], [0, err0], "r")
    plt.text(0.1, err0 - 0.3, f"{err0:.3f}", fontsize=10)

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot markers for every hour
    max_hour = int(len(pred_errors) / 6)
    for h in range(1, max_hour + 1):
        idx = int(h * 6)
        if idx < len(pred_errors):
            err = pred_errors[idx]
            
            # Color logic: Green for early, Orange for mid, Red for late
            if h <= 3: color = "green"
            elif h <= 12: color = "orange"
            else: color = "red"
            
            plt.plot(h, err, "o", color=color, markersize=4)
            plt.plot([h, h], [0, err], color=color, linestyle=":", linewidth=0.5)
            
            # Add text label (staggered to avoid overlap)
            if h <= 4 or h % 2 == 0:  # Label first 4 hours, then every 2nd hour
                y_offset = -0.3 if h % 2 == 0 else 0.3
                plt.text(h, err + y_offset, f"{err:.2f}", fontsize=8, ha='center')

    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    # plt.xlim([0, 4])  # Removed to show full duration
    # plt.ylim([0, 7.5]) # Removed to allow auto-scaling
    plt.savefig(cf.savedir + "prediction_error.png")




    # Save raw error array
    np.save(os.path.join(cf.savedir, "errors.npy"), pred_errors)
    np.save(os.path.join(cf.savedir, "trajectories.npy"), saved_trajs)
    print("Saved:", os.path.join(cf.savedir, "errors.npy"))


    # Yeah, done!!!
