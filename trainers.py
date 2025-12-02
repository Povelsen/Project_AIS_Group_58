"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import math
import logging
import pickle
import json

from tqdm import tqdm
import numpy as np
import pickle
import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no Qt issues)
import matplotlib.pyplot as plt



import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils

from trAISformer import TB_LOG

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        model_out = model(seqs_cond)
        if isinstance(model_out, tuple) and len(model_out) == 3:
             logits, _, port_probs = model_out
        elif isinstance(model_out, tuple):
             logits, _ = model_out
             port_probs = None
        else:
             logits = model_out
             port_probs = None
             
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes

        # Append auxiliary features from the last step if present
        if seqs.shape[-1] > 4:
            aux_feats = seqs[:, -1, 4:]
            x_sample = torch.cat((x_sample, aux_feats), dim=-1)

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)

    # Return both sequence and the last port probability distribution
    return seqs, port_probs


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None,
                 device=torch.device("cpu"), aisdls={}, INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

        # Try to load DMA coastline polygons
        self.coastline_polygons = None
        try:
            coast_path = os.path.join(self.config.datadir, "dma_coastline_polygons.pkl")
            if os.path.exists(coast_path):
                with open(coast_path, "rb") as f:
                    self.coastline_polygons = pickle.load(f)
        except Exception as e:
            logging.warning(f"Could not load coastline polygons: {e}")

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
        


    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            """Modified to handle port labels."""
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_n = 0, 0
    
            for it, batch_data in pbar:
                # Unpack batch - handle both with and without port labels
                if len(batch_data) == 6:
                    seqs, masks, seqlens, mmsis, time_starts, port_labels = batch_data
                    port_labels = port_labels.to(self.device)
                else:
                    seqs, masks, seqlens, mmsis, time_starts = batch_data
                    port_labels = None

                # Place data on correct device
                seqs = seqs.to(self.device)
                masks = masks[:, :-1].to(self.device)

                # --- Curriculum Learning Logic ---
                if is_train and hasattr(config, 'curriculum_learning') and config.curriculum_learning:
                    # Determine max_seqlen for this epoch
                    curr_max_seqlen = config.max_seqlen # Default to full
                    for ep_limit, seq_limit in config.curriculum_schedule:
                        if epoch <= ep_limit:
                            curr_max_seqlen = seq_limit
                            break
                    
                    # Crop sequences if needed
                    if seqs.shape[1] > curr_max_seqlen:
                        seqs = seqs[:, :curr_max_seqlen, :]
                        masks = masks[:, :curr_max_seqlen-1] # mask is 1 shorter than seq (targets)
                        if port_labels is not None:
                            port_labels = port_labels[:, :curr_max_seqlen]
                # ---------------------------------

                # Forward the model
                with torch.set_grad_enabled(is_train):
                    if model.predict_ports:
                        logits, loss, port_probs = model(seqs, 
                                                 masks=masks, 
                                                 with_targets=True,
                                                 port_labels=port_labels)
                    else:
                        logits, loss = model(seqs, masks=masks, with_targets=True)
            
                    loss = loss.mean()
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
        
                if is_train:
                    # Backprop and update
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    # tb logging
                    if TB_LOG:
                        tb.add_scalar("loss",
                                      loss.item(),
                                      epoch * n_batches + it)
                        tb.add_scalar("lr",
                                      lr,
                                      epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            # Average loss over all samples in this epoch
            epoch_loss = d_loss / d_n

            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {epoch_loss:.5f}, {d_reg_loss / d_n:.5f}, lr {lr:e}."
                    )
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {epoch_loss:.5f}, lr {lr:e}."
                    )
            else:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {epoch_loss:.5f}."
                    )
                else:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {epoch_loss:.5f}."
                    )

            # Return epoch loss for BOTH train and valid
            return float(epoch_loss)


        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        train_losses = []
        valid_losses = []

        # Early stopping variables
        epochs_without_improvement = 0
        early_stop_triggered = False

        for epoch in range(config.max_epochs):

            train_loss = run_epoch('Training', epoch=epoch)
            
            # --- Curriculum Update Check ---
            # Check if curriculum advanced this epoch
            if hasattr(config, 'curriculum_learning') and config.curriculum_learning:
                # Calculate what the max_seqlen was for this epoch
                curr_max_seqlen = config.max_seqlen
                for ep_limit, seq_limit in config.curriculum_schedule:
                    if epoch <= ep_limit:
                        curr_max_seqlen = seq_limit
                        break
                
                # If it changed from the previous epoch (and it's not the first one)
                if epoch > 0 and curr_max_seqlen != prev_max_seqlen:
                    logging.info(f"Curriculum advanced (seqlen {prev_max_seqlen} -> {curr_max_seqlen}). Resetting early stopping patience.")
                    epochs_without_improvement = 0
                
                prev_max_seqlen = curr_max_seqlen
            else:
                prev_max_seqlen = None
            # -------------------------------

            if self.test_dataset is not None:
                valid_loss = run_epoch('Valid', epoch=epoch)
            else:
                valid_loss = None

            train_losses.append(train_loss)
            if valid_loss is not None:
                valid_losses.append(valid_loss)

            # Early stopping / best checkpoint logic
            if self.test_dataset is not None and config.early_stopping:
                # Check if validation loss improved
                if valid_loss < (best_loss - config.min_delta):
                    best_loss = valid_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    if self.config.ckpt_path is not None:
                        self.save_checkpoint(best_epoch + 1)
                    logging.info(f"Validation loss improved! New best: {best_loss:.5f}")
                else:
                    epochs_without_improvement += 1
                    logging.info(f"No improvement for {epochs_without_improvement}/{config.patience} epochs")
                    
                    if epochs_without_improvement >= config.patience:
                        logging.info(f"Early stopping triggered! Best epoch was {best_epoch + 1}")
                        early_stop_triggered = True
                        break
            else:
                # Original logic without early stopping
                good_model = self.test_dataset is None or valid_loss < best_loss
                if self.config.ckpt_path is not None and good_model:
                    best_loss = valid_loss if valid_loss is not None else train_loss
                    best_epoch = epoch
                    self.save_checkpoint(best_epoch + 1)

            ## SAMPLE AND PLOT
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            batch_data = next(iter(aisdls["test"]))
            if len(batch_data) == 6:
                seqs, masks, seqlens, mmsis, time_starts, port_labels = batch_data
            else:
                seqs, masks, seqlens, mmsis, time_starts = batch_data
            n_plots = 7
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            preds, port_probs = sample(raw_model,
                        seqs_init,
                        96 - init_seqlen,
                        temperature=1.0,
                        sample=False, # True for sampling, False for argmax
                        sample_mode=self.config.sample_mode,
                        r_vicinity=self.config.r_vicinity,
                        top_k=self.config.top_k)

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')

            # Convert normalized [0,1) lat/lon back to degrees
            lat_min, lat_max = self.config.lat_min, self.config.lat_max
            lon_min, lon_max = self.config.lon_min, self.config.lon_max

            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()

            # Set up Cartopy map
            proj = ccrs.PlateCarree()
            fig = plt.figure(figsize=(9, 6), dpi=150)
            ax = plt.axes(projection=proj)

            # Focus on ROI
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
            ax.coastlines(resolution="10m", linewidth=0.5)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            # Add DMA coastline polygons if available
            if self.coastline_polygons is not None:
                for poly in self.coastline_polygons:
                    # poly[:,0] = lat, poly[:,1] = lon
                    ax.plot(poly[:, 1], poly[:, 0],
                            transform=proj, linewidth=0.4, color="grey", alpha=0.7)

            cmap = plt.cm.get_cmap("jet")
            n_plots = min(n_plots, seqs.size(0))

            for idx in range(n_plots):
                c = cmap(float(idx) / max(n_plots - 1, 1))
                seqlen = seqlens[idx].item()

                # De-normalize ground-truth
                lat_true = inputs_np[idx][:seqlen, 0] * (lat_max - lat_min) + lat_min
                lon_true = inputs_np[idx][:seqlen, 1] * (lon_max - lon_min) + lon_min

                # De-normalize predicted full sequence
                lat_pred = preds_np[idx][:, 0] * (lat_max - lat_min) + lat_min
                lon_pred = preds_np[idx][:, 1] * (lon_max - lon_min) + lon_min

                # Prefix region
                lat_prefix = lat_true[:init_seqlen]
                lon_prefix = lon_true[:init_seqlen]

                # Plot input prefix
                ax.plot(lon_prefix, lat_prefix,
                        "o-", markersize=3, linewidth=1.0,
                        color=c, transform=proj, label=None)

                # Plot full ground-truth trajectory
                ax.plot(lon_true, lat_true,
                        "-.", linewidth=0.8,
                        color=c, transform=proj)

                # Plot predicted continuation (from init_seqlen onward)
                ax.plot(lon_pred[init_seqlen:], lat_pred[init_seqlen:],
                        "x-", markersize=4, linewidth=0.8,
                        color=c, transform=proj)

            plt.title(f"Predicted trajectories (epoch {epoch + 1})")
            plt.savefig(img_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # AFTER the loop ends, add this:

        # Plot training vs validation loss curves
        if len(train_losses) > 0 and len(valid_losses) > 0:
            epochs_range = range(1, len(train_losses) + 1)
            plt.figure(figsize=(8, 5), dpi=150)
            plt.plot(epochs_range, train_losses, label="Train loss")
            plt.plot(epochs_range, valid_losses, label="Valid loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            curve_path = os.path.join(self.savedir, "loss_curve.png")
            plt.savefig(curve_path, dpi=150)
            plt.close()

        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Last epoch: {epoch + 1:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)

        # Save train/val loss history
        history = {
            "train_loss": train_losses,
            "val_loss": valid_losses if len(valid_losses) > 0 else None
        }

        # Save as pickle
        with open(os.path.join(self.savedir, "history.pkl"), "wb") as f:
            pickle.dump(history, f)

        # Save as JSON
        with open(os.path.join(self.savedir, "config.json"), "w") as f:
            json.dump(vars(self.config), f, indent=4)

        # Create /plots folder for visualizations
        plot_dir = os.path.join(self.savedir, "plots")
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

