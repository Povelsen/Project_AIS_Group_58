import os
import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

#For the default model, use the below:
# from config_files.config_trAISformer_default import Config

#To adjust the experiments, use below:
from config_files.config_file_with_FLAGS import Config
from models import TrAISformer
from datasets import AISDataset, AISDataset_Enhanced
from trainers import sample
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

if __name__ == "__main__":
    cf = Config()

    # --- Load test dataset ---
    test_pkl = os.path.join(cf.datadir, cf.testset_name)
    with open(test_pkl, "rb") as f:
        data = pickle.load(f)
    data = [x for x in data if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
    
    if cf.use_port_features or cf.use_coast_features:
        test_dataset = AISDataset_Enhanced(
            data, 
            max_seqlen=cf.max_seqlen + 1,
            use_port_features=cf.use_port_features,
            use_coast_features=cf.use_coast_features,
            predict_ports=cf.predict_ports,
            device=cf.device
        )
    else:
        test_dataset = AISDataset(data, max_seqlen=cf.max_seqlen + 1)

    # Pick one trajectory
    item = test_dataset[0]
    if len(item) == 6:
        seq, mask, seqlen, mmsi, time_start, port_labels = item
    else:
        seq, mask, seqlen, mmsi, time_start = item
        port_labels = None
        
    seq = seq.unsqueeze(0)  # (1, T, 4)

    # --- Load trained model ---
    device = cf.device
    model = TrAISformer(cf)
    state = torch.load(cf.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    init_seqlen = cf.init_seqlen
    steps = cf.max_seqlen - cf.init_seqlen  # predict full future steps

    with torch.no_grad():
        preds, port_probs = sample(
            model,
            seq[:, :init_seqlen].to(device),
            steps=steps,
            temperature=1.0,
            sample=True,
            sample_mode=cf.sample_mode,
            r_vicinity=cf.r_vicinity,
            top_k=cf.top_k,
        )

    preds_np = preds.squeeze(0).cpu().numpy()
    inputs_np = seq.squeeze(0).cpu().numpy()

    lat_min, lat_max = cf.lat_min, cf.lat_max
    lon_min, lon_max = cf.lon_min, cf.lon_max

    # De-normalize
    lat_true = inputs_np[:seqlen, 0] * (lat_max - lat_min) + lat_min
    lon_true = inputs_np[:seqlen, 1] * (lon_max - lon_min) + lon_min

    lat_pred = preds_np[:, 0] * (lat_max - lat_min) + lat_min
    lon_pred = preds_np[:, 1] * (lon_max - lon_min) + lon_min

    # Optional: load DMA coastline polygons
    coast_path = os.path.join(cf.datadir, "dma_coastline_polygons.pkl")
    coastline_polygons = None
    if os.path.exists(coast_path):
        with open(coast_path, "rb") as f:
            coastline_polygons = pickle.load(f)

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 10))
    
    # Use Google Tiles for satellite imagery
    tiler = cimgt.GoogleTiles(style='satellite')
    proj = tiler.crs

    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    # Set extent to cover the trajectory and some surroundings
    # We need to transform the extent to the projection of the tiles
    margin = 2.0
    WEST = min(lon_min, np.min(lon_true), np.min(lon_pred))
    EAST = max(lon_max, np.max(lon_true), np.max(lon_pred))
    SOUTH = min(lat_min, np.min(lat_true), np.min(lat_pred))
    NORTH = max(lat_max, np.max(lat_true), np.max(lat_pred))
    extent = [WEST - margin, EAST + margin, SOUTH - margin, NORTH + margin]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add satellite image
    ax.add_image(tiler, 8) # Zoom level 8

    # Add features (coastlines might be redundant with satellite, but good for clarity)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='yellow')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='yellow')
    
    # Plot ALL ports
    ports_meta_path = os.path.join(cf.datadir, "ports_metadata.pkl")
    ports_meta = []
    if os.path.exists(ports_meta_path):
        with open(ports_meta_path, "rb") as f:
            ports_meta = pickle.load(f)
        
        if ports_meta:
            port_lats = [p["lat"] for p in ports_meta]
            port_lons = [p["lon"] for p in ports_meta]
            ax.plot(port_lons, port_lats, '.', color='cyan', markersize=3, transform=ccrs.PlateCarree(), label="Ports", alpha=0.5)

    # Plot trajectory
    # Note: transform=ccrs.PlateCarree() is needed for lat/lon data
    ax.plot(lon_true[:init_seqlen], lat_true[:init_seqlen],
            "o-", markersize=3, linewidth=1.0, color="white", transform=ccrs.PlateCarree(), label="Observed")
    ax.plot(lon_true[init_seqlen:], lat_true[init_seqlen:],
            "o-", markersize=3, linewidth=1.0, color="tab:green", transform=ccrs.PlateCarree(), label="True future")
    ax.plot(lon_pred[init_seqlen:], lat_pred[init_seqlen:],
            "x-", markersize=5, linewidth=1.0, color="tab:red", transform=ccrs.PlateCarree(), label="Predicted future")

    # Port Accuracy Visualization
    title_extra = ""
    if port_probs is not None and port_labels is not None:
        pred_port_idx = torch.argmax(port_probs[0, -1]).item()
        true_port_idx = port_labels[0].item()
        
        # Load ports metadata (already loaded above if exists)
        if ports_meta:
            pred_port_data = next((p for p in ports_meta if p["id"] == pred_port_idx), None)
            true_port_data = next((p for p in ports_meta if p["id"] == true_port_idx), None)
            
            if pred_port_data and true_port_data:
                dist = haversine_distance(pred_port_data["lat"], pred_port_data["lon"],
                                          true_port_data["lat"], true_port_data["lon"])
                title_extra = f"\nPort Acc: {dist:.2f} km (Pred: {pred_port_idx}, True: {true_port_idx})"
                
                # Plot ports
                ax.plot(true_port_data["lon"], true_port_data["lat"], "P", color="lime", markersize=10, markeredgecolor='black', transform=ccrs.PlateCarree(), label="True Dest")
                ax.plot(pred_port_data["lon"], pred_port_data["lat"], "X", color="red", markersize=10, markeredgecolor='black', transform=ccrs.PlateCarree(), label="Pred Dest")

    plt.legend()
    plt.title(f"True vs predicted ship trajectory (Cartopy map){title_extra}")
    out_path = os.path.join(cf.savedir, "cartopy_example.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Cartopy map to: {out_path}")
