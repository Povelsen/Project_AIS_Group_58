import os
import glob
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.neighbors import BallTree

# ================================
# CONFIGURATION
# ================================
NORTH, WEST, SOUTH, EAST = 60.0, 0.0, 50.0, 20.0
LAT_MIN, LAT_MAX = SOUTH, NORTH
LON_MIN, LON_MAX = WEST, EAST

# Cleaning Parameters (Restored from original)
SOG_MAX_FILTER = 30.0
EMPIRICAL_SPEED_MAX = 40.0 # knots
MIN_POINTS = 20
MIN_DURATION_SEC = 4 * 3600
MAX_GAP_SEC = 2 * 3600
MAX_DURATION_SEC = 20 * 3600
RESAMPLE_SEC = 600

# Feature Engineering Config
MAX_COAST_DIST_M = 50000.0  # Normalize distances up to 50km
INPUT_FOLDER = "/work3/s214381/AIS_data/raw_data_csv"
OUTPUT_FOLDER = "/work3/s214381/AIS_data/cleaned_ais_data" #/work3/s214381/AIS_data/raw_data_csv
COASTLINE_PKL = "./data/ct_dma/dma_coastline_polygons.pkl"
PORT_CSV = "./data/port_locodes.csv" #/zhome/d6/b/167576/projekt_AIS_group58/data/port_locodes.csv

EARTH_RADIUS_M = 6371000.0
NM_IN_M = 1852.0

# ================================
# HELPERS
# ================================

def haversine_m(lat1, lon1, lat2, lon2):
    """Calculates distance in meters (Vectorized)."""
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c

def load_all_csv(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files: raise RuntimeError(f"No CSVs in {folder}")
    df_list = []
    for f in tqdm(files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.replace(r"^#\s*", "", regex=True)
            # Preprocess IMMEDIATELY to save memory
            df = preprocess_df(df)
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def load_ports(csv_path):
    if not os.path.exists(csv_path): return pd.DataFrame()
    df = pd.read_csv(csv_path, sep=";", header=None)
    df.columns = ["Name", "LOCODE", "Coords"]
    valid_ports = []
    for _, row in df.iterrows():
        try:
            points = []
            for p in str(row["Coords"]).split(","):
                lon, lat = map(float, p.strip().split())
                points.append((lat, lon))
            if not points: continue
            pts = np.array(points)
            c_lat, c_lon = pts[:,0].mean(), pts[:,1].mean()
            if (SOUTH-1 <= c_lat <= NORTH+1) and (WEST-1 <= c_lon <= EAST+1):
                valid_ports.append({"id": len(valid_ports), "lat": c_lat, "lon": c_lon})
        except: continue
    
    # Save metadata for model reference
    with open(os.path.join(OUTPUT_FOLDER, "ports_metadata.pkl"), "wb") as f:
        pickle.dump(valid_ports, f)
    return pd.DataFrame(valid_ports)

def prepare_kd_trees(coast_pkl, port_df):
    trees = {'coast': None, 'port': None}
    
    # Coast Tree
    if os.path.exists(coast_pkl):
        with open(coast_pkl, "rb") as f: polys = pickle.load(f)
        coast_points = np.vstack(polys)
        trees['coast'] = BallTree(np.radians(coast_points), metric='haversine')
        print(f"Coast Tree built: {len(coast_points)} points")
    
    # Port Tree
    if not port_df.empty:
        trees['port'] = BallTree(np.radians(port_df[['lat', 'lon']].values), metric='haversine')
        print(f"Port Tree built: {len(port_df)} points")
        
    return trees

def preprocess_df(df):
    # 1. Select ONLY needed columns immediately
    keep_cols = ["MMSI", "Timestamp", "Latitude", "Longitude", "SOG", "COG", "Type of mobile", "Ship type", "Navigational status"]
    # Filter columns that exist in the dataframe
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols]

    # Standard filtering from original paper
    if "Type of mobile" in df.columns:
        df = df[df["Type of mobile"] == "Class A"]
    if "Ship type" in df.columns:
        df = df[df["Ship type"].astype(str).str.contains("Cargo", na=False)]
    
    # Drop filtering columns now that we're done with them
    df = df.drop(columns=["Type of mobile", "Ship type"], errors="ignore")

    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df = df[df["SOG"] < SOG_MAX_FILTER]
    
    if "Navigational status" in df.columns:
        df = df[~df["Navigational status"].isin(["Moored", "At anchor", "Not under command"])]
        df = df.drop(columns=["Navigational status"]) # Drop after use

    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[(df["Latitude"] <= NORTH) & (df["Latitude"] >= SOUTH) & 
            (df["Longitude"] >= WEST) & (df["Longitude"] <= EAST)]
    
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["unix"] = df["Timestamp"].astype("int64") // 10**9
    
    # Drop original Timestamp object column to save memory
    df = df.drop(columns=["Timestamp"])

    # Downcast types to save memory
    df["Latitude"] = df["Latitude"].astype(np.float32)
    df["Longitude"] = df["Longitude"].astype(np.float32)
    df["SOG"] = df["SOG"].astype(np.float32)
    df["COG"] = df["COG"].astype(np.float32)
    df["MMSI"] = df["MMSI"].astype(np.int32)
    
    return df

# ================================
# TRAJECTORY BUILDING (ROBUST)
# ================================

def build_enriched_trajectories(df, trees):
    trajs = []
    max_points_window = int(MAX_DURATION_SEC // RESAMPLE_SEC) + 1
    
    grouped = df.groupby("MMSI")
    for mmsi, g in tqdm(grouped, desc="Processing Trajectories"):
        g = g.sort_values("unix")
        
        # 1. Resample
        # Since we dropped Timestamp, we need to recreate it temporarily for resampling OR use unix
        # Recreating from unix is safer/easier for pandas resampling
        g["Timestamp"] = pd.to_datetime(g["unix"], unit="s")
        g_res = g.set_index("Timestamp").resample(f"{RESAMPLE_SEC}s").first().dropna(subset=["Latitude", "Longitude", "SOG", "COG"])
        if len(g_res) < 2: continue
        g_res["unix"] = g_res.index.astype(np.int64) // 10**9
        
        # 2. Empirical Speed Check (Restored)
        lat, lon = g_res["Latitude"].values, g_res["Longitude"].values
        t = g_res["unix"].values
        dists_m = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt_hours = np.diff(t).astype(float) / 3600.0
        dt_hours[dt_hours == 0] = np.nan
        speed_kn = (dists_m / NM_IN_M) / dt_hours
        
        ok = np.ones(len(g_res), dtype=bool)
        ok[1:] &= (speed_kn <= EMPIRICAL_SPEED_MAX) | np.isnan(speed_kn)
        g_res = g_res[ok]
        if len(g_res) < MIN_POINTS: continue

        # 3. Split by Time Gap
        times = g_res["unix"].values
        splits = np.where(np.diff(times) > MAX_GAP_SEC)[0] + 1
        segments = np.split(g_res, splits)
        
        for seg in segments:
            if len(seg) < MIN_POINTS: continue
            
            # 4. Split by Duration (Restored)
            seg_times = seg["unix"].values
            total_dur = seg_times[-1] - seg_times[0]
            if total_dur < MIN_DURATION_SEC: continue
            
            # Sub-segment logic
            sub_segments = []
            if total_dur <= MAX_DURATION_SEC:
                sub_segments.append(seg)
            else:
                seg = seg.reset_index(drop=True)
                for start in range(0, len(seg), max_points_window):
                    sub = seg.iloc[start : start + max_points_window]
                    if len(sub) < MIN_POINTS: continue
                    sub_t = sub["unix"].values
                    if sub_t[-1] - sub_t[0] >= MIN_DURATION_SEC:
                        sub_segments.append(sub)
            
            # 5. Feature Engineering (New)
            for sub_seg in sub_segments:
                # Query Trees
                q_pts = np.radians(sub_seg[['Latitude', 'Longitude']].values)
                
                # Coast Feature
                coast_feat = np.zeros(len(sub_seg))
                if trees['coast']:
                    dist_rad, _ = trees['coast'].query(q_pts, k=1)
                    coast_feat = np.clip((dist_rad.flatten() * EARTH_RADIUS_M) / MAX_COAST_DIST_M, 0, 1)

                # Port Feature
                port_feat = np.zeros(len(sub_seg))
                if trees['port']:
                    _, idxs = trees['port'].query(q_pts, k=1)
                    port_feat = idxs.flatten().astype(float)

                # Normalize & Pack
                lat_n = np.clip((sub_seg["Latitude"].values - LAT_MIN)/(LAT_MAX - LAT_MIN), 0, 0.9999)
                lon_n = np.clip((sub_seg["Longitude"].values - LON_MIN)/(LON_MAX - LON_MIN), 0, 0.9999)
                sog_n = np.clip(sub_seg["SOG"].values / SOG_MAX_FILTER, 0, 0.9999)
                cog_n = np.clip(sub_seg["COG"].values / 360.0, 0, 0.9999)
           
                # Find nearest port to FINAL position
                final_pos = np.radians([[sub_seg.iloc[-1]["Latitude"], 
                                         sub_seg.iloc[-1]["Longitude"]]])
                _, nearest_port_idx = trees['port'].query(final_pos, k=1)
                destination_port = nearest_port_idx[0, 0]
            
                # Create port labels: all timesteps have same destination
                port_labels = np.full(len(sub_seg), destination_port, dtype=np.int32)
            
                traj_arr = np.column_stack([
                    lat_n, lon_n, sog_n, cog_n,
                    sub_seg["unix"].values,
                    np.full(len(sub_seg), mmsi),
                    port_feat,    # Col 6
                    coast_feat,   # Col 7
                    port_labels   # Col 8: destination port label
                ])
                
                trajs.append({"mmsi": int(mmsi), "traj": traj_arr})

    return trajs

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("1. Loading Data...")
    print("1. Loading Data...")
    df = load_all_csv(INPUT_FOLDER)
    # df = preprocess_df(df)  <-- Moved inside load_all_csv
    
    print("2. Preparing Trees...")
    port_df = load_ports(PORT_CSV)
    trees = prepare_kd_trees(COASTLINE_PKL, port_df)
    
    print("3. Building Enriched Trajectories...")
    all_trajs = build_enriched_trajectories(df, trees)
    
    np.random.shuffle(all_trajs)
    N = len(all_trajs)
    n_train = int(0.7 * N)
    n_valid = int(0.15 * N)
    
    print(f"Saving {N} trajectories...")
    with open(f"{OUTPUT_FOLDER}/ct_dma_train.pkl", "wb") as f:
        pickle.dump(all_trajs[:n_train], f)
    with open(f"{OUTPUT_FOLDER}/ct_dma_valid.pkl", "wb") as f:
        pickle.dump(all_trajs[n_train:n_train+n_valid], f)
    with open(f"{OUTPUT_FOLDER}/ct_dma_test.pkl", "wb") as f:
        pickle.dump(all_trajs[n_train+n_valid:], f)
        
    print("Done!")