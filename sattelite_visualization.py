import folium
import numpy as np
import webbrowser
import os

# =================== CONFIG ===================
from config_files.config_file_with_FLAGS import Config
cf = Config()

MAPBOX_TOKEN = "pk.eyJ1IjoiaXR1cmFzaSIsImEiOiJjbWllYmQ1bHkwMHkyM2ZyMWVvbHJ1a2EzIn0.QfeitLotjtKoaGsuCMDV-w"

TRAJ_PATH = os.path.join(cf.savedir, "trajectories.npy")
SAVE_HTML = os.path.join(cf.savedir, "map_vis.html")
PORTS_META_PATH = os.path.join(cf.datadir, "ports_metadata.pkl")
# ===============================================

import pickle
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Load ports metadata
ports_meta = []
if os.path.exists(PORTS_META_PATH):
    with open(PORTS_META_PATH, "rb") as f:
        ports_meta = pickle.load(f)
    print(f"Loaded {len(ports_meta)} ports metadata")
else:
    print(f"Warning: Ports metadata not found at {PORTS_META_PATH}")

# Load trajectories
data = np.load(TRAJ_PATH, allow_pickle=True)

# Pick a random sample
idx = np.random.randint(0, len(data))
sample = data[idx]

print(f"Visualizing trajectory {idx} out of {len(data)}")

# Get the full trajectories with the NEW format
true_full = sample["true"]  # Full ground truth
pred_init = sample["pred_init"]  # Initial context (observed)
pred_future = sample["pred_future"]  # Future predictions
init_len = sample["init_seqlen"]

# Combine initial context + future predictions for complete predicted trajectory
pred_full = np.vstack([pred_init, pred_future])

# Port info
pred_port_idx = sample.get("pred_port", -1)
true_port_idx = sample.get("true_port", -1)

port_accuracy_msg = "No port info"
pred_port_coords = None
true_port_coords = None

if pred_port_idx != -1 and true_port_idx != -1 and ports_meta:
    pred_port_data = next((p for p in ports_meta if p["id"] == pred_port_idx), None)
    true_port_data = next((p for p in ports_meta if p["id"] == true_port_idx), None)
    
    if pred_port_data and true_port_data:
        dist = haversine_distance(pred_port_data["lat"], pred_port_data["lon"],
                                  true_port_data["lat"], true_port_data["lon"])
        port_accuracy_msg = f"Port Dist: {dist:.2f} km"
        if pred_port_idx == true_port_idx:
            port_accuracy_msg += " (CORRECT)"
            
        pred_port_coords = [pred_port_data["lat"], pred_port_data["lon"]]
        true_port_coords = [true_port_data["lat"], true_port_data["lon"]]
        print(f"Port Prediction: {port_accuracy_msg}")
        print(f"  Predicted: Port {pred_port_idx} at {pred_port_coords}")
        print(f"  True:      Port {true_port_idx} at {true_port_coords}")

print(f"\nTrajectory info:")
print(f"  Total length: {len(true_full)} points")
print(f"  Initial context: {init_len} points")
print(f"  Future predictions: {len(pred_future)} points")
print(f"  {port_accuracy_msg}")
print(f"\nCoordinate ranges:")
print(f"  Lat: {true_full[:, 0].min():.2f} to {true_full[:, 0].max():.2f}")
print(f"  Lon: {true_full[:, 1].min():.2f} to {true_full[:, 1].max():.2f}")

# Starting location
start_lat = float(true_full[0, 0])
start_lon = float(true_full[0, 1])

print(f"\nStart position: ({start_lat:.4f}, {start_lon:.4f})")

# ======== Create Map with Satellite ========
tile_url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}"

m = folium.Map(
    location=[start_lat, start_lon],
    zoom_start=10,
    tiles=tile_url,
    attr="Mapbox"
)

# Convert to lists
true_coords = [[float(lat), float(lon)] for lat, lon in true_full]
pred_coords = [[float(lat), float(lon)] for lat, lon in pred_full]
init_coords = [[float(lat), float(lon)] for lat, lon in pred_init]
future_true_coords = [[float(lat), float(lon)] for lat, lon in true_full[init_len:]]
future_pred_coords = [[float(lat), float(lon)] for lat, lon in pred_future]

# ======== Draw INITIAL CONTEXT (observed AIS data) ========
folium.PolyLine(
    locations=init_coords,
    color="blue",
    weight=5,
    opacity=1.0,
    tooltip="Observed AIS Track"
).add_to(m)

# ======== Draw TRUE future trajectory ========
folium.PolyLine(
    locations=future_true_coords,
    color="green",
    weight=4,
    opacity=0.8,
    tooltip="Ground Truth Future"
).add_to(m)

# ======== Draw PREDICTED future trajectory ========
folium.PolyLine(
    locations=future_pred_coords,
    color="red",
    weight=4,
    opacity=0.8,
    tooltip="Predicted Future"
).add_to(m)

# ======== Mark key points ========
folium.CircleMarker(
    location=true_coords[0],
    radius=8,
    color="darkblue",
    fill=True,
    fill_color="blue",
    fill_opacity=1.0,
    tooltip="START"
).add_to(m)

folium.CircleMarker(
    location=init_coords[-1],
    radius=7,
    color="purple",
    fill=True,
    fill_color="purple",
    fill_opacity=1.0,
    tooltip="Prediction Starts Here"
).add_to(m)

folium.CircleMarker(
    location=true_coords[-1],
    radius=7,
    color="darkgreen",
    fill=True,
    fill_color="green",
    fill_opacity=0.8,
    tooltip="True End"
).add_to(m)

folium.CircleMarker(
    location=pred_coords[-1],
    radius=7,
    color="darkred",
    fill=True,
    fill_color="red",
    fill_opacity=0.8,
    tooltip="Predicted End"
).add_to(m)

# ======== Draw PORTS ========
if true_port_coords:
    folium.Marker(
        location=true_port_coords,
        popup=f"True Destination Port (ID: {true_port_idx})",
        icon=folium.Icon(color="green", icon="anchor", prefix="fa")
    ).add_to(m)

if pred_port_coords:
    color = "green" if pred_port_idx == true_port_idx else "red"
    folium.Marker(
        location=pred_port_coords,
        popup=f"Predicted Destination Port (ID: {pred_port_idx})",
        icon=folium.Icon(color=color, icon="anchor", prefix="fa")
    ).add_to(m)
    
    # Draw line between predicted and true port if they are different
    if pred_port_idx != true_port_idx:
        folium.PolyLine(
            locations=[true_port_coords, pred_port_coords],
            color="orange",
            weight=2,
            dash_array="5, 5",
            tooltip=f"Error: {port_accuracy_msg}"
        ).add_to(m)

# Add legend
legend_html = f'''
<div style="position: fixed; 
            bottom: 50px; right: 50px; width: 250px; height: 220px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
<p><strong>AIS Trajectory Prediction</strong></p>
<p><span style="color:blue;">━━━</span> Observed Track</p>
<p><span style="color:green;">━━━</span> True Future</p>
<p><span style="color:red;">━━━</span> Predicted Future</p>
<p><span style="color:purple;">●</span> Prediction Start</p>
<hr>
<p><strong>Port Accuracy:</strong></p>
<p>{port_accuracy_msg}</p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ======== Fit bounds ========
all_lats = [coord[0] for coord in true_coords]
all_lons = [coord[1] for coord in true_coords]
m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])

# ======== SAVE + OPEN ========
m.save(SAVE_HTML)
print(f"\n✓ Saved map to: {SAVE_HTML}")
webbrowser.open("file://" + os.path.abspath(SAVE_HTML))