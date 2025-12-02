import numpy as np
import os
from config_files.config_file_with_FLAGS import Config

cf = Config()
TRAJ_PATH = os.path.join(cf.savedir, "trajectories.npy")

print(f"Loading {TRAJ_PATH}...")
if not os.path.exists(TRAJ_PATH):
    print("Error: File not found!")
    exit()

data = np.load(TRAJ_PATH, allow_pickle=True)
print(f"Loaded {len(data)} trajectories.")

# Define the WRONG transformation (that was hardcoded)
# lat = norm * 2 + 50
# lon = norm * 3 - 7
WRONG_LAT_RANGE = 2.0
WRONG_LAT_MIN = 50.0
WRONG_LON_RANGE = 3.0
WRONG_LON_MIN = -7.0

# Define the CORRECT transformation (from config)
CORRECT_LAT_RANGE = cf.lat_max - cf.lat_min
CORRECT_LAT_MIN = cf.lat_min
CORRECT_LON_RANGE = cf.lon_max - cf.lon_min
CORRECT_LON_MIN = cf.lon_min

print(f"Correcting coordinates...")
print(f"  Lat: norm * {WRONG_LAT_RANGE} + {WRONG_LAT_MIN}  ->  norm * {CORRECT_LAT_RANGE} + {CORRECT_LAT_MIN}")
print(f"  Lon: norm * {WRONG_LON_RANGE} + {WRONG_LON_MIN}  ->  norm * {CORRECT_LON_RANGE} + {CORRECT_LON_MIN}")

count = 0
for i in range(len(data)):
    # Keys: 'true', 'pred_init', 'pred_future'
    for key in ['true', 'pred_init', 'pred_future']:
        if key in data[i]:
            arr = data[i][key]
            
            # 1. Normalize (Undo Wrong)
            norm_lat = (arr[:, 0] - WRONG_LAT_MIN) / WRONG_LAT_RANGE
            norm_lon = (arr[:, 1] - WRONG_LON_MIN) / WRONG_LON_RANGE
            
            # 2. Denormalize (Apply Correct)
            new_lat = norm_lat * CORRECT_LAT_RANGE + CORRECT_LAT_MIN
            new_lon = norm_lon * CORRECT_LON_RANGE + CORRECT_LON_MIN
            
            # Update
            arr[:, 0] = new_lat
            arr[:, 1] = new_lon
            data[i][key] = arr
            
    count += 1

# Save back
BACKUP_PATH = TRAJ_PATH + ".bak"
if not os.path.exists(BACKUP_PATH):
    os.rename(TRAJ_PATH, BACKUP_PATH)
    print(f"Backed up original file to {BACKUP_PATH}")

np.save(TRAJ_PATH, data)
print(f"Successfully corrected {count} trajectories and saved to {TRAJ_PATH}")
