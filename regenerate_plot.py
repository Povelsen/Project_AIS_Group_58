import numpy as np
import matplotlib.pyplot as plt
import os
from config_files.config_file_with_FLAGS import Config

cf = Config()
errors_path = os.path.join(cf.savedir, "errors.npy")

if os.path.exists(errors_path):
    print(f"Loading errors from {errors_path}")
    pred_errors = np.load(errors_path)
    
    plt.figure(figsize=(10, 6), dpi=150) # Slightly wider

    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors, color="blue", linewidth=1.5)

    err0 = pred_errors[0]
    plt.plot(0, err0, "o", color="black")
    plt.text(0.1, err0 + 0.1, f"{err0:.2f}", fontsize=8)

    # Add grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

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
                y_offset = -0.5 if h % 2 == 0 else 0.3
                plt.text(h, err + y_offset, f"{err:.2f}", fontsize=8, ha='center', fontweight='bold')

    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.title(f"Prediction Error over Time ({max_hour} hours)")
    
    out_path = os.path.join(cf.savedir, "prediction_error_full.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved updated plot to {out_path}")
else:
    print(f"File not found: {errors_path}")
