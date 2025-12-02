import matplotlib.pyplot as plt
import numpy as np
import os

def plot_error_hist(errors, savedir):
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=40, color='skyblue', edgecolor='black')
    plt.xlabel("Prediction Error (km)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(savedir, "error_histogram.png"), dpi=200)
    plt.close()

def plot_error_timeline(errors, savedir):
    plt.figure(figsize=(6,4))
    plt.plot(errors, marker='.')
    plt.xlabel("Time Horizon (steps)")
    plt.ylabel("Average Error (km)")
    plt.title("Prediction Error Over Time")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(savedir, "error_timeline.png"), dpi=200)
    plt.close()
