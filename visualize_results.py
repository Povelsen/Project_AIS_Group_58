import os, numpy as np

# ───── Select experiment manually ─────
EXP_NAME = "long_run_curriculum_v2/ct_dma_pos_bs64_lr0.0001"      # CHANGE this 👈
# e.g. "early_stop", "lower_lr", "high_dropout", etc.

# ───── Choose config if needed ─────
# from config_files.config_trAISformer_default import Config
from config_files.config_file_with_FLAGS import Config

from vislib.evaluation import plot_error_hist, plot_error_timeline
from vislib.curves import plot_loss_curve

# Load config (only to know dataset info etc.)
cf = Config()

# ───── Build project-root path ─────
project_root = os.path.dirname(os.path.abspath(__file__))

# Use selected experiment name instead of cf.savedir
savedir = os.path.join(project_root, "results", EXP_NAME)

print(f"📍 Visualizing results in: {savedir}")

# ─────────────────────────── Errors ────────────────────────────
errors_path = os.path.join(savedir, "errors.npy")
if os.path.exists(errors_path):
    errors = np.load(errors_path)
    plot_error_hist(errors, savedir)
    plot_error_timeline(errors, savedir)
else:
    print("⚠️ No errors.npy found, skipping error plots")

# ─────────────────────────── Loss Curves ─────────────────────────
history_path = os.path.join(savedir, "history.pkl")
if os.path.exists(history_path):
    plot_loss_curve(history_path, savedir)
else:
    print("⚠️ No history.pkl found, skipping loss curves")

print("🎉 Visualization complete! Check folder:", savedir)
