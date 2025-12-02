# vislib/curves.py
import matplotlib.pyplot as plt
import os
from .style import set_style


import matplotlib.pyplot as plt
import pickle
import os

def plot_loss_curve(history_path, savedir):
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    train, valid = history["train_loss"], history["val_loss"]
    epochs = range(1, len(train) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, train, label="Train Loss", marker='o')
    plt.plot(epochs, valid, label="Valid Loss", marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(savedir, "loss_curve.png"), dpi=200)
    plt.close()


def plot_training(history: dict, save_dir: str, name="Model"):
    set_style()
    train = history.get("train_loss")
    val = history.get("val_loss")

    epochs = range(1, len(train)+1)
    plt.figure(figsize=(7,5))
    plt.plot(epochs, train, label="Train", linewidth=2)
    if val:
        plt.plot(epochs, val, label="Validation", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} — Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{name}_training_curve.png")
    plt.close()


