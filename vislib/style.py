# vislib/style.py
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    plt.style.use("default")
    sns.set_palette("husl")
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
    })
