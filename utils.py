import matplotlib.pyplot as plt
import numpy as np

def show_samples(x, y_true, y_pred=None, n=12):
    plt.figure(figsize=(12,6))
    idxs = np.random.choice(range(len(x)), n, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(3,4,i+1)
        plt.imshow(x[idx].squeeze(), cmap="gray")
        title = f"True: {y_true[idx]}"
        if y_pred is not None:
            title += f" / Pred: {y_pred[idx]}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
