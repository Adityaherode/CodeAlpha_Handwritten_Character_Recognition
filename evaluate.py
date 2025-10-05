import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, x_test, y_test, history, model_dir="mnist_model"):
    results = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {results[1]*100:.2f}%")

    # Plot training curves
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_curves.png"))
    plt.show()

    # Confusion matrix
    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.show()

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
