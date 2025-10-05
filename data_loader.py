import numpy as np
from tensorflow import keras

def load_data(val_split=0.1):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize and expand channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Split validation set
    val_count = int(val_split * x_train.shape[0])
    x_val = x_train[:val_count]
    y_val = y_train[:val_count]
    x_train = x_train[val_count:]
    y_train = y_train[val_count:]

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
