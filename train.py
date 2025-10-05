import os
from tensorflow import keras
from tensorflow.keras import callbacks

def train_model(model, x_train, y_train, x_val, y_val, model_dir="mnist_model", epochs=20, batch_size=128):
    os.makedirs(model_dir, exist_ok=True)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )

    earlystop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, reduce_lr_cb, earlystop_cb],
        verbose=2
    )

    model.save(os.path.join(model_dir, "final_model.h5"))
    return history
