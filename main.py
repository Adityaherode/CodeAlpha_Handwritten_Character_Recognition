from data_loader import load_data
from model_builder import build_model
from train import train_model
from evaluate import evaluate_model

def main():
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Build model
    model = build_model()
    model.summary()

    # Train model
    history = train_model(model, x_train, y_train, x_val, y_val)

    # Evaluate model
    evaluate_model(model, x_test, y_test, history)

if __name__ == "__main__":
    main()
