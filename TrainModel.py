import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from DataLoader import load_match_data, prepare_train_test, TARGET_LABELS, FEATURE_COLUMNS

MODEL_PATH = "fifa_model.pkl"
DATA_PATH = "results.csv"


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = load_match_data(data_path)
    X_train, X_test, y_train, y_test, encoders = prepare_train_test(df)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[TARGET_LABELS[i] for i in sorted(TARGET_LABELS)]))

    feature_importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    print(feature_importances)

    artifact = {
        "model": model,
        "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS,
        "target_labels": TARGET_LABELS,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved trained model and metadata to {model_path}")
    return artifact


if __name__ == "__main__":
    train_model()
