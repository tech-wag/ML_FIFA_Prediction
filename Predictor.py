import pickle
import pandas as pd
from DataLoader import TARGET_LABELS, load_match_data, build_single_match_features, FEATURE_COLUMNS, RAG_COLUMNS
from rag_enrich import enrich_match

MODEL_PATH = "fifa_model.pkl"


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def predict_match(home_team, away_team, tournament, neutral=False, model_path=MODEL_PATH):
    artifact = load_model(model_path)
    model = artifact["model"]
    encoders = artifact["encoders"]
    df = load_match_data()

    features = build_single_match_features(home_team, away_team, tournament, neutral, df, encoders)
    try:
        rag_features = enrich_match(home_team, away_team)
    except Exception as exc:
        rag_features = {col: 0.0 for col in RAG_COLUMNS}
        print(f"Warning: failed to fetch RAG features: {exc}")

    features += [rag_features.get(col, 0.0) for col in RAG_COLUMNS]
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    return {
        "prediction": int(prediction),
        "label": TARGET_LABELS[prediction],
        "probabilities": {
            TARGET_LABELS[i]: float(probabilities[idx])
            for idx, i in enumerate(sorted(TARGET_LABELS))
        },
        "features": {
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "neutral": bool(neutral),
            "home_team_strength": features[4],
            "away_team_strength": features[5],
            "h2h_home_wins": features[10],
            "h2h_draws": features[11],
            "h2h_away_wins": features[12],
            "h2h_goal_diff": features[13],
            **{col: features[14 + idx] for idx, col in enumerate(RAG_COLUMNS)},
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict FIFA match outcomes from a trained Random Forest.")
    parser.add_argument("home_team", help="Home team name")
    parser.add_argument("away_team", help="Away team name")
    parser.add_argument("tournament", help="Tournament name")
    parser.add_argument("--neutral", action="store_true", help="Set if the match is at a neutral venue")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained model file")

    args = parser.parse_args()
    result = predict_match(args.home_team, args.away_team, args.tournament, args.neutral, args.model)
    print("Prediction:", result["label"])
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label}: {prob:.2%}")
