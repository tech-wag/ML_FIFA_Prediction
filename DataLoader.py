import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_LABELS = {
    1: "Home Win",
    0: "Draw",
    2: "Away Win",
}

RAG_COLUMNS = [
    "home_articles",
    "away_articles",
    "home_injury_mentions",
    "away_injury_mentions",
    "home_kw_injury",
    "home_kw_injured",
    "home_kw_squad",
    "home_kw_suspension",
    "home_kw_suspended",
    "home_kw_out",
    "home_kw_doubt",
    "home_kw_retire",
    "home_kw_withdraw",
    "away_kw_injury",
    "away_kw_injured",
    "away_kw_squad",
    "away_kw_suspension",
    "away_kw_suspended",
    "away_kw_out",
    "away_kw_doubt",
    "away_kw_retire",
    "away_kw_withdraw",
]

FEATURE_COLUMNS = [
    "home_team_code",
    "away_team_code",
    "tournament_code",
    "neutral",
    "home_team_strength",
    "away_team_strength",
    "home_team_goal_diff",
    "away_team_goal_diff",
    "home_team_win_rate",
    "away_team_win_rate",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_goal_diff",
] + RAG_COLUMNS


def load_match_data(path="results.csv", year_cutoff=2000):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].dt.year >= year_cutoff]
    df = df.dropna(subset=["home_team", "away_team", "tournament", "neutral", "home_score", "away_score"])
    df["neutral"] = df["neutral"].astype(int)

    def determine_outcome(row):
        if row["home_score"] > row["away_score"]:
            return 1
        if row["home_score"] == row["away_score"]:
            return 0
        return 2

    df["outcome"] = df.apply(determine_outcome, axis=1)
    return df


def build_encoders(df):
    encoders = {}
    for col in ["home_team", "away_team", "tournament"]:
        categories = sorted(df[col].dropna().unique())
        encoders[col] = {value: idx for idx, value in enumerate(categories)}
    return encoders


def encode_match_features(df, encoders):
    df = df.copy()
    df["home_team_code"] = df["home_team"].map(encoders["home_team"]).fillna(-1).astype(int)
    df["away_team_code"] = df["away_team"].map(encoders["away_team"]).fillna(-1).astype(int)
    df["tournament_code"] = df["tournament"].map(encoders["tournament"]).fillna(-1).astype(int)
    return df


def _safe_ratio(numerator, denominator):
    return float(numerator) / denominator if denominator else 0.0


def _default_team_stats():
    return {"matches": 0, "points": 0, "goals_for": 0, "goals_against": 0, "wins": 0, "draws": 0}


def _default_h2h_stats():
    return {"team1_wins": 0, "team2_wins": 0, "draws": 0, "goal_diff_team1": 0}


def _build_h2h_key(team_a, team_b):
    return tuple(sorted([team_a, team_b]))


def _invert_result(result):
    if result == 1:
        return 2
    if result == 2:
        return 1
    return 0


def _team_feature_values(stats):
    matches = stats["matches"]
    goal_diff = stats["goals_for"] - stats["goals_against"]
    return (
        _safe_ratio(stats["points"], matches),
        _safe_ratio(goal_diff, matches),
        _safe_ratio(stats["wins"], matches),
    )


def _normalize_h2h_for_home(home_team, away_team, h2h_stats):
    key = _build_h2h_key(home_team, away_team)
    entry = h2h_stats.get(key, _default_h2h_stats())
    if key == (home_team, away_team):
        return entry["team1_wins"], entry["draws"], entry["team2_wins"], entry["goal_diff_team1"]
    return entry["team2_wins"], entry["draws"], entry["team1_wins"], -entry["goal_diff_team1"]


def _update_team_stats(stats, goals_for, goals_against, result):
    stats["matches"] += 1
    stats["goals_for"] += goals_for
    stats["goals_against"] += goals_against
    stats["wins"] += 1 if result == 1 else 0
    stats["draws"] += 1 if result == 0 else 0
    stats["points"] += 3 if result == 1 else 1 if result == 0 else 0


def _update_h2h_stats(h2h_stats, home_team, away_team, home_score, away_score):
    key = _build_h2h_key(home_team, away_team)
    entry = h2h_stats.setdefault(key, _default_h2h_stats().copy())
    if key == (home_team, away_team):
        team1_score, team2_score = home_score, away_score
    else:
        team1_score, team2_score = away_score, home_score

    if team1_score > team2_score:
        entry["team1_wins"] += 1
    elif team1_score == team2_score:
        entry["draws"] += 1
    else:
        entry["team2_wins"] += 1
    entry["goal_diff_team1"] += team1_score - team2_score


def add_history_features(df):
    df = df.sort_values("date").reset_index(drop=True)
    team_stats = {}
    h2h_stats = {}
    history = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])

        home_stats = team_stats.get(home_team, _default_team_stats())
        away_stats = team_stats.get(away_team, _default_team_stats())

        home_strength, home_goal_diff, home_win_rate = _team_feature_values(home_stats)
        away_strength, away_goal_diff, away_win_rate = _team_feature_values(away_stats)
        h2h_home_wins, h2h_draws, h2h_away_wins, h2h_goal_diff = _normalize_h2h_for_home(home_team, away_team, h2h_stats)

        history.append(
            {
                "home_team_strength": home_strength,
                "away_team_strength": away_strength,
                "home_team_goal_diff": home_goal_diff,
                "away_team_goal_diff": away_goal_diff,
                "home_team_win_rate": home_win_rate,
                "away_team_win_rate": away_win_rate,
                "h2h_home_wins": h2h_home_wins,
                "h2h_draws": h2h_draws,
                "h2h_away_wins": h2h_away_wins,
                "h2h_goal_diff": h2h_goal_diff,
            }
        )

        result = 1 if home_score > away_score else 0 if home_score == away_score else 2
        _update_team_stats(team_stats.setdefault(home_team, _default_team_stats()), home_score, away_score, result)
        _update_team_stats(team_stats.setdefault(away_team, _default_team_stats()), away_score, home_score, _invert_result(result))
        _update_h2h_stats(h2h_stats, home_team, away_team, home_score, away_score)

    history_df = pd.DataFrame(history)
    df = pd.concat([df.reset_index(drop=True), history_df], axis=1)
    for col in RAG_COLUMNS:
        df[col] = 0.0
    return df


def build_match_history_stats(df):
    team_stats = {}
    h2h_stats = {}
    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        result = 1 if home_score > away_score else 0 if home_score == away_score else 2

        _update_team_stats(team_stats.setdefault(home_team, _default_team_stats()), home_score, away_score, result)
        _update_team_stats(team_stats.setdefault(away_team, _default_team_stats()), away_score, home_score, _invert_result(result))
        _update_h2h_stats(h2h_stats, home_team, away_team, home_score, away_score)

    return team_stats, h2h_stats


def build_single_match_features(home_team, away_team, tournament, neutral, df, encoders):
    team_stats, h2h_stats = build_match_history_stats(df)
    home_stats = team_stats.get(home_team, _default_team_stats())
    away_stats = team_stats.get(away_team, _default_team_stats())
    home_strength, home_goal_diff, home_win_rate = _team_feature_values(home_stats)
    away_strength, away_goal_diff, away_win_rate = _team_feature_values(away_stats)
    h2h_home_wins, h2h_draws, h2h_away_wins, h2h_goal_diff = _normalize_h2h_for_home(home_team, away_team, h2h_stats)

    return [
        encoders["home_team"].get(home_team, -1),
        encoders["away_team"].get(away_team, -1),
        encoders["tournament"].get(tournament, -1),
        int(neutral),
        home_strength,
        away_strength,
        home_goal_diff,
        away_goal_diff,
        home_win_rate,
        away_win_rate,
        h2h_home_wins,
        h2h_draws,
        h2h_away_wins,
        h2h_goal_diff,
    ]


def prepare_train_test(df, test_size=0.2, random_state=42):
    df = add_history_features(df)
    encoders = build_encoders(df)
    df = encode_match_features(df, encoders)
    for col in RAG_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    X = df[FEATURE_COLUMNS]
    y = df["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, encoders
