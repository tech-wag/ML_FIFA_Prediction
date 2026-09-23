import re
import sqlite3
from pathlib import Path

import pandas as pd


class DatasetQueryEngine:
    def __init__(self, csv_path="results.csv", dataset_name=None):
        self.csv_path = csv_path
        self.dataset_name = dataset_name or self._infer_dataset_name(csv_path)
        self.df = self._load_dataframe(csv_path)
        self.df.columns = [str(col).strip() for col in self.df.columns]
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.df.to_sql("dataset", self.conn, index=False, if_exists="replace")
        self._seed_known_values()

    @staticmethod
    def _infer_dataset_name(csv_path):
        normalized = (csv_path or "").lower()
        if "deliver" in normalized:
            return "ipl_deliveries"
        if "match" in normalized:
            return "ipl_matches"
        if "housing" in normalized or "california" in normalized:
            return "california_housing"
        return "football_results"

    def _load_dataframe(self, csv_path):
        path = Path(csv_path)
        if path.exists():
            return pd.read_csv(path)

        if self.dataset_name == "california_housing":
            from sklearn.datasets import fetch_california_housing

            housing = fetch_california_housing(as_frame=True)
            return housing.frame.copy()

        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    def _seed_known_values(self):
        cols = set(str(column) for column in self.df.columns)
        self.table_name = "dataset"

        if self.dataset_name == "football_results":
            if "home_team" not in cols or "away_team" not in cols:
                self.teams = []
                return
            team_values = self.df["home_team"].dropna().astype(str).tolist() + self.df["away_team"].dropna().astype(str).tolist()
            self.teams = sorted(set(team_values), key=lambda value: (-len(value), value))
            return

        if self.dataset_name == "ipl_matches":
            if "winner" not in cols or "team1" not in cols or "team2" not in cols:
                self.teams = []
                return
            team_values = self.df["winner"].dropna().astype(str).tolist() + self.df["team1"].dropna().astype(str).tolist() + self.df["team2"].dropna().astype(str).tolist()
            self.teams = sorted(set(team_values), key=lambda value: (-len(value), value))
            return

        if self.dataset_name == "ipl_deliveries":
            if "batting_team" not in cols or "bowling_team" not in cols:
                self.teams = []
                return
            team_values = self.df["batting_team"].dropna().astype(str).tolist() + self.df["bowling_team"].dropna().astype(str).tolist()
            self.teams = sorted(set(team_values), key=lambda value: (-len(value), value))
            return

        self.teams = []

    @staticmethod
    def _escape_sql(value):
        return str(value).replace("'", "''")

    def _find_team_matches(self, query_text):
        normalized = query_text.lower()
        matches = []
        for team in getattr(self, "teams", []):
            if team.lower() in normalized:
                matches.append(team)
        return matches

    def _find_column_value(self, query_text, candidates):
        normalized = query_text.lower()
        for candidate in candidates:
            if candidate.lower() in normalized:
                return candidate
        return None

    def build_query(self, query_text: str) -> str:
        query_text = (query_text or "").strip()
        if not query_text:
            return f"SELECT * FROM {self.table_name} LIMIT 20"

        normalized = query_text.lower()
        year_match = re.search(r"\b(19|20)\d{2}\b", query_text)

        if self.dataset_name == "football_results":
            if "how many" in normalized and "home" in normalized and ("win" in normalized or "wins" in normalized):
                team = self._find_team_matches(query_text)
                if team:
                    team_name = self._escape_sql(team[0])
                    return (
                        f"SELECT COUNT(*) AS count FROM {self.table_name} "
                        f"WHERE home_team = '{team_name}' AND home_score > away_score"
                    )

            if "how many" in normalized and "away" in normalized and ("win" in normalized or "wins" in normalized):
                team = self._find_team_matches(query_text)
                if team:
                    team_name = self._escape_sql(team[0])
                    return (
                        f"SELECT COUNT(*) AS count FROM {self.table_name} "
                        f"WHERE away_team = '{team_name}' AND away_score > home_score"
                    )

            if year_match and "played" in normalized:
                year = int(year_match.group(0))
                return (
                    f"SELECT COUNT(*) AS count FROM {self.table_name} "
                    f"WHERE date >= '{year}-01-01' AND date < '{year + 1}-01-01'"
                )

            team_hits = self._find_team_matches(query_text)
            if "between" in normalized and len(team_hits) >= 2:
                team_a, team_b = team_hits[:2]
                team_a_sql = self._escape_sql(team_a)
                team_b_sql = self._escape_sql(team_b)
                return (
                    f"SELECT date, home_team, away_team, home_score, away_score, tournament FROM {self.table_name} "
                    f"WHERE (home_team = '{team_a_sql}' AND away_team = '{team_b_sql}') "
                    f"OR (home_team = '{team_b_sql}' AND away_team = '{team_a_sql}') ORDER BY date DESC LIMIT 20"
                )

            if "show" in normalized or "list" in normalized or "matches" in normalized:
                if len(team_hits) >= 2:
                    team_a, team_b = team_hits[:2]
                    team_a_sql = self._escape_sql(team_a)
                    team_b_sql = self._escape_sql(team_b)
                    return (
                        f"SELECT date, home_team, away_team, home_score, away_score, tournament FROM {self.table_name} "
                        f"WHERE (home_team = '{team_a_sql}' AND away_team = '{team_b_sql}') "
                        f"OR (home_team = '{team_b_sql}' AND away_team = '{team_a_sql}') ORDER BY date DESC LIMIT 20"
                    )

            if "count" in normalized:
                return f"SELECT COUNT(*) AS count FROM {self.table_name}"

            return f"SELECT * FROM {self.table_name} ORDER BY date DESC LIMIT 20"

        if self.dataset_name == "ipl_matches":
            team_hits = self._find_team_matches(query_text)
            if "how many" in normalized and ("win" in normalized or "wins" in normalized):
                if team_hits:
                    team_name = self._escape_sql(team_hits[0])
                    return (
                        f"SELECT COUNT(*) AS count FROM {self.table_name} "
                        f"WHERE winner = '{team_name}'"
                    )

            if "show" in normalized or "list" in normalized or "matches" in normalized:
                if len(team_hits) >= 2:
                    team_a, team_b = team_hits[:2]
                    team_a_sql = self._escape_sql(team_a)
                    team_b_sql = self._escape_sql(team_b)
                    return (
                        f"SELECT date, team1, team2, winner, result, result_margin FROM {self.table_name} "
                        f"WHERE (team1 = '{team_a_sql}' AND team2 = '{team_b_sql}') "
                        f"OR (team1 = '{team_b_sql}' AND team2 = '{team_a_sql}') ORDER BY date DESC LIMIT 20"
                    )

            if "count" in normalized:
                return f"SELECT COUNT(*) AS count FROM {self.table_name}"

            if "winner" in normalized or "winning" in normalized:
                if team_hits:
                    team_name = self._escape_sql(team_hits[0])
                    return f"SELECT COUNT(*) AS count FROM {self.table_name} WHERE winner = '{team_name}'"

            return f"SELECT * FROM {self.table_name} ORDER BY date DESC LIMIT 20"

        if self.dataset_name == "ipl_deliveries":
            team_hits = self._find_team_matches(query_text)
            team_name = team_hits[0] if team_hits else None

            if "total runs" in normalized or "sum of runs" in normalized or "runs scored" in normalized:
                if team_name:
                    team_name_sql = self._escape_sql(team_name)
                    return (
                        f"SELECT SUM(total_runs) AS total_runs FROM {self.table_name} "
                        f"WHERE batting_team = '{team_name_sql}'"
                    )

            if "show" in normalized or "list" in normalized or "deliveries" in normalized:
                if team_name:
                    team_name_sql = self._escape_sql(team_name)
                    return (
                        f"SELECT match_id, batting_team, bowling_team, SUM(total_runs) AS total_runs FROM {self.table_name} "
                        f"WHERE batting_team = '{team_name_sql}' GROUP BY match_id, batting_team, bowling_team ORDER BY total_runs DESC LIMIT 20"
                    )

            if "wicket" in normalized or "dismissed" in normalized:
                if team_name:
                    team_name_sql = self._escape_sql(team_name)
                    return (
                        f"SELECT COUNT(*) AS wickets FROM {self.table_name} "
                        f"WHERE batting_team = '{team_name_sql}' AND is_wicket = 1"
                    )

            if "count" in normalized:
                return f"SELECT COUNT(*) AS count FROM {self.table_name}"

            return f"SELECT * FROM {self.table_name} LIMIT 20"

        if self.dataset_name == "california_housing":
            if "average" in normalized and "house" in normalized and ("value" in normalized or "price" in normalized):
                return f"SELECT AVG(median_house_value) AS avg_median_house_value FROM {self.table_name}"

            if "near bay" in normalized or "nearest bay" in normalized or "bay" in normalized:
                return f"SELECT * FROM {self.table_name} WHERE ocean_proximity = 'NEAR BAY' ORDER BY median_house_value DESC LIMIT 20"

            if "near ocean" in normalized or "ocean" in normalized:
                return f"SELECT * FROM {self.table_name} WHERE ocean_proximity = 'NEAR OCEAN' ORDER BY median_house_value DESC LIMIT 20"

            if "count" in normalized:
                return f"SELECT COUNT(*) AS count FROM {self.table_name}"

            if "average" in normalized and "income" in normalized:
                return f"SELECT AVG(median_income) AS avg_median_income FROM {self.table_name}"

            return f"SELECT * FROM {self.table_name} ORDER BY median_house_value DESC LIMIT 20"

        return f"SELECT * FROM {self.table_name} LIMIT 20"

    def execute(self, query_text: str):
        sql_query = self.build_query(query_text)
        return pd.read_sql_query(sql_query, self.conn)

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
