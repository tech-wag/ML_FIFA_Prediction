import re
import sqlite3

import pandas as pd


class DatasetQueryEngine:
    def __init__(self, csv_path="results.csv"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df.columns = [str(col).strip() for col in self.df.columns]
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.df.to_sql("results", self.conn, index=False, if_exists="replace")

        team_values = self.df["home_team"].dropna().astype(str).tolist() + self.df["away_team"].dropna().astype(str).tolist()
        self.teams = sorted(set(team_values), key=lambda value: (-len(value), value))

    @staticmethod
    def _escape_sql(value):
        return str(value).replace("'", "''")

    def _find_team_matches(self, query_text):
        normalized = query_text.lower()
        matches = []
        for team in self.teams:
            if team.lower() in normalized:
                matches.append(team)
        return matches

    def build_query(self, query_text: str) -> str:
        query_text = (query_text or "").strip()
        if not query_text:
            return "SELECT * FROM results ORDER BY date DESC LIMIT 20"

        normalized = query_text.lower()
        year_match = re.search(r"\b(19|20)\d{2}\b", query_text)

        if "how many" in normalized and "home" in normalized and ("win" in normalized or "wins" in normalized):
            team = self._find_team_matches(query_text)
            if team:
                team_name = self._escape_sql(team[0])
                return (
                    f"SELECT COUNT(*) AS count FROM results "
                    f"WHERE home_team = '{team_name}' AND home_score > away_score"
                )

        if "how many" in normalized and "away" in normalized and ("win" in normalized or "wins" in normalized):
            team = self._find_team_matches(query_text)
            if team:
                team_name = self._escape_sql(team[0])
                return (
                    f"SELECT COUNT(*) AS count FROM results "
                    f"WHERE away_team = '{team_name}' AND away_score > home_score"
                )

        if year_match and "played" in normalized:
            year = int(year_match.group(0))
            return (
                f"SELECT COUNT(*) AS count FROM results "
                f"WHERE date >= '{year}-01-01' AND date < '{year + 1}-01-01'"
            )

        team_hits = self._find_team_matches(query_text)
        if "between" in normalized and len(team_hits) >= 2:
            team_a, team_b = team_hits[:2]
            team_a_sql = self._escape_sql(team_a)
            team_b_sql = self._escape_sql(team_b)
            return (
                "SELECT date, home_team, away_team, home_score, away_score, tournament "
                "FROM results WHERE (home_team = '" + team_a_sql + "' AND away_team = '" + team_b_sql + "') "
                "OR (home_team = '" + team_b_sql + "' AND away_team = '" + team_a_sql + "') "
                "ORDER BY date DESC LIMIT 20"
            )

        if "show" in normalized or "list" in normalized or "matches" in normalized:
            if len(team_hits) >= 2:
                team_a, team_b = team_hits[:2]
                team_a_sql = self._escape_sql(team_a)
                team_b_sql = self._escape_sql(team_b)
                return (
                    "SELECT date, home_team, away_team, home_score, away_score, tournament "
                    "FROM results WHERE (home_team = '" + team_a_sql + "' AND away_team = '" + team_b_sql + "') "
                    "OR (home_team = '" + team_b_sql + "' AND away_team = '" + team_a_sql + "') "
                    "ORDER BY date DESC LIMIT 20"
                )

        if "count" in normalized:
            return "SELECT COUNT(*) AS count FROM results"

        return "SELECT * FROM results ORDER BY date DESC LIMIT 20"

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
