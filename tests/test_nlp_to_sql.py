import pandas as pd

from nlp_to_sql import DatasetQueryEngine


def test_count_home_wins_query_builds_sql():
    engine = DatasetQueryEngine("results.csv")
    sql = engine.build_query("How many matches did Brazil win at home?")

    assert "SELECT COUNT(*)" in sql.upper()
    assert "home_team = 'Brazil'" in sql
    assert "home_score > away_score" in sql


def test_show_matches_query_builds_sql():
    engine = DatasetQueryEngine("results.csv")
    sql = engine.build_query("Show me matches between Brazil and Argentina")

    assert "SELECT" in sql.upper()
    assert "Brazil" in sql
    assert "Argentina" in sql
    assert "LIMIT 20" in sql.upper()


def test_execute_query_returns_dataframe():
    engine = DatasetQueryEngine("results.csv")
    df = engine.execute("How many matches were played in 2022?")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "count" in df.columns[0].lower() or df.iloc[0, 0] >= 0
