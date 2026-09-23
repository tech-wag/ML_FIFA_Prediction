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


def test_ipl_matches_query_builds_sql():
    engine = DatasetQueryEngine("matches.csv")
    sql = engine.build_query("How many matches did Royal Challengers Bangalore win?")

    assert "SELECT COUNT(*)" in sql.upper()
    assert "winner = 'Royal Challengers Bangalore'" in sql


def test_ipl_deliveries_query_builds_sql():
    engine = DatasetQueryEngine("deliveries.csv")
    sql = engine.build_query("Show total runs by Mumbai Indians")

    assert "SUM(total_runs)" in sql.upper()
    assert "batting_team = 'Mumbai Indians'" in sql


def test_housing_query_builds_sql(tmp_path):
    csv_path = tmp_path / "housing.csv"
    pd.DataFrame(
        [
            {
                "longitude": -122.23,
                "latitude": 37.88,
                "median_house_value": 250000,
                "ocean_proximity": "NEAR BAY",
            },
            {
                "longitude": -122.11,
                "latitude": 37.75,
                "median_house_value": 350000,
                "ocean_proximity": "INLAND",
            },
        ]
    ).to_csv(csv_path, index=False)

    engine = DatasetQueryEngine(str(csv_path))
    sql = engine.build_query("Show houses near the bay")

    assert "SELECT" in sql.upper()
    assert "ocean_proximity = 'NEAR BAY'" in sql
