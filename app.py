import streamlit as st
from Predictor import predict_match, load_model
from nlp_to_sql import DatasetQueryEngine


def get_query_engine():
    return DatasetQueryEngine("results.csv")


def render_probability_row(label: str, value: float, color: str) -> str:
    return (
        "<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>"
        f"<div style='flex:0 0 180px;font-weight:600;'>{label}</div>"
        "<div style='flex:1;background:#e6e6e6;border-radius:999px;overflow:hidden;height:18px;'>"
        f"<div style='width:{value*100:.1f}%;background:{color};height:100%;border-radius:999px;'></div>"
        "</div>"
        f"<div style='flex:0 0 55px;text-align:right;font-variant-numeric:tabular-nums;'>{value*100:.1f}%</div>"
        "</div>"
    )


st.set_page_config(page_title="FIFA Match Predictor", page_icon="⚽", layout="centered")

st.title("FIFA Match Result Predictor")
st.write("Predict the outcome of a football match using a trained Random Forest model.")

st.subheader("📊 Ask the dataset a natural-language question")
query_prompt = st.text_input(
    "Question about the football results dataset",
    value="How many matches did Brazil win at home?",
    key="dataset_query_input",
)
if st.button("Run dataset query", key="run_dataset_query"):
    query_engine = get_query_engine()
    try:
        sql_query = query_engine.build_query(query_prompt)
        result_df = query_engine.execute(query_prompt)
        st.code(sql_query, language="sql")
        st.dataframe(result_df.head(20), use_container_width=True)
    finally:
        query_engine.close()

artifact = load_model()
encoders = artifact["encoders"]

# Build team and tournament lists (teams may be same across home/away encoders)
teams = sorted(set(list(encoders["home_team"].keys()) + list(encoders["away_team"].keys())))
tournaments = sorted(encoders["tournament"].keys())

# Default the neutral checkbox to checked per user request
neutral = st.checkbox("Neutral venue", value=True, key="neutral_checkbox")

if neutral:
    team_a = st.selectbox("Team A", teams, key="team_a")
    team_b = st.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0, key="team_b")
    tournament = st.selectbox("Tournament", tournaments)
else:
    home_team = st.selectbox("Home Team", teams, key="home_team")
    away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0, key="away_team")
    tournament = st.selectbox("Tournament", tournaments)

predict_btn = st.button("Predict")

if predict_btn:
    if neutral:
        home = team_a
        away = team_b
        selected_team_a = team_a
        selected_team_b = team_b
    else:
        home = home_team
        away = away_team
        selected_team_a = home_team
        selected_team_b = away_team

    if home == away:
        st.warning("Please select two different teams.")
    else:
        result = predict_match(home, away, tournament, neutral)
        
        # Display prediction with better formatting
        st.subheader("🎯 Match Prediction")
        
        # Get probabilities
        probs = result["probabilities"]
        
        if neutral:
            # For neutral venues, show team-specific labels
            home_win_prob = probs.get("Home Win", 0)
            draw_prob = probs.get("Draw", 0)
            away_win_prob = probs.get("Away Win", 0)
            
            # Determine the most likely outcome
            max_prob = max(home_win_prob, draw_prob, away_win_prob)
            if max_prob == home_win_prob:
                prediction_text = f"**{selected_team_a} Win**"
                prediction_color = "🟢"
            elif max_prob == draw_prob:
                prediction_text = "**Draw**"
                prediction_color = "🟡"
            else:
                prediction_text = f"**{selected_team_b} Win**"
                prediction_color = "🔵"
            
            st.markdown(f"{prediction_color} {prediction_text}")
            
            # Display probabilities with team names
            st.subheader("📊 Win Probabilities")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"🔴 {selected_team_a} Win", f"{home_win_prob*100:.1f}%")
            with col2:
                st.metric("🟡 Draw", f"{draw_prob*100:.1f}%")
            with col3:
                st.metric(f"🔵 {selected_team_b} Win", f"{away_win_prob*100:.1f}%")
            
            # Horizontal progress bars for the probability breakdown
            st.write("**Probability progress**")
            st.markdown(
                render_probability_row(
                    f"🔴 {selected_team_a} Win",
                    home_win_prob,
                    "linear-gradient(90deg,#ff6b6b,#ff9a8b)",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_probability_row(
                    "🟡 Draw",
                    draw_prob,
                    "linear-gradient(90deg,#f4d35e,#f6e27f)",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_probability_row(
                    f"🔵 {selected_team_b} Win",
                    away_win_prob,
                    "linear-gradient(90deg,#4d7cff,#7aa6ff)",
                ),
                unsafe_allow_html=True,
            )
        else:
            # For home/away venues, use standard labels
            st.write(f"**{result['label']}**")
            st.subheader("📊 Probabilities")
            st.write("**Probability progress**")
            st.markdown(
                render_probability_row(
                    "🔴 Home Win",
                    result['probabilities'].get('Home Win', 0),
                    "linear-gradient(90deg,#ff6b6b,#ff9a8b)",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_probability_row(
                    "🟡 Draw",
                    result['probabilities'].get('Draw', 0),
                    "linear-gradient(90deg,#f4d35e,#f6e27f)",
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_probability_row(
                    "🔵 Away Win",
                    result['probabilities'].get('Away Win', 0),
                    "linear-gradient(90deg,#4d7cff,#7aa6ff)",
                ),
                unsafe_allow_html=True,
            )

        st.subheader("📋 Input Features")
        st.write(result["features"])
