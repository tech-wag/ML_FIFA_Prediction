import streamlit as st
from Predictor import predict_match, load_model

st.set_page_config(page_title="FIFA Match Predictor", page_icon="⚽", layout="centered")

st.title("FIFA Match Result Predictor")
st.write("Predict the outcome of a football match using a trained Random Forest model.")

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
            
            # Visual bar chart with custom labels
            prob_dict = {
                f"{selected_team_a} Win": home_win_prob,
                "Draw": draw_prob,
                f"{selected_team_b} Win": away_win_prob
            }
            st.bar_chart(prob_dict)
        else:
            # For home/away venues, use standard labels
            st.write(f"**{result['label']}**")
            st.subheader("📊 Probabilities")
            st.bar_chart(result["probabilities"])

        st.subheader("📋 Input Features")
        st.write(result["features"])
