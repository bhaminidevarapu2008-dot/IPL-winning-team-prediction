import streamlit as st
import pickle
import pandas as pd

# Supported choices mapping structural inputs cleanly
teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Delhi Capitals',
    'Punjab Kings', 'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants'
]

cities = ['Mumbai', 'Chennai', 'Bengaluru', 'Kolkata', 'Delhi', 'Hyderabad', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Chandigarh']

# Load model pipeline securely
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'pipe.pkl' not found in this directory. Please place the downloaded model file here.")

st.title('IPL Live Match Win Predictor')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams), index=0)
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams), index=1)

selected_city = st.selectbox('Select Host City', sorted(cities))
target = st.number_input('Target Score', min_value=1, value=180, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, value=0.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=10, value=0, step=1)

if st.button('Predict Win Probability'):
    if batting_team == bowling_team:
        st.error("Batting team and Bowling team cannot be the same!")
    elif score >= target:
        st.success(f"{batting_team} has already won the match!")
    elif wickets >= 10 or overs >= 20:
        if score < target - 1:
            st.success(f"{bowling_team} has won the match!")
        elif score == target - 1:
            st.success("Match Tied!")
    else:
        # Prevent division by zero errors securely
        runs_left = target - score
        balls_left = max(0, 120 - int(overs * 6))
        wickets_left = 10 - wickets
        
        crr = score / overs if overs > 0 else 0.0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0

        # Construct identical Pandas row architecture matching training step features
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        try:
            # Predict probability processing array elements
            result = pipe.predict_proba(input_df)
            loss_prob = result[0][0]
            win_prob = result[0][1]

            st.markdown("---")
            st.header("Match Winning Probability")
            st.subheader(f"🏏 {batting_team}:  **{round(win_prob * 100, 2)}%**")
            st.subheader(f"🥎 {bowling_team}:  **{round(loss_prob * 100, 2)}%**")
        except Exception as e:
            st.error(f"Prediction Error encountered: {e}")
