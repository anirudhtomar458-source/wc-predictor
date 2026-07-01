import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(page_title="World Cup Live Predictor", page_icon="⚽", layout="centered")

st.title("⚽ FIFA World Cup Live Match Predictor")
st.write("Simulate live matches and watch the win probability scorecard shift in real-time.")
@st.cache_data
def load_and_train():
    df = pd.read_csv('fifa_world_cup_historical_dataset_1930_2026.csv')
    le = LabelEncoder()
    df['target_label'] = le.fit_transform(df['target_outcome'])
    
    features = [
        'team_a_rating', 'team_b_rating', 'team_a_roll_xg', 'team_b_roll_xg',
        'team_a_market_value_m_eur', 'team_b_market_value_m_eur',
        'points_per_game_team_a', 'points_per_game_team_b', 'head_to_head_win_ratio_a'
    ]
    
    X = df[features]
    y = df['target_label']
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)
    return model, df, features

model, df, features = load_and_train()

# Get absolute unique sorted list of all teams from the dataset
all_teams = sorted(list(set(df['team_a'].unique()).union(set(df['team_b'].unique()))))

# 2. Sidebar Interactive Controller Inputs
st.sidebar.header("🕹️ Match Control Center")

team_a = st.sidebar.selectbox("Select Team A (Home/Anchor)", all_teams, index=all_teams.index("Belgium"))
team_b = st.sidebar.selectbox("Select Team B (Away/Challenger)", all_teams, index=all_teams.index("Senegal"))

if team_a == team_b:
    st.sidebar.error("Error: Please select two different teams.")

match_minute = st.sidebar.slider("Match Clock (Minutes)", min_value=1, max_value=90, value=1)

col1, col2 = st.sidebar.columns(2)
with col1:
    goals_a = st.number_input(f"{team_a} Goals", min_value=0, max_value=10, value=0)
    red_a = st.number_input(f"{team_a} Red Cards", min_value=0, max_value=4, value=0)
with col2:
    goals_b = st.number_input(f"{team_b} Goals", min_value=0, max_value=10, value=0)
    red_b = st.number_input(f"{team_b} Red Cards", min_value=0, max_value=4, value=0)

# 3. Predictor Logic Core
try:
    stats_a = df[(df['team_a'] == team_a) | (df['team_b'] == team_a)].sort_values(by='date').iloc[-1]
    stats_b = df[(df['team_a'] == team_b) | (df['team_b'] == team_b)].sort_values(by='date').iloc[-1]
    
    rating_a = stats_a['team_a_rating'] if stats_a['team_a'] == team_a else stats_a['team_b_rating']
    rating_b = stats_b['team_b_rating'] if stats_b['team_a'] == team_b else stats_b['team_b_rating']
    roll_xg_a = stats_a['team_a_roll_xg'] if stats_a['team_a'] == team_a else stats_a['team_b_roll_xg']
    roll_xg_b = stats_b['team_b_roll_xg'] if stats_b['team_a'] == team_b else stats_b['team_b_rating'] # structural fallback
    mv_a = stats_a['team_a_market_value_m_eur'] if stats_a['team_a'] == team_a else stats_a['team_b_market_value_m_eur']
    mv_b = stats_b['team_b_market_value_m_eur'] if stats_b['team_a'] == team_b else stats_b['team_b_market_value_m_eur']
    ppg_a = stats_a['points_per_game_team_a'] if stats_a['team_a'] == team_a else stats_a['points_per_game_team_b']
    ppg_b = stats_b['points_per_game_team_b'] if stats_b['team_a'] == team_b else stats_b['points_per_game_team_b']
    
    input_data = pd.DataFrame([[rating_a, rating_b, roll_xg_a, roll_xg_b, mv_a, mv_b, ppg_a, ppg_b, 0.5]], columns=features)
    base_probs = model.predict_proba(input_data)[0]
    p_draw, p_loss, p_win = base_probs[0], base_probs[1], base_probs[2]
    
    # Physics/Time adjustments
    time_factor = match_minute / 90.0
    score_diff = goals_a - goals_b
    
    if score_diff > 0:
        p_win += (0.5 * score_diff) * time_factor
        p_loss -= (0.4 * score_diff) * time_factor
        p_draw -= (0.1 * score_diff) * time_factor
    elif score_diff < 0:
        p_loss += (0.5 * abs(score_diff)) * time_factor
        p_win -= (0.4 * abs(score_diff)) * time_factor
        p_draw -= (0.1 * abs(score_diff)) * time_factor
    else:
        if match_minute > 70:
            p_draw += 0.5 * time_factor
            p_win -= 0.25 * time_factor
            p_loss -= 0.25 * time_factor

    p_win -= (red_a * 0.15)
    p_loss -= (red_b * 0.15)
    
    raw_totals = np.clip([p_win, p_draw, p_loss], 0.01, 0.99)
    normalized_probs = raw_totals / np.sum(raw_totals)
    
    # 4. Render Beautiful Google Scorecard UI
    st.markdown("---")
    st.subheader("📊 Live Scoreboard")
    
    # Create big live scoreboard text layout
    st.markdown(f"<h1 style='text-align: center; color: #4F8BF9;'>{team_a} {goals_a} — {goals_b} {team_b}</h1>", unsafe_style_allowed=True)
    st.markdown(f"<p style='text-align: center;'><b>Match Clock: {match_minute}'</b></p>", unsafe_style_allowed=True)
    
    st.subheader("🔮 Win Probability (90 MIN)")
    
    # Render interactive columns mimicking the percentage bar layouts
    c1, c2, c3 = st.columns(3)
    c1.metric(label=f"{team_a} Win", value=f"{normalized_probs[0]*100:.1f}%")
    c2.metric(label="Draw / Extra Time", value=f"{normalized_probs[1]*100:.1f}%")
    c3.metric(label=f"{team_b} Win", value=f"{normalized_probs[2]*100:.1f}%")
    
    # Progress bar display visualization hack
    st.progress(float(normalized_probs[0]))

except Exception as e:
    st.error(f"Waiting for custom configurations or baseline retrieval data mismatch error: {e}")