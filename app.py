import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Set up clean web page configuration
st.set_page_config(page_title="World Cup Live Predictor", page_icon="⚽", layout="centered")

st.title("⚽ FIFA World Cup Live Match Predictor")
st.write("Simulate live matches and watch the win probability scorecard shift in real-time.")

# 1. Load Data & Train Baseline Model (Cached for speed optimization)
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

team_a = st.sidebar.selectbox("Select Team A", all_teams, index=all_teams.index("Belgium"))
team_b = st.sidebar.selectbox("Select Team B", all_teams, index=all_teams.index("Senegal"))

if team_a == team_b:
    st.sidebar.error("Error: Please select two different teams.")

# Mode selection toggle switch
is_live = st.sidebar.checkbox("🟢 Activate Live Match Mode", value=False)

# Initialize live metrics to zero/neutral by default
match_minute = 1
goals_a, goals_b = 0, 0
shots_a, shots_b = 0, 0
sot_a, sot_b = 0, 0
possession_a = 50
pass_a, pass_b = 0, 0
acc_a, acc_b = 80.0, 80.0
fouls_a, fouls_b = 0, 0
yc_a, yc_b = 0, 0
rc_a, rc_b = 0, 0
offsides_a, offsides_b = 0, 0
corners_a, corners_b = 0, 0

if is_live:
    st.sidebar.markdown("### ⏱️ Match Clock & Score")
    match_minute = st.sidebar.slider("Match Clock (Minutes)", min_value=1, max_value=90, value=45)
    
    col_g1, col_g2 = st.sidebar.columns(2)
    with col_g1:
        goals_a = st.number_input(f"{team_a} Goals", min_value=0, max_value=10, value=0)
    with col_g2:
        goals_b = st.number_input(f"{team_b} Goals", min_value=0, max_value=10, value=0)

    st.sidebar.markdown("### 📊 Live Match Stats")
    
    # Possession Split
    possession_a = st.sidebar.slider(f"{team_a} Possession %", min_value=15, max_value=85, value=50)
    possession_b = 100 - possession_a
    
    # Shots and Shots on Target
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        shots_a = st.number_input(f"{team_a} Total Shots", min_value=0, max_value=40, value=5)
        sot_a = st.number_input(f"{team_a} Shots on Target", min_value=0, max_value=20, value=2)
    with col_s2:
        shots_b = st.number_input(f"{team_b} Total Shots", min_value=0, max_value=40, value=4)
        sot_b = st.number_input(f"{team_b} Shots on Target", min_value=0, max_value=20, value=1)
        
    # Passing Stats
    col_p1, col_p2 = st.sidebar.columns(2)
    with col_p1:
        pass_a = st.number_input(f"{team_a} Total Passes", min_value=0, max_value=1000, value=200)
        acc_a = st.slider(f"{team_a} Pass Accuracy %", min_value=40.0, max_value=100.0, value=80.0)
    with col_p2:
        pass_b = st.number_input(f"{team_b} Total Passes", min_value=0, max_value=1000, value=200)
        acc_b = st.slider(f"{team_b} Pass Accuracy %", min_value=40.0, max_value=100.0, value=80.0)

    # Discipline and Set Pieces
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        fouls_a = st.number_input(f"{team_a} Fouls", min_value=0, max_value=30, value=5)
        yc_a = st.number_input(f"{team_a} Yellow Cards", min_value=0, max_value=5, value=0)
        rc_a = st.number_input(f"{team_a} Red Cards", min_value=0, max_value=4, value=0)
        offsides_a = st.number_input(f"{team_a} Offsides", min_value=0, max_value=10, value=1)
        corners_a = st.number_input(f"{team_a} Corners", min_value=0, max_value=20, value=2)
    with col_d2:
        fouls_b = st.number_input(f"{team_b} Fouls", min_value=0, max_value=30, value=6)
        yc_b = st.number_input(f"{team_b} Yellow Cards", min_value=0, max_value=5, value=0)
        rc_b = st.number_input(f"{team_b} Red Cards", min_value=0, max_value=4, value=0)
        offsides_b = st.number_input(f"{team_b} Offsides", min_value=0, max_value=10, value=1)
        corners_b = st.number_input(f"{team_b} Corners", min_value=0, max_value=20, value=1)
else:
    st.info("ℹ️ App running in Pre-Match Baseline Mode. Turn on 'Activate Live Match Mode' in the sidebar to simulate match statistics.")

# 3. Predictor Logic Core
try:
    stats_a = df[(df['team_a'] == team_a) | (df['team_b'] == team_a)].sort_values(by='date').iloc[-1]
    stats_b = df[(df['team_a'] == team_b) | (df['team_b'] == team_b)].sort_values(by='date').iloc[-1]
    
    rating_a = stats_a['team_a_rating'] if stats_a['team_a'] == team_a else stats_a['team_b_rating']
    rating_b = stats_b['team_b_rating'] if stats_b['team_a'] == team_b else stats_b['team_b_rating']
    roll_xg_a = stats_a['team_a_roll_xg'] if stats_a['team_a'] == team_a else stats_a['team_b_roll_xg']
    roll_xg_b = stats_b['team_b_roll_xg'] if stats_b['team_a'] == team_b else stats_b['team_b_rating']
    mv_a = stats_a['team_a_market_value_m_eur'] if stats_a['team_a'] == team_a else stats_a['team_b_market_value_m_eur']
    mv_b = stats_b['team_b_market_value_m_eur'] if stats_b['team_a'] == team_b else stats_b['team_b_market_value_m_eur']
    ppg_a = stats_a['points_per_game_team_a'] if stats_a['team_a'] == team_a else stats_a['points_per_game_team_b']
    ppg_b = stats_b['points_per_game_team_b'] if stats_b['team_a'] == team_b else stats_b['points_per_game_team_b']
    
    input_data = pd.DataFrame([[rating_a, rating_b, roll_xg_a, roll_xg_b, mv_a, mv_b, ppg_a, ppg_b, 0.5]], columns=features)
    base_probs = model.predict_proba(input_data)[0]
    p_draw, p_loss, p_win = base_probs[0], base_probs[1], base_probs[2]
    
    # Advanced Live Statistics Weight Modifiers
    if is_live:
        time_factor = match_minute / 90.0
        
        # Core scoreline impact
        score_diff = goals_a - goals_b
        if score_diff > 0:
            p_win += (0.45 * score_diff) * time_factor
            p_loss -= (0.35 * score_diff) * time_factor
        elif score_diff < 0:
            p_loss += (0.45 * abs(score_diff)) * time_factor
            p_win -= (0.35 * abs(score_diff)) * time_factor
        else:
            if match_minute > 70:
                p_draw += 0.4 * time_factor
                p_win -= 0.2 * time_factor
                p_loss -= 0.2 * time_factor

        # Micro-momentum metrics modifications (Shots on target, Possession, Corners)
        sot_diff = sot_a - sot_b
        p_win += (sot_diff * 0.015)
        p_loss -= (sot_diff * 0.015)
        
        poss_diff = possession_a - 50
        p_win += (poss_diff * 0.002)
        p_loss -= (poss_diff * 0.002)
        
        corners_diff = corners_a - corners_b
        p_win += (corners_diff * 0.005)
        
        # Penalizing for red cards and performance errors
        p_win -= (rc_a * 0.18)
        p_loss -= (rc_b * 0.18)
        p_win -= (yc_a * 0.02)
        p_loss -= (yc_b * 0.02)

    # Re-normalize probabilities array safely
    raw_totals = np.clip([p_win, p_draw, p_loss], 0.01, 0.99)
    normalized_probs = raw_totals / np.sum(raw_totals)
    
    # 4. Render Beautiful Scoreboard layout
    st.markdown("---")
    st.markdown(f"<h1 style='text-align: center; color: #4F8BF9;'>{team_a} {goals_a} — {goals_b} {team_b}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'><b>Match Clock: {match_minute}'</b></p>", unsafe_allow_html=True)
    
    st.subheader("🔮 Win Probability Scorecard")
    c1, c2, c3 = st.columns(3)
    c1.metric(label=f"{team_a} Win", value=f"{normalized_probs[0]*100:.1f}%")
    c2.metric(label="Draw / Extra Time", value=f"{normalized_probs[1]*100:.1f}%")
    c3.metric(label=f"{team_b} Win", value=f"{normalized_probs[2]*100:.1f}%")
    st.progress(float(normalized_probs[0]))

    # Render a comparative layout table for live stats comparison
    if is_live:
        st.subheader("📊 Match Statistics Comparison")
        stats_comparison = pd.DataFrame({
            f"{team_a}": [possession_a, shots_a, sot_a, pass_a, f"{acc_a}%", fouls_a, yc_a, rc_a, offsides_a, corners_a],
            "Metric Vector": ["Possession %", "Shots", "Shots on Target", "Total Passes", "Pass Accuracy", "Fouls", "Yellow Cards", "Red Cards", "Offsides", "Corners"],
            f"{team_b}": [possession_b, shots_b, sot_b, pass_b, f"{acc_b}%", fouls_b, yc_b, rc_b, offsides_b, corners_b]
        }).set_index("Metric Vector")
        st.table(stats_comparison)

except Exception as e:
    st.error(f"Waiting for custom configurations or baseline retrieval data mismatch error: {e}")
