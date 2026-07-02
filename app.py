import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import requests

# Set page to wide layout for the side-by-side columns
st.set_page_config(page_title="World Cup Live Predictor", page_icon="⚽", layout="wide")

# --- 🔑 YOUR RAPIDAPI KEY ---
API_KEY = "5cba91e315msh16a9035f15e963ap164cb9jsn8919d5d6747c"

st.title("⚽ FIFA World Cup Dual-Engine Predictor")
st.write("Track the actual live game metrics on the left, and simulate 'what-if' tactical adjustments on the right.")

# 1. Load Data & Train Baseline Model
@st.cache_data
def load_and_train():
    df = pd.read_csv('fifa_world_cup_historical_dataset_1930_2026.csv')
    
    features = [
        'team_a_rating', 'team_b_rating', 'team_a_roll_xg', 'team_b_roll_xg',
        'team_a_market_value_m_eur', 'team_b_market_value_m_eur',
        'points_per_game_team_a', 'points_per_game_team_b', 'head_to_head_win_ratio_a'
    ]
    
    # Bulletproof fallback for missing columns
    for col in features:
        if col not in df.columns:
            df[col] = 1500.0 if 'rating' in col else (200.0 if 'market_value' in col else 1.5)

    if 'target_outcome' not in df.columns:
        df['target_outcome'] = 'Draw' 

    le = LabelEncoder()
    df['target_label'] = le.fit_transform(df['target_outcome'])
    
    X = df[features]
    y = df['target_label']
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)
    
    return model, df, features

model, df, features = load_and_train()
all_teams = sorted(list(set(df['team_a'].unique()).union(set(df['team_b'].unique()))))

# 2. Team Selection
st.markdown("### 🕹️ Global Match Selection")
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    team_a = st.selectbox("Select Team A", all_teams, index=all_teams.index("France") if "France" in all_teams else 0)
with col_sel2:
    team_b = st.selectbox("Select Team B", all_teams, index=all_teams.index("Germany") if "Germany" in all_teams else 1)

if team_a == team_b:
    st.error("Please select two different international teams.")
    st.stop()

def get_team_baselines(name):
    t_data = df[(df['team_a'] == name) | (df['team_b'] == name)]
    if not t_data.empty:
        l = t_data.sort_values(by='date').iloc[-1]
        r = l['team_a_rating'] if l['team_a'] == name else l['team_b_rating']
        m = l['team_a_market_value_m_eur'] if l['team_a'] == name else l['team_b_market_value_m_eur']
        return r, m
    return 1600, 200.0

r_a, m_a = get_team_baselines(team_a)
r_b, m_b = get_team_baselines(team_b)

# 3. Live Web API Fetcher (Cached for 60 seconds to protect your free limit)
@st.cache_data(ttl=60)
def fetch_live_match_data(t_a, t_b):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    headers = {"X-RapidAPI-Key": API_KEY, "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"}
    
    try:
        response = requests.get(url, headers=headers, params={"live": "all"})
        data = response.json()
        
        for match in data.get("response", []):
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]
            
            # Check if selected teams match the live API feed
            if (t_a in home or t_b in home) and (t_a in away or t_b in away):
                minute = match["fixture"]["status"]["elapsed"]
                
                # Check for Extra Time formatting
                extra_time = 0
                if minute > 90:
                    extra_time = minute - 90
                    minute = 90
                
                return {
                    "status": "LIVE",
                    "minute": minute,
                    "extra_time": extra_time,
                    "goals_a": match["goals"]["home"],
                    "goals_b": match["goals"]["away"],
                    "rc_a": 0, 
                    "rc_b": 0,
                    "poss_a": 50
                }
        return {"status": "NOT_PLAYING"}
    except:
        return {"status": "ERROR"}

live_data = fetch_live_match_data(team_a, team_b)

# Fallbacks if match isn't live
live_minute_api = live_data.get("minute", 0)
live_xt_api = live_data.get("extra_time", 0)
live_goals_a_api = live_data.get("goals_a", 0)
live_goals_b_api = live_data.get("goals_b", 0)
live_poss_a_api = live_data.get("poss_a", 50)
live_rc_a_api = live_data.get("rc_a", 0)
live_rc_b_api = live_data.get("rc_b", 0)

# Prediction Math Engine
def compute_prediction(rating_a, rating_b, mv_a, mv_b, minute, extra_time, g_a, g_b, poss_a, rc_a, rc_b):
    input_data = pd.DataFrame([[rating_a, rating_b, 1.5, 1.5, mv_a, mv_b, 1.8, 1.8, 0.5]], columns=features)
    base_probs = model.predict_proba(input_data)[0]
    p_draw, p_loss, p_win = base_probs[0], base_probs[1], base_probs[2]
    
    total_minutes = minute + extra_time
    time_factor = min(total_minutes / 90.0, 1.3) # Cap time factor for extra time
    
    score_diff = g_a - g_b
    if score_diff > 0:
        p_win += (0.45 * score_diff) * time_factor
        p_loss -= (0.35 * score_diff) * time_factor
    elif score_diff < 0:
        p_loss += (0.45 * abs(score_diff)) * time_factor
        p_win -= (0.35 * abs(score_diff)) * time_factor
    else:
        # If drawn and in extra time, draw probability spikes heavily
        if total_minutes > 70:
            p_draw += 0.4 * time_factor
            p_win -= 0.2 * time_factor
            p_loss -= 0.2 * time_factor

    p_win += ((poss_a - 50) * 0.002) - (rc_a * 0.18)
    p_loss -= ((poss_a - 50) * 0.002) + (rc_b * 0.18)
    
    raw_totals = np.clip([p_win, p_draw, p_loss], 0.01, 0.99)
    return raw_totals / np.sum(raw_totals)

st.markdown("---")

# 4. DUAL COLUMN LAYOUT
left_column, right_column = st.columns(2)

# ========== LEFT COLUMN: REAL LIVE FEED ==========
with left_column:
    st.header("📡 Live Official Feed")
    
    if live_data["status"] == "LIVE":
        st.success("🟢 LIVE MATCH DETECTED! Pulling real-time web data...")
        time_display = f"{live_minute_api}'" + (f" + {live_xt_api}' ET" if live_xt_api > 0 else "")
        st.subheader(f"⏱️ Game Status: {time_display}")
    elif live_data["status"] == "ERROR":
        st.warning("⚠️ API connection error. Showing baseline simulation.")
        st.subheader("⏱️ Game Status: Pre-Match (0')")
    else:
        st.info("ℹ️ Teams not currently playing live. Showing baseline.")
        st.subheader("⏱️ Game Status: Pre-Match (0')")
    
    st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>{team_a} {live_goals_a_api} — {live_goals_b_api} {team_b}</h2>", unsafe_allow_html=True)
    
    live_probs = compute_prediction(r_a, r_b, m_a, m_b, live_minute_api, live_xt_api, live_goals_a_api, live_goals_b_api, live_poss_a_api, live_rc_a_api, live_rc_b_api)
    
    st.markdown("#### 🎯 Active Win Probability")
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric(f"{team_a} Win", f"{live_probs[0]*100:.1f}%")
    lc2.metric("Draw", f"{live_probs[1]*100:.1f}%")
    lc3.metric(f"{team_b} Win", f"{live_probs[2]*100:.1f}%")
    st.progress(float(live_probs[0]))

# ========== RIGHT COLUMN: SANDBOX SIMULATOR ==========
with right_column:
    st.header("🎛️ Sandbox Simulator")
    st.write("Override match events below to see how the model reacts.")
    
    sc_time1, sc_time2 = st.columns(2)
    with sc_time1:
        sim_minute = st.slider("Simulated Clock (Mins)", 1, 90, int(live_minute_api))
    with sc_time2:
        sim_xt = st.number_input("Extra Time (Mins)", min_value=0, max_value=30, value=int(live_xt_api))
    
    sc1, sc2 = st.columns(2)
    with sc1:
        sim_goals_a = st.number_input(f"Simulate {team_a} Goals", 0, 12, int(live_goals_a_api))
        sim_rc_a = st.slider(f"Simulate {team_a} Red Cards", 0, 4, int(live_rc_a_api))
    with sc2:
        sim_goals_b = st.number_input(f"Simulate {team_b} Goals", 0, 12, int(live_goals_b_api))
        sim_rc_b = st.slider(f"Simulate {team_b} Red Cards", 0, 4, int(live_rc_b_api))
        
    sim_poss_a = st.slider(f"Simulate {team_a} Possession %", 15, 85, int(live_poss_a_api))
    
    sandbox_probs = compute_prediction(r_a, r_b, m_a, m_b, sim_minute, sim_xt, sim_goals_a, sim_goals_b, sim_poss_a, sim_rc_a, sim_rc_b)
    
    st.markdown("#### 🔮 Sandbox Win Probability")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric(f"{team_a} Win", f"{sandbox_probs[0]*100:.1f}%")
    rc2.metric("Draw", f"{sandbox_probs[1]*100:.1f}%")
    rc3.metric(f"{team_b} Win", f"{sandbox_probs[2]*100:.1f}%")
    st.progress(float(sandbox_probs[0]))
