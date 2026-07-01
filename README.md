# ⚽ FIFA World Cup Live Match Predictor

Welcome! This is an interactive web application that acts like a live Google Scorecard, predicting football match outcomes in real-time. You can select any two international teams to see a pre-match baseline probability, and then dynamically adjust mid-game events (like the match clock, current goals, or sudden red cards) to watch the win probability shift instantly.

The project is built using Python, trained on historical data with an XGBoost machine learning classifier, and deployed as a live web dashboard using Streamlit.

---

## 🚀 How It Works

Most basic football prediction systems only look at final scorelines, which introduces a lot of random variance. This system focuses on underlying performance metrics:
1. **Pre-Match Baseline:** When you pick two teams, the system pulls their latest historical data—squad market values, Elo ratings, and rolling Expected Goals (xG)—to calculate a neutral starting probability.
2. **Live Match Physics:** Once you adjust the sliders in the sidebar dashboard, the model applies a time-decay physics simulation. As the clock ticks closer to 90 minutes, the impact of live goals and player red cards scales exponentially, shifting the scorecard dynamically.

---

## 📊 The Data Strategy

The dataset powers everything from 1930 up to the 2026 tournament cycle. Building it required a blend of automation and collaborative refinement:

* **Web Scrapping & Data Aggregation:** Raw match results, historical records, and team frequencies were gathered from various open football databases.
* **AI-Assisted Context (Google Gemini):** Because historical data from earlier decades (like the 1930s to 1970s) lacks modern advanced metrics, I utilized Google Gemini to systematically structure, clean, and backfill the dataset with balanced baseline approximations (such as estimated team Elo tracking and inflation-adjusted squad valuations).
* **Debugging Collaboration:** Gemini was also an active pairing partner during the development process—helping me troubleshoot broken matrix shapes, refine the time-decay math code blocks, and fix logic bugs in the interactive data-pull functions.

---

## 🛠️ Project Architecture & Tech Stack

* **Frontend UI:** Streamlit (For building the interactive web controls and sliders completely in Python).
* **Machine Learning:** XGBoost Classifier & Scikit-Learn (To map complex team matchups and predict win/draw/loss probabilities).
* **Data Manipulation:** Pandas & NumPy.

---

## 💻 Running the Project Locally

If you want to run this project on your machine instead of the live web link, follow these quick steps:

1. Clone or download this repository.
2. Ensure you have the required libraries installed via your terminal:
   ```bash
   pip install -r requirements.txt
