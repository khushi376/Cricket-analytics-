# 🏏 Cricket Player Analytics Platform

A data science platform for analysing cricket player performance 
across IPL, T20I, and ODI formats — built for franchise scouts, 
coaches, and recruiters.

## 🔴 Live Demo
[Click here to open the app](YOUR_STREAMLIT_URL_HERE)

## 📊 What it does
- Search any player across IPL + International formats
- View career stats, batting avg, strike rate, impact score
- Compare 2 players head-to-head with radar chart
- Scout players by role, format, and performance filters
- Phase-wise analysis (powerplay / middle / death overs)

## 🧠 ML Models
- **Impact Score** — Random Forest model (R² = 0.991)
- **Player Clustering** — K-Means (5 play styles)

## 📦 Data Sources
- Kaggle IPL Dataset (2008–2025)
- Cricsheet.org — T20I, ODI (Ashwin CSV format)

## 🛠️ Tech Stack
Python • Pandas • Scikit-learn • XGBoost • Streamlit • Plotly

## 🚀 Run locally
pip install -r requirements.txt
streamlit run app/streamlit_app.py