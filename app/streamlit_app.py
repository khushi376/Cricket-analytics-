import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# ── Page config ──
st.set_page_config(
    page_title="Cricket Analytics Platform",
    page_icon="🏏",
    layout="wide"
)

# ── Load data ──
@st.cache_data
def load_data():
    features = pd.read_csv("data/processed/player_features.csv")
    try:
        master = pd.read_csv("data/processed/ipl_all.csv")
        # Add phase column if missing
        if "phase" not in master.columns:
            master["over"] = master["ball"].astype(str)\
                             .str.split(".").str[0].astype(int)
            master["phase"] = master["over"].apply(
                lambda x: "powerplay" if x < 6
                else ("death" if x >= 15 else "middle")
            )
        # Add format column if missing
        if "format" not in master.columns:
            master["format"] = "IPL"
    except FileNotFoundError:
        master = pd.DataFrame()
    return features, master

@st.cache_resource
def load_models():
    model, kmeans = None, None
    try:
        with open("data/models/impact_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        pass
    try:
        with open("data/models/kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)
    except FileNotFoundError:
        pass
    return model, kmeans

features, master = load_data()
model, kmeans = load_models()

CLUSTER_LABELS = {
    0: "⚡ Aggressive Batter",
    1: "🛡️ Anchor Batter",
    2: "🎯 Death Bowler",
    3: "🌀 Spinner",
    4: "🔄 Allrounder"
}

# ── Sidebar ──
st.sidebar.title("🏏 Cricket Analytics")
page = st.sidebar.radio("Navigate", [
    "🏠 Player Profile",
    "📈 Performance Charts",
    "🆚 Compare Players",
    "🔍 Scout Players"
])

all_players = sorted(features["player"].dropna().unique())
selected_player = st.sidebar.selectbox("Search Player", all_players)
selected_format = st.sidebar.selectbox("Format", ["All", "IPL", "T20I", "ODI"])

def get_player_data(player, fmt):
    if fmt == "All":
        return features[features["player"] == player]
    return features[
        (features["player"] == player) &
        (features["format"] == fmt)
    ]

def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except Exception:
        return default

# ════════════════════════════════
# PAGE 1 — Player Profile
# ════════════════════════════════
if page == "🏠 Player Profile":
    pdata = get_player_data(selected_player, selected_format)

    if pdata.empty:
        st.warning(f"No {selected_format} data found for {selected_player}.")
        st.stop()

    st.title(f"🏏 {selected_player}")
    row = pdata.iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Impact Score", f"{safe_float(row.get('impact_score')):.1f}")
    col2.metric("Matches",      int(safe_float(row.get("matches"))))
    col3.metric("Batting Avg",  f"{safe_float(row.get('batting_avg')):.1f}")
    col4.metric("Strike Rate",  f"{safe_float(row.get('avg_sr')):.1f}")
    col5.metric("Role",         str(row.get("role", "—")))

    cluster = row.get("cluster", None)
    try:
        if cluster is not None and not np.isnan(float(cluster)):
            label = CLUSTER_LABELS.get(int(float(cluster)), f"Cluster {int(float(cluster))}")
            st.info(f"🎭 Play Style: **{label}**")
    except Exception:
        pass

    st.divider()

    # Format comparison
    all_fmt = features[features["player"] == selected_player]
    if len(all_fmt) > 1:
        st.subheader("Format Comparison")
        display_cols = ["format", "matches", "batting_avg", "avg_sr",
                        "total_runs", "fifties", "hundreds",
                        "economy", "wickets", "impact_score"]
        cols_present = [c for c in display_cols if c in all_fmt.columns]
        st.dataframe(
            all_fmt[cols_present].set_index("format").round(2),
            use_container_width=True
        )

    # Batting stats
    st.subheader("Batting Stats")
    bat_cols = ["total_runs", "total_balls", "batting_avg", "avg_sr",
                "highest_score", "fifties", "hundreds",
                "avg_boundary", "avg_dot_pct", "consistency"]
    bat_cols = [c for c in bat_cols if c in pdata.columns]
    st.dataframe(pdata[bat_cols].round(2), use_container_width=True)

    # Bowling stats
    wickets_val = safe_float(pdata["wickets"].sum() if "wickets" in pdata.columns else 0)
    if wickets_val > 0:
        st.subheader("Bowling Stats")
        bowl_cols = ["wickets", "overs_bowled", "economy",
                     "bowling_avg", "strike_rate_bowl", "dot_pct_bowl"]
        bowl_cols = [c for c in bowl_cols if c in pdata.columns]
        st.dataframe(pdata[bowl_cols].round(2), use_container_width=True)

# ════════════════════════════════
# PAGE 2 — Performance Charts
# ════════════════════════════════
elif page == "📈 Performance Charts":
    st.title(f"📈 {selected_player} — Performance Charts")

    if master.empty:
        st.warning("Ball-by-ball data not available on this deployment.")
        st.stop()

    player_balls = master[master["striker"] == selected_player].copy()

    if player_balls.empty:
        st.warning(f"No ball-by-ball data found for {selected_player}. "
                   f"Note: player names use short form e.g. 'V Kohli', 'RG Sharma'.")
        st.stop()

    # Fix is_wide
    player_balls["is_wide"] = player_balls["wides"].notna() & \
                               (player_balls["wides"].astype(str) != "0") & \
                               (player_balls["wides"].astype(str) != "nan")

    # Chart 1 — Season runs
    season_runs = (
        player_balls.groupby("season")["runs_off_bat"]
        .sum().reset_index()
    )
    fig1 = px.bar(
        season_runs, x="season", y="runs_off_bat",
        title=f"{selected_player} — IPL Runs per Season",
        labels={"runs_off_bat": "Runs", "season": "Season"},
        color_discrete_sequence=["#1D9E75"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2 — Phase strike rate
    legal_balls = player_balls[~player_balls["is_wide"]].copy()
    if not legal_balls.empty and "phase" in legal_balls.columns:
        phase_stats = (
            legal_balls.groupby("phase")
            .agg(runs=("runs_off_bat", "sum"), balls=("ball", "count"))
            .reset_index()
        )
        phase_stats = phase_stats[phase_stats["balls"] > 0]
        phase_stats["sr"] = (phase_stats["runs"] / phase_stats["balls"] * 100).round(1)

        phase_order = ["powerplay", "middle", "death"]
        phase_stats["phase"] = pd.Categorical(
            phase_stats["phase"], categories=phase_order, ordered=True
        )
        phase_stats = phase_stats.sort_values("phase")

        fig2 = px.bar(
            phase_stats, x="phase", y="sr",
            color="phase",
            title=f"{selected_player} — Strike Rate by Phase",
            labels={"sr": "Strike Rate", "phase": "Phase"},
            color_discrete_map={
                "powerplay": "#1D9E75",
                "middle": "#378ADD",
                "death": "#E05C2A"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 — Innings distribution
    innings_runs = (
        player_balls.groupby("match_id")["runs_off_bat"]
        .sum().reset_index()
    )
    fig3 = px.histogram(
        innings_runs, x="runs_off_bat",
        title=f"{selected_player} — Innings Score Distribution",
        labels={"runs_off_bat": "Runs in Innings"},
        nbins=30,
        color_discrete_sequence=["#378ADD"]
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════
# PAGE 3 — Compare Players
# ════════════════════════════════
elif page == "🆚 Compare Players":
    st.title("🆚 Compare Players")

    col1, col2 = st.columns(2)
    p1 = col1.selectbox("Player 1", all_players, index=0)
    p2 = col2.selectbox("Player 2", all_players, index=min(1, len(all_players)-1))
    fmt = st.selectbox("Format", ["IPL", "T20I", "ODI"])

    d1 = features[(features["player"] == p1) & (features["format"] == fmt)]
    d2 = features[(features["player"] == p2) & (features["format"] == fmt)]

    if d1.empty or d2.empty:
        st.warning("One or both players have no data in this format. Try a different format.")
        st.stop()

    r1, r2 = d1.iloc[0], d2.iloc[0]

    RADAR_STATS = ["batting_avg", "avg_sr", "consistency",
                   "avg_boundary", "impact_score", "economy"]
    labels = ["Batting Avg", "Strike Rate", "Consistency",
              "Boundary %", "Impact Score", "Economy"]

    v1 = [safe_float(r1.get(s)) for s in RADAR_STATS]
    v2 = [safe_float(r2.get(s)) for s in RADAR_STATS]

    combined = [max(a, b) or 1 for a, b in zip(v1, v2)]
    v1n = [a / m * 100 for a, m in zip(v1, combined)]
    v2n = [a / m * 100 for a, m in zip(v2, combined)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v1n + [v1n[0]], theta=labels + [labels[0]],
        fill="toself", name=p1
    ))
    fig.add_trace(go.Scatterpolar(
        r=v2n + [v2n[0]], theta=labels + [labels[0]],
        fill="toself", name=p2, opacity=0.7
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"{p1} vs {p2} — {fmt}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stats Comparison")
    comp = pd.DataFrame({
        "Stat": labels,
        p1: [round(v, 1) for v in v1],
        p2: [round(v, 1) for v in v2]
    })
    st.dataframe(comp.set_index("Stat"), use_container_width=True)

# ════════════════════════════════
# PAGE 4 — Scout Players
# ════════════════════════════════
elif page == "🔍 Scout Players":
    st.title("🔍 Scout Players")
    st.caption("Filter players by role, format, and performance thresholds")

    col1, col2, col3 = st.columns(3)
    f_format = col1.selectbox("Format", ["IPL", "T20I", "ODI"])
    f_role   = col2.selectbox("Role", ["All", "Batter", "Bowler", "Allrounder"])
    f_min_m  = col3.slider("Min Matches", 1, 50, 10)

    col4, col5 = st.columns(2)
    f_min_sr  = col4.slider("Min Strike Rate", 0, 200, 100)
    f_min_imp = col5.slider("Min Impact Score", 0, 100, 20)

    filtered = features[features["format"] == f_format].copy()
    filtered = filtered[filtered["matches"] >= f_min_m]
    filtered = filtered[filtered["avg_sr"].fillna(0) >= f_min_sr]
    filtered = filtered[filtered["impact_score"].fillna(0) >= f_min_imp]

    if f_role != "All":
        filtered = filtered[filtered["role"] == f_role]

    def get_cluster_label(x):
        try:
            if pd.notna(x):
                return CLUSTER_LABELS.get(int(float(x)), "—")
        except Exception:
            pass
        return "—"

    filtered["cluster_label"] = filtered["cluster"].apply(get_cluster_label)

    display = ["player", "role", "matches", "batting_avg", "avg_sr",
               "total_runs", "wickets", "economy", "impact_score", "cluster_label"]
    display = [c for c in display if c in filtered.columns]

    st.markdown(f"**{len(filtered)} players found**")
    st.dataframe(
        filtered[display].sort_values("impact_score", ascending=False)
        .reset_index(drop=True).round(2),
        use_container_width=True
    )

    csv = filtered[display].to_csv(index=False)
    st.download_button(
        "⬇️ Download as CSV",
        csv,
        file_name=f"scout_{f_format}_{f_role}.csv",
        mime="text/csv"
    )