# dashboard.py
"""
Streamlit dashboard for monitoring Flappy Bird PPO runs.

Usage:
    streamlit run dashboard.py
"""

import os
import json
from typing import List

import pandas as pd
import streamlit as st


# ----------------------------
# UTILITIES
# ----------------------------

LOG_DIR = "logs"


def list_runs(log_dir: str = LOG_DIR) -> List[str]:
    """Return list of run directories inside logs/."""
    if not os.path.exists(log_dir):
        return []
    runs = []
    for name in os.listdir(log_dir):
        run_path = os.path.join(log_dir, name)
        if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, "metrics.csv")):
            runs.append(name)
    runs.sort()
    return runs


def load_metrics(run_name: str) -> pd.DataFrame:
    path = os.path.join(LOG_DIR, run_name, "metrics.csv")
    df = pd.read_csv(path)
    # Ensure numeric columns for good plotting
    for col in df.columns:
        if col == "step":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # try numeric, fallback to original
            new_col = pd.to_numeric(df[col], errors="ignore")
            df[col] = new_col
    return df


def load_hparams(run_name: str):
    path = os.path.join(LOG_DIR, run_name, "hparams.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def list_videos(run_name: str) -> List[str]:
    videos_dir = os.path.join(LOG_DIR, run_name, "videos")
    if not os.path.exists(videos_dir):
        return []
    videos = [
        f for f in os.listdir(videos_dir)
        if f.lower().endswith(".mp4")
    ]
    videos.sort()
    return [os.path.join(videos_dir, v) for v in videos]


# ----------------------------
# STREAMLIT APP
# ----------------------------

st.set_page_config(
    page_title="Flappy Bird PPO Dashboard",
    layout="wide",
)

st.title("🐤 Flappy Bird PPO Training Dashboard")

# Sidebar: run selection
st.sidebar.header("Run Selection")

runs = list_runs()
if not runs:
    st.sidebar.warning(
        "No runs found in 'logs/'. Start a training run with logging first.")
    st.stop()

selected_run = st.sidebar.selectbox("Choose a run", runs, index=len(runs) - 1)

st.sidebar.markdown(f"**Selected run:** `{selected_run}`")

# Load data for selected run
df = load_metrics(selected_run)
hparams = load_hparams(selected_run)
videos = list_videos(selected_run)

# Let user choose which metric(s) to show
all_metric_cols = [c for c in df.columns if c != "step"]
default_metrics = [c for c in all_metric_cols if c in [
    "avg_reward", "policy_loss", "value_loss", "entropy"]]
if not default_metrics:
    default_metrics = all_metric_cols[:4]

selected_metrics = st.sidebar.multiselect(
    "Metrics to plot (time series)",
    options=all_metric_cols,
    default=default_metrics,
)

# Tabs for structure
tab_overview, tab_metrics, tab_compare, tab_videos = st.tabs(
    ["📌 Overview", "📈 Metrics", "📊 Compare Runs", "🎬 Videos"]
)

# ----------------------------
# OVERVIEW TAB
# ----------------------------
with tab_overview:
    st.subheader("Run Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Hyperparameters")
        if hparams:
            # Show as nice table
            hp_df = pd.DataFrame(
                [{"parameter": k, "value": v} for k, v in hparams.items()]
            )
            st.table(hp_df)
        else:
            st.info("No hparams.json found for this run.")

    with col2:
        st.markdown("### Training Summary")

        if "step" in df.columns:
            max_step = int(df["step"].max())
        else:
            max_step = len(df)

        avg_reward_col = "avg_reward" if "avg_reward" in df.columns else None
        latest_reward = df[avg_reward_col].iloc[-1] if avg_reward_col else None

        bullet_points = []

        bullet_points.append(f"- **Total logged steps:** {max_step:,}")

        if avg_reward_col:
            bullet_points.append(
                f"- **Last logged avg_reward:** {latest_reward:.2f}")

        if "entropy" in df.columns:
            entropy_last = df["entropy"].iloc[-1]
            bullet_points.append(f"- **Last entropy:** {entropy_last:.4f}")

        if "policy_loss" in df.columns:
            pl = df["policy_loss"].iloc[-1]
            bullet_points.append(f"- **Last policy loss:** {pl:.4f}")

        if "value_loss" in df.columns:
            vl = df["value_loss"].iloc[-1]
            bullet_points.append(f"- **Last value loss:** {vl:.4f}")

        st.markdown("\n".join(bullet_points)
                    if bullet_points else "Not enough data to summarize yet.")

    st.markdown("---")
    st.markdown("### Quick Reward Plot")

    if "step" in df.columns and "avg_reward" in df.columns:
        chart_df = df[["step", "avg_reward"]].dropna()
        chart_df = chart_df.set_index("step")
        st.line_chart(chart_df, height=300)
    else:
        st.info(
            "Need 'step' and 'avg_reward' columns in metrics.csv to show this plot.")

# ----------------------------
# METRICS TAB
# ----------------------------
with tab_metrics:
    st.subheader("Time Series Metrics")

    if "step" in df.columns and selected_metrics:
        for metric in selected_metrics:
            if metric not in df.columns:
                continue
            st.markdown(f"#### {metric}")
            chart_df = df[["step", metric]].dropna()
            if chart_df.empty:
                st.info(f"No data for metric '{metric}'.")
                continue
            chart_df = chart_df.set_index("step")
            st.line_chart(chart_df, height=250)
            st.markdown("---")
    else:
        st.info("No metrics selected or 'step' column missing.")

    st.markdown("### Raw Metrics Table")
    st.dataframe(df.tail(200))

# ----------------------------
# COMPARE RUNS TAB
# ----------------------------
with tab_compare:
    st.subheader("Compare Runs")

    compare_runs = st.multiselect(
        "Select runs to compare",
        options=runs,
        default=[selected_run],
    )

    metric_for_compare = st.selectbox(
        "Metric to compare across runs",
        options=all_metric_cols,
        index=all_metric_cols.index(
            "avg_reward") if "avg_reward" in all_metric_cols else 0,
    )

    if compare_runs:
        combined = []
        for rn in compare_runs:
            try:
                df_rn = load_metrics(rn)
                if metric_for_compare not in df_rn.columns:
                    continue
                sub = df_rn[["step", metric_for_compare]].copy()
                sub["run"] = rn
                combined.append(sub)
            except Exception as e:
                st.warning(f"Could not load run '{rn}': {e}")

        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            # Pivot-like view
            st.markdown(f"#### {metric_for_compare} vs step for selected runs")
            # We'll keep it simple: filter by run using Streamlit's native features
            # Convert for plotting:
            # wide form: index=step, columns=run, values=metric
            pivot = combined_df.pivot_table(
                index="step",
                columns="run",
                values=metric_for_compare,
                aggfunc="mean",
            )
            st.line_chart(pivot, height=400)
        else:
            st.info("No comparable data found for the selected runs / metric.")
    else:
        st.info("Select at least one run to compare.")

# ----------------------------
# VIDEOS TAB
# ----------------------------
with tab_videos:
    st.subheader("Evaluation Videos")

    if not videos:
        st.info(
            "No videos found for this run.\n\n"
            "You can save evaluation MP4s into:\n"
            f"`logs/{selected_run}/videos/`"
        )
    else:
        nice_names = [os.path.basename(v) for v in videos]
        selected_video_name = st.selectbox("Choose a video", nice_names)
        selected_video_path = videos[nice_names.index(selected_video_name)]

        st.markdown(f"**Playing:** `{selected_video_name}`")
        with open(selected_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
