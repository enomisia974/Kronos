"""Streamlit dashboard — reads from DB, zero ML compute."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from kronos_system.data.database import (
    read_asset_catalog, read_latest_prediction, read_prediction_history,
    read_rolling_metrics, read_latest_sentiment, read_prices,
)

st.set_page_config(page_title="Kronos Dashboard", layout="wide")
st.title("Kronos Quantitative Research")
st.caption(f"Ultimo aggiornamento: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

assets = [dict(r) for r in read_asset_catalog()]
asset_ids = [a["id"] for a in assets]

selected = st.sidebar.selectbox("Seleziona Asset", asset_ids, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Catalogo Asset**")
for a in assets:
    st.sidebar.write(f"{a['id']} — {a['asset_type']}")

col1, col2, col3 = st.columns(3)

latest = read_latest_prediction(selected)
sent = read_latest_sentiment(selected)

if latest:
    latest = dict(latest)
    prob = latest["probability"]
    signal = latest["signal"]
    conf = latest["confidence"]

    sig_color = "#22c55e" if prob > 0.6 else "#ef4444" if prob < 0.4 else "#f59e0b"
    with col1:
        st.metric("Segnale", signal, f"{prob*100:.1f}%", delta_color="off")
        st.markdown(f"<div style='background:{sig_color};height:8px;border-radius:4px;width:{prob*100:.0f}%'></div>",
                    unsafe_allow_html=True)
        st.caption(f"Confidenza: {conf*100:.1f}%")

    with col2:
        st.metric("Probabilità Rialzo", f"{prob*100:.1f}%")
        st.caption(f"Versione modello: {latest['model_version']}")

    with col3:
        st.metric("Data", latest["date"])

    # Sentiment
    if sent:
        sent = dict(sent)
        st.sidebar.metric("Sentiment Ultimo", f"{sent['score']:+.3f}", f"{sent['count']} notizie")
else:
    st.warning(f"Nessuna predizione disponibile per {selected}")

# Metrics
st.subheader("Walk-Forward Validation")
metrics = read_rolling_metrics(selected)
if metrics:
    mdf = pd.DataFrame([dict(r) for r in metrics])
    acc_mean = mdf["accuracy"].mean()
    acc_std = mdf["accuracy"].std()

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Accuratezza Media", f"{acc_mean*100:.1f}%", f"±{acc_std*100:.1f}%")
    col_m2.metric("Fold", int(mdf["fold"].max()) + 1)
    col_m3.metric("Baseline Media", f"{mdf['baseline_accuracy'].mean()*100:.1f}%")
    col_m4.metric("Componenti PCA", int(mdf["n_components_pca"].iloc[-1]) if "n_components_pca" in mdf.columns else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mdf["fold"], y=mdf["accuracy"], name="Accuratezza",
                         marker_color="#3b82f6"))
    fig.add_trace(go.Scatter(x=mdf["fold"], y=mdf["baseline_accuracy"],
                             mode="lines+markers", name="Baseline (naive)",
                             line=dict(color="#ef4444", dash="dot")))
    fig.update_layout(template="none", height=300,
                      xaxis_title="Fold", yaxis_title="Accuratezza",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nessuna metrica disponibile. Esegui il training prima.")

# Prediction history
st.subheader("Cronologia Predizioni")
history = read_prediction_history(selected, 60)
if history:
    hdf = pd.DataFrame([dict(r) for r in history])
    hdf["date"] = pd.to_datetime(hdf["date"])
    hdf = hdf.sort_values("date")

    fig2 = go.Figure()
    colors = ["#22c55e" if p > 0.6 else "#ef4444" if p < 0.4 else "#f59e0b"
              for p in hdf["probability"]]
    fig2.add_trace(go.Scatter(x=hdf["date"], y=hdf["probability"] * 100,
                              mode="lines+markers",
                              marker=dict(color=colors, size=8),
                              line=dict(color="#94a3b8", width=1),
                              name="Probabilità"))
    fig2.add_hline(y=60, line_dash="dash", line_color="#22c55e", opacity=0.5)
    fig2.add_hline(y=40, line_dash="dash", line_color="#ef4444", opacity=0.5)
    fig2.update_layout(template="none", height=250,
                       yaxis_title="Probabilità (%)",
                       hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Nessuna cronologia disponibile.")

# Price chart
st.subheader("Prezzo")
prices = read_prices(selected, 120)
if prices:
    pdf = pd.DataFrame([dict(r) for r in prices])
    pdf = pdf.sort_values("date")
    fig3 = go.Figure()
    fig3.add_trace(go.Candlestick(
        x=pdf["date"], open=pdf["open"], high=pdf["high"],
        low=pdf["low"], close=pdf["close"], name="Prezzo"
    ))
    fig3.update_layout(template="none", height=400,
                       xaxis_rangeslider_visible=False)
    st.plotly_chart(fig3, use_container_width=True)
