"""Plotly charts styled to match the MediGuard palette."""
from __future__ import annotations

import plotly.graph_objects as go

BLUE, NAVY, TEAL = "#1769E0", "#142B4F", "#0D9488"
RED, AMBER, GREEN = "#E05A5A", "#E3A008", "#2FA36B"
GRID = "#E6EEF9"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def class_color(name: str) -> str:
    n = name.lower()
    return RED if n == "fake" else GREEN if n == "real" else BLUE


def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=NAVY, size=13),
        title_font=dict(size=15, color=NAVY), hoverlabel=dict(font_family=FONT),
        legend=dict(orientation="h", y=-0.15), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def class_distribution(counts: dict[str, int], title: str = "Class distribution") -> go.Figure:
    labels = list(counts)
    fig = go.Figure(go.Pie(labels=labels, values=[counts[k] for k in labels], hole=0.62,
                           marker=dict(colors=[class_color(k) for k in labels], line=dict(color="#fff", width=3)),
                           textinfo="label+value", sort=False))
    fig.add_annotation(text=f"<b>{sum(counts.values())}</b><br>images", showarrow=False, font=dict(size=15, color=NAVY))
    return _layout(fig, title=title, showlegend=False)


def probability_bars(scores: dict[str, float]) -> go.Figure:
    names = list(scores)
    vals = [scores[n] * 100 for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", text=[f"{v:.1f}%" for v in vals],
                           textposition="outside", marker=dict(color=[class_color(n) for n in names]),
                           hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_xaxes(range=[0, 115], ticksuffix="%")
    return _layout(fig, height=170, title="Model class scores")


def confusion_matrix(matrix: list[list[int]], labels: list[str], title="Confusion matrix (test set)") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
                               colorscale=[[0, "#F5F9FF"], [1, BLUE]], showscale=False,
                               text=matrix, texttemplate="%{text}", textfont=dict(size=18)))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=340, title=title)


def training_curves(history: dict, metric: str = "accuracy") -> go.Figure:
    fig = go.Figure()
    train = history.get(metric) or []
    val = history.get(f"val_{metric}") or []
    epochs = list(range(1, len(train) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=train, mode="lines+markers", name=f"Training {metric}", line=dict(color=BLUE, width=3)))
    if val:
        fig.add_trace(go.Scatter(x=epochs, y=val, mode="lines+markers", name=f"Validation {metric}", line=dict(color=TEAL, width=3, dash="dot")))
    if history.get("fine_tune_start_epoch"):
        fig.add_vline(x=history["fine_tune_start_epoch"] + 0.5, line_dash="dash", line_color=AMBER,
                      annotation_text="Fine-tuning starts", annotation_position="top")
    return _layout(fig, title=f"{metric.capitalize()} per epoch", xaxis_title="Epoch")


def metric_bars(metrics: dict[str, float]) -> go.Figure:
    names = list(metrics)
    vals = [metrics[n] * 100 for n in names]
    fig = go.Figure(go.Bar(x=names, y=vals, text=[f"{v:.1f}%" for v in vals], textposition="outside",
                           marker=dict(color=[BLUE, TEAL, NAVY, "#5B8DEF"][: len(names)])))
    fig.update_yaxes(range=[0, 110], ticksuffix="%")
    return _layout(fig, height=300, title="Test-set metrics")


def history_timeline(df) -> go.Figure:
    fig = go.Figure()
    if df is not None and len(df):
        d = df.copy()
        d["day"] = d["timestamp"].str[:10]
        grouped = d.groupby(["day", d["predicted_class"].fillna("Not classified")]).size().unstack(fill_value=0)
        for col in grouped.columns:
            fig.add_trace(go.Bar(x=grouped.index, y=grouped[col], name=col,
                                 marker_color=class_color(col) if col != "Not classified" else "#9FB3D1"))
        fig.update_layout(barmode="stack")
    return _layout(fig, height=280, title="Analyses per day")
