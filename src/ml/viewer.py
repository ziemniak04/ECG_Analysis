"""Tables and figures for the app. Only plotly and pandas here, no streamlit,
so it can be tested without opening the app.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ml.data import MITBIH_CLASSES, PTBXL_CLASSES
from src.ml.inference import RHYTHM_WINDOW

BEAT_NAMES = {"N": "normal", "S": "supraventricular", "V": "ventricular (PVC)",
              "F": "fusion", "Q": "unknown / paced"}
RHYTHM_NAMES = {"SR": "sinus rhythm", "AFIB": "atrial fibrillation",
                "STACH": "sinus tachycardia", "SBRAD": "sinus bradycardia"}
COLORS = {"N": "#9aa5b1", "S": "#e08a1e", "V": "#c0392b", "F": "#7d3c98", "Q": "#2e86c1",
          "SR": "#9aa5b1", "AFIB": "#c0392b", "STACH": "#e08a1e", "SBRAD": "#2e86c1"}
NORMAL = {"beats": "N", "rhythm": "SR"}  # everything else counts as flagged
LINE = "#2c3e50"
BAND = "#1b436e"


def _rect(x0, x1, color, opacity):
    # a shaded block over the whole plot height. Building the dict by hand is much
    # faster than fig.add_vrect when there are dozens of them
    return dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1,
                fillcolor=color, opacity=opacity, line_width=0, layer="below")


def fmt_time(seconds):
    m, s = divmod(round(float(seconds), 1), 60)
    return f"{int(m):02d}:{s:04.1f}"


def envelope(y, n_bins=2000):
    """Min/max per bucket, so a long signal can be drawn with a few thousand points
    and the peaks are still there. Returns sample positions and values."""
    n = len(y)
    if n <= 2 * n_bins:
        return np.arange(n), np.asarray(y)
    size = n // n_bins
    blocks = np.asarray(y[:size * n_bins]).reshape(n_bins, size)
    x = np.repeat(np.arange(n_bins) * size + size / 2, 2)
    out = np.empty(2 * n_bins)
    out[0::2] = blocks.min(1)
    out[1::2] = blocks.max(1)
    return x, out


# ------------------------------------------------------------------ tables
def beat_events(res, fs):
    """One row per classified beat (normal ones too)."""
    peaks, pred, proba = res["peaks"], res["pred"], res["proba"]
    all_peaks = res["all_peaks"]
    idx = np.searchsorted(all_peaks, peaks)
    rr = np.full(len(peaks), np.nan)
    ok = idx > 0
    rr[ok] = (peaks[ok] - all_peaks[idx[ok] - 1]) / fs
    t = peaks / fs
    return pd.DataFrame({"sample": peaks, "time_s": t, "time": [fmt_time(x) for x in t],
                         "class": [MITBIH_CLASSES[k] for k in pred],
                         "confidence": proba.max(1), "rr_before": rr})


def rhythm_events(res, fs):
    """One row per 10 s window."""
    starts, pred, proba = res["starts"], res["pred"], res["proba"]
    t = starts / fs
    df = pd.DataFrame({"sample": starts, "time_s": t, "time": [fmt_time(x) for x in t],
                       "class": [PTBXL_CLASSES[k] for k in pred], "confidence": proba.max(1)})
    for k, c in enumerate(PTBXL_CLASSES):
        df[f"p_{c}"] = proba[:, k]
    return df


# ----------------------------------------------------------------- figures
def overview_figure(a, lead, events, task, pos, length):
    """The whole recording (filtered, one lead) and under it where the flags are."""
    fs = a["fs"]
    sig = a["filtered"][:, lead]
    duration = len(sig) / fs
    x, y = envelope(sig)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=x / fs, y=y, mode="lines", line=dict(width=1, color=LINE),
                             showlegend=False), row=1, col=1)

    normal = NORMAL[task]
    flagged = events[events["class"] != normal]
    if task == "beats":
        n_bins = int(min(300, max(20, duration / 2)))
        edges = np.linspace(0, duration, n_bins + 1)
        for cls in MITBIH_CLASSES:
            t = flagged.loc[flagged["class"] == cls, "time_s"].to_numpy()
            if cls == normal or len(t) == 0:
                continue
            counts, _ = np.histogram(t, bins=edges)
            fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
                                 marker_color=COLORS[cls], name=cls), row=2, col=1)
        fig.update_yaxes(title_text="flagged", row=2, col=1)
    else:
        for cls in PTBXL_CLASSES:
            t = events.loc[events["class"] == cls, "time_s"].to_numpy()
            if len(t) == 0:
                continue
            fig.add_trace(go.Bar(x=t + RHYTHM_WINDOW / fs / 2, y=np.ones(len(t)),
                                 width=RHYTHM_WINDOW / fs, marker_color=COLORS[cls], name=cls),
                          row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)

    fig.update_layout(shapes=[_rect(pos, min(pos + length, duration), BAND, 0.2)])
    fig.update_layout(barmode="stack", height=270, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(range=[0, duration])
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    return fig


def detail_figure(a, lead, start, length, events, task):
    """Raw and filtered signal of one lead for a short stretch, with the classified beats / windows."""
    fs = a["fs"]
    i0 = int(start * fs)
    i1 = min(int((start + length) * fs), len(a["raw"]))
    t = np.arange(i0, i1) / fs

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("raw", "filtered (0.5-45 Hz)"))
    fig.add_trace(go.Scatter(x=t, y=a["raw"][i0:i1, lead], mode="lines",
                             line=dict(width=1, color="#7b8794"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=a["filtered"][i0:i1, lead], mode="lines",
                             line=dict(width=1, color=LINE), showlegend=False), row=2, col=1)

    normal = NORMAL[task]
    if task == "beats":
        inside = events[(events["sample"] >= i0) & (events["sample"] < i1)]
        for cls in MITBIH_CLASSES:
            part = inside[inside["class"] == cls]
            if part.empty:
                continue
            y = a["filtered"][part["sample"].to_numpy(), lead]
            if cls == normal:
                fig.add_trace(go.Scatter(x=part["time_s"], y=y, mode="markers", name=cls,
                                         marker=dict(size=5, color=COLORS[cls])), row=2, col=1)
            else:
                fig.add_trace(go.Scatter(x=part["time_s"], y=y, mode="markers+text", name=cls,
                                         text=[cls] * len(part), textposition="top center",
                                         textfont=dict(color=COLORS[cls], size=13),
                                         marker=dict(size=9, color=COLORS[cls])), row=2, col=1)
        # light red block behind every flagged beat
        shapes = [_rect(t - 0.25, t + 0.45, COLORS["V"], 0.08)
                  for t in inside.loc[inside["class"] != normal, "time_s"]]
        fig.update_layout(shapes=shapes)
    else:
        shapes, notes = [], []
        near = events[(events["sample"] + RHYTHM_WINDOW > i0) & (events["sample"] < i1)]
        for w0, cls in zip(near["sample"], near["class"]):
            shapes.append(_rect(w0 / fs, (w0 + RHYTHM_WINDOW) / fs, COLORS[cls],
                                0.05 if cls == normal else 0.15))
            notes.append(dict(x=max(w0, i0) / fs, y=1.0, yref="paper", text=cls, showarrow=False,
                              xanchor="left", yanchor="bottom",
                              font=dict(color=COLORS[cls], size=13)))
        fig.update_layout(shapes=shapes)
        fig.update_layout(annotations=list(fig.layout.annotations) + notes)

    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.14), hovermode="x")
    return fig


def all_leads_figure(a, lead_names, start, length):
    """All 12 leads (filtered) one under the other."""
    fs = a["fs"]
    i0 = int(start * fs)
    i1 = min(int((start + length) * fs), len(a["raw"]))
    t = np.arange(i0, i1) / fs
    n = len(lead_names)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.004)
    for k, name in enumerate(lead_names):
        fig.add_trace(go.Scatter(x=t, y=a["filtered"][i0:i1, k], mode="lines",
                                 line=dict(width=1, color=LINE), showlegend=False), row=k + 1, col=1)
        fig.update_yaxes(title_text=name, showticklabels=False, showgrid=False,
                         title_standoff=2, row=k + 1, col=1)
    fig.update_xaxes(title_text="time (s)", row=n, col=1)
    fig.update_layout(height=780, margin=dict(l=10, r=10, t=10, b=10), hovermode="x")
    return fig