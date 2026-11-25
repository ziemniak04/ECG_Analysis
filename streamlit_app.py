"""ECG viewer: load a recording, see the whole signal and every beat / window
that the model flagged.

Run from the repo root:   streamlit run streamlit_app.py
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from src.ml import inference as inf
from src.ml import viewer as vw
from src.ml.data import ROOT

st.set_page_config(page_title="ECG analysis", layout="wide")

MODEL_LABELS = {
    "mitbih_rr": "MIT-BIH, with RR timing",
    "mitbih": "MIT-BIH, signal only",
    "ptbxl": "PTB-XL rhythm",
}


@st.cache_resource
def get_model(name):
    return inf.load_model(name)


@st.cache_data
def model_list():
    return inf.list_models()


@st.cache_data
def local_examples():
    files = {}
    for f in sorted(ROOT.glob("data/mitdb/**/*.hea")):
        files[f"MIT-BIH {f.stem}"] = str(f)
    for f in sorted(ROOT.glob("data/ptbxl/records100/*/*_lr.hea"))[:500]:
        files[f"PTB-XL {f.stem}"] = str(f)
    return files


def run_analysis(task, model_name, signal, fs, names, use_resample, max_minutes):
    t0 = time.perf_counter()
    model = get_model(model_name)
    target = inf.MITBIH_FS if task == "beats" else inf.PTBXL_FS
    notes = []

    if task == "rhythm":
        signal, note = inf.order_leads(signal, names)
        if note:
            notes.append(note)
        lead_names = list(inf.PTBXL_LEADS)
    else:
        lead_names = names if names else ["lead 1", "lead 2"]

    if abs(fs - target) > 1e-6:
        signal = inf.resample(signal, fs, target)
        notes.append(
            f"Resampled from {fs:g} Hz to {target} Hz. The models never saw resampled "
            "data in training, so treat the predictions with extra care."
        )
    signal = signal[:int(max_minutes * 60 * target)]

    filtered, normalized = inf.prepare(signal, target)
    if task == "beats":
        peaks = inf.detect_peaks(normalized, target)
        t_model = time.perf_counter()
        res = inf.classify_beats(model, normalized, peaks)
        res["all_peaks"] = peaks
        events = vw.beat_events(res, target)
    else:
        t_model = time.perf_counter()
        res = inf.classify_rhythm(model, filtered)
        events = vw.rhythm_events(res, target)

    return {
        "raw": signal,
        "filtered": filtered,
        "fs": target,
        "notes": notes,
        "lead_names": lead_names,
        "result": res,
        "events": events,
        "task": task,
        "model_ms": (time.perf_counter() - t_model) * 1000,
        "total_s": time.perf_counter() - t0,
    }


def split_note(task, source):
    part = None
    if task == "beats" and source.isdigit():
        part = inf.split_of_mitbih_record(source)
    elif task == "rhythm" and source.split("_")[0].isdigit():
        part = inf.split_of_ptbxl_ecg(source.split("_")[0])
    return {
        "train": "This record was in the training split, so the model has already seen it. "
        "Results on it are better than they would be on a new patient.",
        "val": "This record was in the validation split.",
        "test": "This record was in the test split, the model never saw it in training.",
    }.get(part)


def test_results(model_name):
    path = ROOT / "results" / "ml" / f"{model_name}_test_metrics.json"
    return json.loads(path.read_text()) if path.is_file() else None


problem = None
data = None
use_resample = False

with st.sidebar:
    st.title("ECG analysis")
    st.subheader("ECG file")
    source_mode = st.radio(
        "Source", ["Upload", "Local record"], horizontal=True, label_visibility="collapsed"
    )
    try:
        if source_mode == "Upload":
            uploads = st.file_uploader(
                "Files",
                type=["hea", "dat", "csv"],
                accept_multiple_files=True,
                help="A WFDB record needs the .hea and the .dat file together. "
                "A csv needs one column per lead.",
            )
            csvs = [u for u in uploads if u.name.lower().endswith(".csv")]
            if csvs:
                data = inf.read_csv_file(csvs[0].getvalue(), csvs[0].name)
                data["fs"] = st.number_input("Sampling rate of the csv (Hz)", 1.0, 5000.0, 360.0)
            elif uploads:
                data = inf.read_wfdb_files({u.name: u.getvalue() for u in uploads})
        else:
            examples = local_examples()
            if examples:
                choice = st.selectbox("Record", list(examples))
                data = inf.read_wfdb_path(examples[choice])
            else:
                st.caption("No records found in the data folder.")
    except ValueError as e:
        problem = str(e)

    model_box = st.container()
    with st.expander("Advanced"):
        use_resample = st.checkbox("Resample to the model's rate if it differs", value=False)
        max_minutes = st.slider("Analyse at most (minutes)", 1, 60, 30)

if problem:
    st.error(problem)
    st.stop()

models = model_list()
if not models:
    st.error("No trained models in the models folder. Train them first, see docs/ml_pipeline.md.")
    st.stop()

if data is None:
    st.title("ECG analysis")
    st.write(
        "Choose a recording in the sidebar. You can upload a WFDB record (the .hea and .dat "
        "file together) or a csv with one column per lead, or open a record from the data "
        "folder on this computer."
    )
    st.write(
        "Two-lead recordings at 360 Hz are analysed beat by beat. Twelve-lead recordings at "
        "100 Hz are analysed in 10 second windows."
    )
    st.stop()

n_leads = data["signal"].shape[1]
if n_leads == 2:
    task, target = "beats", inf.MITBIH_FS
elif n_leads == 12:
    task, target = "rhythm", inf.PTBXL_FS
else:
    st.error(f"The file has {n_leads} leads. The models need 2 leads (beats) or 12 leads (rhythm).")
    st.stop()

candidates = [m for m, c in models.items() if c == n_leads]
if not candidates:
    st.error(f"No trained model for {n_leads} leads in the models folder.")
    st.stop()
with model_box:
    st.subheader("Model")
    default = candidates.index("mitbih_rr") if "mitbih_rr" in candidates else 0
    model_name = st.selectbox(
        "Model",
        candidates,
        index=default,
        label_visibility="collapsed",
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )

if abs(data["fs"] - target) > 1e-6 and not use_resample:
    st.error(
        f"The file is sampled at {data['fs']:g} Hz, the model needs {target} Hz. "
        "Tick 'Resample' under Advanced in the sidebar to continue anyway."
    )
    st.stop()

fingerprint = hashlib.md5(data["signal"].tobytes()).hexdigest()[:10]
key = (task, model_name, fingerprint, round(data["fs"], 3), use_resample, max_minutes)
if st.session_state.get("analysis_key") != key:
    try:
        with st.spinner("Analysing..."):
            st.session_state["analysis"] = run_analysis(
                task, model_name, data["signal"], data["fs"], data["names"], use_resample, max_minutes
            )
    except ValueError as e:
        st.session_state["analysis_key"] = None
        st.error(str(e))
        st.stop()
    st.session_state["analysis_key"] = key
    ev0 = st.session_state["analysis"]["events"]
    flagged0 = ev0[ev0["class"] != vw.NORMAL[task]]
    st.session_state["pos"] = float(max(0.0, flagged0["time_s"].iloc[0] - 5)) if len(flagged0) else 0.0
    st.session_state["last_jump"] = None

a = st.session_state["analysis"]
events = a["events"]
fs = a["fs"]
duration = len(a["raw"]) / fs
normal = vw.NORMAL[task]
flagged = events[events["class"] != normal]
uid = f"{fingerprint}_{model_name}"

st.header(f"{data['source']}")
st.caption(
    f"{duration / 60:.1f} min at {fs} Hz, {n_leads} leads, "
    f"model: {MODEL_LABELS.get(model_name, model_name)}"
)
note = split_note(task, data["source"])
if note:
    st.info(note)
for n in a["notes"]:
    st.info(n)

c1, c2, c3 = st.columns(3)
if task == "beats":
    hr = 60 * fs / np.median(np.diff(a["result"]["all_peaks"]))
    c1.metric("Beats", len(events))
    c2.metric("Median heart rate", f"{hr:.0f} bpm")
    unit = "beats"
else:
    top = a["result"]["mean_proba"].argmax()
    c1.metric("Windows of 10 s", len(events))
    c2.metric("Most likely rhythm", f"{inf.PTBXL_CLASSES[top]}")
    unit = "windows"
c3.metric(f"Flagged {unit}", f"{len(flagged)}  ({100 * len(flagged) / len(events):.0f} %)")
if len(flagged):
    counts = flagged["class"].value_counts()
    st.caption("Flagged: " + ", ".join(f"{c} {n}" for c, n in counts.items()))
else:
    st.caption("Nothing was flagged.")


@st.fragment
def viewer():
    overview_box = st.container()
    slider_box = st.container()
    ctrl = st.columns([1, 1, 1.4, 1.4])
    left, right = st.columns([3, 1.4])

    if task == "beats":
        lead_name = ctrl[2].selectbox("Lead", a["lead_names"], key="lead")
    else:
        lead_name = ctrl[2].selectbox("Lead", a["lead_names"], index=1, key="lead")
    lead = a["lead_names"].index(lead_name)
    length = ctrl[3].selectbox(
        "Window", [5, 10, 20, 30], index=1, key="win_len", format_func=lambda s: f"{s} s"
    )
    max_start = max(0.0, duration - length)

    with right:
        classes = list(events["class"].unique())
        flag_classes = [c for c in classes if c != normal]
        label = "Show classes" if task == "beats" else "Show rhythms"
        pick = st.multiselect(label, classes, default=flag_classes, key=f"cls_{uid}")
        shown = events[events["class"].isin(pick)].reset_index(drop=True)
        st.caption(f"{len(shown)} in the list. Click a row to jump to it.")

        if task == "beats":
            table = pd.DataFrame(
                {
                    "time": shown["time"],
                    "class": shown["class"],
                    "conf.": shown["confidence"].round(2),
                    "RR before (s)": shown["rr_before"].round(2),
                }
            )
        else:
            table = pd.DataFrame(
                {
                    "time": shown["time"],
                    "rhythm": shown["class"],
                    "conf.": shown["confidence"].round(2),
                }
            )
        table_key = f"table_{uid}_{'-'.join(sorted(pick))}"
        sel = st.dataframe(
            table,
            hide_index=True,
            height=430,
            use_container_width=True,
            key=table_key,
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = sel.selection.rows
        if rows:
            t = float(shown["time_s"].iloc[rows[0]])
            if st.session_state.get("last_jump") != (table_key, t):
                st.session_state["last_jump"] = (table_key, t)
                centre = t if task == "beats" else t + inf.RHYTHM_WINDOW / fs / 2
                st.session_state["pos"] = float(min(max(centre - length / 2, 0.0), max_start))
        if len(flagged):
            st.download_button(
                "Download flagged list (csv)",
                flagged.drop(columns="sample").to_csv(index=False),
                file_name=f"{data['source']}_flagged.csv",
                mime="text/csv",
            )

    times = shown["time_s"].to_numpy()
    if task == "rhythm":
        times = times + inf.RHYTHM_WINDOW / fs / 2

    def go(direction):
        win = st.session_state["win_len"]
        centre = st.session_state["pos"] + win / 2
        if direction > 0:
            later = times[times > centre + 0.05]
            t = later[0] if len(later) else None
        else:
            earlier = times[times < centre - 0.05]
            t = earlier[-1] if len(earlier) else None
        if t is not None:
            st.session_state["pos"] = float(min(max(t - win / 2, 0.0), max(0.0, duration - win)))

    centre_now = st.session_state.get("pos", 0.0) + length / 2
    ctrl[0].button(
        "Previous flagged",
        on_click=go,
        args=(-1,),
        use_container_width=True,
        disabled=not (times < centre_now - 0.05).any(),
    )
    ctrl[1].button(
        "Next flagged",
        on_click=go,
        args=(1,),
        use_container_width=True,
        disabled=not (times > centre_now + 0.05).any(),
    )

    st.session_state["pos"] = float(min(max(st.session_state.get("pos", 0.0), 0.0), max_start))
    if max_start > 0:
        with slider_box:
            pos = st.slider("Position (s)", 0.0, float(max_start), step=0.5, key="pos")
    else:
        pos = 0.0

    with overview_box:
        st.plotly_chart(
            vw.overview_figure(a, lead, events, task, pos, length),
            use_container_width=True,
            config={"staticPlot": True},
            key=f"overview_{uid}",
        )
    with left:
        if task == "rhythm" and st.checkbox("Show all 12 leads (filtered)", key="all_leads"):
            fig = vw.all_leads_figure(a, a["lead_names"], pos, length)
        else:
            fig = vw.detail_figure(a, lead, pos, length, events, task)
        st.plotly_chart(fig, use_container_width=True, key=f"detail_{uid}")


viewer()

with st.expander("About the model"):
    loaded = get_model(model_name)
    st.write(
        f"Model {model_name}, best epoch {loaded.epoch}, "
        f"{'uses RR timing features' if loaded.n_extra else 'signal only'}. "
        f"Analysis took {a['total_s']:.1f} s, of which the model itself {a['model_ms']:.0f} ms."
    )
    res = test_results(model_name)
    if res:
        s = res["summary"]
        st.write(
            f"Test split ({res['n_samples']} samples): accuracy {s['accuracy']:.3f}, "
            f"macro F1 {s['macro_f1']:.3f}, macro AUPRC {s['macro_auprc']:.3f}."
        )
        df = pd.DataFrame(res["per_class"]).T[
            ["support", "precision", "recall", "specificity", "f1", "auprc"]
        ]
        df["support"] = df["support"].astype(int)
        st.dataframe(df.round(3), use_container_width=True)
    if task == "beats":
        st.write(
            "Every beat is classified on its own from two leads. The beats are found by an "
            "automatic R-peak detector, not by a cardiologist. The rare classes (S, F, Q) come "
            "from very few patients in the training data. F is practically not recognised, "
            "so flagged F beats should not be trusted."
        )
    else:
        st.write(
            "Every 10 s window is put into one of four rhythms. Anything else is forced "
            "into one of these four. Sinus bradycardia is the weakest class."
        )
    st.write(
        "The filtering and normalisation are the same as in training. This is a prototype "
        "from a university project and not a medical device."
    )