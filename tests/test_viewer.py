import numpy as np

from src.ml import viewer as vw


def test_envelope_keeps_the_peaks():
    y = np.zeros(100000)
    y[54321] = 5.0
    y[12345] = -3.0
    x, e = vw.envelope(y, n_bins=1000)
    assert len(e) == 2000 and e.max() == 5.0 and e.min() == -3.0


def test_envelope_short_signal_unchanged():
    y = np.arange(50.0)
    x, e = vw.envelope(y, n_bins=1000)
    assert np.array_equal(e, y)


def test_fmt_time():
    assert vw.fmt_time(5) == "00:05.0"
    assert vw.fmt_time(65.26) == "01:05.3"
    assert vw.fmt_time(59.96) == "01:00.0"  # no 00:60.0


def fake_result(n=30, fs=360):
    peaks = (np.arange(n) * 300 + 400).astype(np.int64)
    pred = np.zeros(n, dtype=int)
    pred[[3, 10, 11]] = 2  # three V beats
    proba = np.full((n, 5), 0.05)
    proba[np.arange(n), pred] = 0.8
    return {"peaks": peaks, "all_peaks": np.concatenate([[100], peaks]), "pred": pred, "proba": proba}


def test_beat_events_and_rr():
    ev = vw.beat_events(fake_result(), 360)
    assert len(ev) == 30 and (ev["class"] == "V").sum() == 3
    assert abs(ev["rr_before"].iloc[1] - 300 / 360) < 1e-9
    assert abs(ev["rr_before"].iloc[0] - 300 / 360) < 1e-9  # previous peak of the first beat is the extra one


def test_overview_and_detail_figures_build():
    res = fake_result()
    ev = vw.beat_events(res, 360)
    sig = np.random.default_rng(0).normal(size=(12000, 2))
    a = {"fs": 360, "raw": sig, "filtered": sig}
    fig = vw.overview_figure(a, 0, ev, "beats", pos=2.0, length=10)
    assert any(t.name == "V" for t in fig.data)
    fig = vw.detail_figure(a, 0, 0.0, 10, ev, "beats")
    assert {"N", "V"} <= {t.name for t in fig.data if t.name}


def test_rhythm_figures_build():
    starts = np.array([0, 1000, 2000])
    res = {"starts": starts, "pred": np.array([0, 1, 0]), "proba": np.full((3, 4), 0.25)}
    ev = vw.rhythm_events(res, 100)
    a = {"fs": 100, "raw": np.zeros((3000, 12)), "filtered": np.zeros((3000, 12))}
    assert any(t.name == "AFIB" for t in vw.overview_figure(a, 1, ev, "rhythm", 0.0, 10).data)
    assert len(vw.all_leads_figure(a, [f"l{i}" for i in range(12)], 0.0, 10).data) == 12