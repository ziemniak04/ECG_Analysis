"""Check plot content, intervals, and safe output creation."""
import matplotlib.pyplot as plt
import pytest
from src.data_loader import load_recording
from src.visualization import plot_recording, plot_comparison, save_figure


def test_all_channels_and_annotations(tmp_path):
    record = load_recording()
    fig = plot_recording(record)
    try:
        assert len(fig.axes) == 2
        expected = sum(record.annotation_positions < 3600)
        for ax in fig.axes:
            assert len(ax.texts) == expected
            assert len(ax.lines[0].get_xdata()) == 3600
            assert "+" in [text.get_text() for text in ax.texts]
        path = save_figure(fig, tmp_path / "ecg.png")
        assert path.read_bytes().startswith(b"\x89PNG")
        with pytest.raises(FileExistsError):
            save_figure(fig, path)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("start,duration", [(-1,10),(0,0),(0,float("nan")),(2000,10)])
def test_invalid_interval(start, duration):
    with pytest.raises(ValueError):
        plot_recording(load_recording(), start, duration)


def test_comparison():
    fig = plot_comparison([load_recording("100"), load_recording("101"), load_recording("200")])
    try:
        assert len(fig.axes) == 3
        assert all(len(ax.lines[0].get_xdata()) == 3600 for ax in fig.axes)
    finally:
        plt.close(fig)
