import pytest
import numpy as np
from plotly.graph_objects import Figure
from diveai.plotting import PlotBuilder

def test_2d_line_plot():
    plot = PlotBuilder(x_label="X Axis", y_label="Y Axis", title="2D Line Plot")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot.add_plot(x, y, plot_type="line", color="red", label="Sine Wave")
    
    assert isinstance(plot.fig, Figure)
    assert len(plot.fig.data) == 1
    assert plot.is_3d is False

def test_2d_scatter_plot():
    plot = PlotBuilder()
    x = [1, 2, 3, 4]
    y = [10, 15, 7, 12]
    plot.add_plot(x, y, plot_type="scatter", color="green", label="Data Points")
    
    assert len(plot.fig.data) == 1
    assert plot.fig.data[0].mode == "markers"

def test_2d_bar_plot():
    plot = PlotBuilder()
    x = ["A", "B", "C"]
    y = [5, 10, 15]
    plot.add_plot(x, y, plot_type="bar", color="blue", label="Bar Data")
    
    assert len(plot.fig.data) == 1
    assert plot.fig.data[0].type == "bar"

def test_3d_plot():
    plot = PlotBuilder(title="3D Line Plot")
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    z = np.cos(x)
    plot.add_plot(x, y, z, plot_type="line", color="purple", label="3D Wave")
    
    assert plot.is_3d is True
    assert len(plot.fig.data) == 1
    assert plot.fig.data[0].type == "scatter3d"

def test_3d_mixed_plot_error():
    plot = PlotBuilder()
    plot.add_plot([1, 2, 3], [4, 5, 6], z=[7, 8, 9], plot_type="scatter")
    
    with pytest.raises(ValueError, match="All plots must be 3D since the first plot was 3D."):
        plot.add_plot([1, 2, 3], [4, 5, 6], plot_type="scatter")

def test_3d_bar_plot_error():
    plot = PlotBuilder()
    with pytest.raises(ValueError, match="3D bar plots are not supported in Plotly."):
        plot.add_plot([1, 2, 3], [4, 5, 6], z=[7, 8, 9], plot_type="bar")

def test_update_plot():
    plot = PlotBuilder()
    x = [0, 1, 2]
    y = [3, 4, 5]
    plot.add_plot(x, y, plot_type="line", color="black", label="Initial Plot")
    plot.update_plot([0, 1, 2], [6, 7, 8], trace_index=0)
    
    assert list(plot.fig.data[0].y) == [6, 7, 8]

def test_update_3d_plot():
    plot = PlotBuilder()
    x = [0, 1, 2]
    y = [3, 4, 5]
    z = [6, 7, 8]
    plot.add_plot(x, y, z, plot_type="scatter", color="orange", label="3D Scatter")
    
    with pytest.raises(ValueError, match="z-coordinates must be provided for 3D plots."):
        plot.update_plot([0, 1, 2], [9, 10, 11], trace_index=0)
    
    plot.update_plot([0, 1, 2], [9, 10, 11], [12, 13, 14], trace_index=0)
    assert list(plot.fig.data[0].z) == [12, 13, 14]

def test_set_labels():
    plot = PlotBuilder()
    plot.set_labels(x_label="Time", y_label="Value", title="Updated Title")
    
    assert plot.fig.layout.title.text == "Updated Title"
    assert plot.fig.layout.xaxis.title.text == "Time"
    assert plot.fig.layout.yaxis.title.text == "Value"
