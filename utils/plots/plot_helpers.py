import matplotlib.pyplot as plt

def label_subplots(axes, labels=None, x_offset=-0.1, y_offset=1.05, fontsize=8):
    """
    Automatically label subplots with (a), (b), (c), ...
    
    Args:
        axes: single Axes, 1D array of Axes, or 2D array of Axes
        labels: list of custom labels (default: (a), (b), ...)
        x_offset, y_offset: relative text position in Axes coords
        fontsize: label font size
    """
    import numpy as np
    
    # Flatten axes for easy iteration
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = np.ravel(axes)

    # Default labels: (a), (b), (c), ...
    if labels is None:
        labels = [f"({chr(97+i)})" for i in range(len(axes))]

    for ax, label in zip(axes, labels):
        ax.text(x_offset, y_offset, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="bottom", ha="right")


def iclr_figsize(layout="single"):
    """
    Return ICLR-ready figure sizes.
    layout: 'single', 'double', '2x2'
    """
    if layout == "single":
        return (3.5, 2.5)   # single column
    elif layout == "double":
        return (7.0, 2.8)   # two panels side by side
    elif layout == "stacked":
        return (7.0, 4.2)   # two panel stacked
    elif layout == "2x2":
        return (7.0, 4.2)   # 2x2 panel figure
    else:
        raise ValueError("layout must be 'single', 'double', or '2x2'")