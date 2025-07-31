import os
import pickle

from utils.general import get_latest_run_dir
from data.toy_functions import generate_meta_task

def load_pickle_plot(pickle_path, show=True):
    """
    Load and optionally display a matplotlib figure saved as a .pkl file.

    Args:
        pickle_path (str): Path to the .pkl file.
        show (bool): Whether to display the plot immediately.

    Returns:
        matplotlib.figure.Figure: The loaded figure object.
    """
    with open(pickle_path, "rb") as f:
        fig = pickle.load(f)

    if show:
        fig.show()

    return fig

def load_toy_task_regression(seed=0):
    x_c, y_c, x_t, y_t, desc = generate_meta_task(seed=seed)

    # Convert to float64 explicitly
    x_c = x_c.double()
    y_c = y_c.double()
    x_t = x_t.double()
    y_t = y_t.double()

    return x_c, y_c, x_t, y_t, {"description": desc}

if __name__ == '__main__':
    # Imports
    from general import get_latest_run_dir

    # Model and metrics
    model_type = 'IC_FDNet'
    metric = 'residual_scatter'

    # Get latest run directory
    run_dir = get_latest_run_dir(model_type)

    if run_dir:
        plot_path = os.path.join(run_dir, "plots", f"{metric}.pkl")
        print("Trying:", plot_path)
        if os.path.exists(plot_path):
            fig = load_pickle_plot(plot_path)
        else:
            print("File not found at path.")
    else:
        print("No run directories found for this model.")
