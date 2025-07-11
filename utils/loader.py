import os
import pickle
import matplotlib.pyplot as plt

def get_latest_run_dir(model_name, base_dir="results"):
    subdirs = [
        os.path.join(base_dir, model_name, d)
        for d in os.listdir(os.path.join(base_dir, model_name))
        if os.path.isdir(os.path.join(base_dir, model_name, d))
    ]
    return max(subdirs, key=os.path.getmtime) if subdirs else None

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


if __name__ == '__main__':
    model = 'IC_FDNet'
    metric = 'residual_scatter'

    run_dir = get_latest_run_dir(model)
    if run_dir:
        plot_path = os.path.join(run_dir, "plots", f"{metric}.pkl")
        print("Trying:", plot_path)
        if os.path.exists(plot_path):
            fig = load_pickle_plot(plot_path)
        else:
            print("File not found at path.")
    else:
        print("No run directories found for this model.")
        
    print('end')
