import matplotlib.pyplot as plt
import numpy as np

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
    elif layout == "4x1":
        return (7.0, 2.2)    # four panels in a row (shorter height)
    elif layout == "1x4":
        return (7.0, 8.8)    # four stacked panels (~2.2" each)
    else:
        raise ValueError("layout must be 'single', 'double', or '2x2'")
    

def merge_and_sort_by_x(*xy_pairs):
    """
    Accepts arbitrary (x, y) pairs, merges them, sorts by x,
    and returns:
        - x_sorted: merged and sorted x values
        - y_sorted: corresponding y values
        - index_maps: list of arrays, each mapping original indices to sorted positions
    """
    # Step 1: Concatenate all x and y
    x_all = np.concatenate([x for x, _ in xy_pairs])
    y_all = np.concatenate([y for _, y in xy_pairs])

    # Step 2: Sort by x
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_sorted = y_all[sort_idx]

    # Step 3: Build index maps for each input set
    index_maps = []
    offset = 0
    for x, _ in xy_pairs:
        n = len(x)
        original_indices = np.arange(offset, offset + n)
        sorted_positions = np.searchsorted(sort_idx, original_indices)
        index_maps.append(sorted_positions)
        offset += n

    return x_sorted, y_sorted, index_maps


def sort_and_track_indices(*arrays):
    """
    Accepts multiple 1D arrays (x0, x1, ..., xn), merges and sorts them,
    and returns:
        - x_sorted: the sorted merged array
        - index_maps: list of arrays, each giving the sorted index of each input array's elements
        - sort_idx: permutation indices that sort the merged array
    """
    # Step 1: Concatenate all arrays
    x_all = np.concatenate(arrays).squeeze()

    # Step 2: Sort and get permutation indices
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]

    # Step 3: Track origin of each element
    source_labels = []
    for i, arr in enumerate(arrays):
        source_labels.extend([i] * len(arr))
    source_labels = np.array(source_labels)

    # Step 4: Apply sort to labels
    sorted_labels = source_labels[sort_idx]

    # Step 5: Build index maps
    index_maps = []
    for i in range(len(arrays)):
        index_maps.append(np.where(sorted_labels == i)[0])

    return x_sorted, index_maps, sort_idx




def comb_metric_dict(train_dict, val_dict, test_dict, sort_idx):
    new_dict = {}
    for model in train_dict.keys():
        if len(train_dict[model]) != 0:
            metrics_train = train_dict[model]
            metrics_val = val_dict[model]
            metrics_test = test_dict[model]
            temp_dict = {}
            for key in metrics_train.keys():
                metrics = np.concatenate((metrics_train[key], metrics_val[key], metrics_test[key]))
                metrics = metrics[sort_idx]
                temp_dict[key] = metrics
            new_dict[model] = temp_dict
    return new_dict

