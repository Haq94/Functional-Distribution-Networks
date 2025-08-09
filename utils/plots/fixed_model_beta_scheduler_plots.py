import os
import matplotlib.pyplot as plt
import numpy as np

from utils.saver.general_saver import save_plot

def plot_training_metrics_overlay(metrics_dicts, seeds=None, 
                                  beta_scheduler_types=None, 
                                  beta_max_arr=None, 
                                  warmup_epochs_arr=None, 
                                  save_dir=None):
    seeds = list(metrics_dicts.keys()) if seeds is None else seeds
    for seed in seeds:
        seed_dict = metrics_dicts[seed]
        for model_type in seed_dict.keys():
            model_dict = seed_dict[model_type]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            metric_keys = ["losses_per_epoch", "mse_per_epoch", "kls_per_epoch", "beta_per_epoch"]
            titles = [r"$Loss$ vs $Epoch$", r"$MSE$ vs $Epoch$", r"$KL$ $Divergence$ vs $Epoch$", r"$\beta$ vs $Epoch$"]

            handles = []
            labels = []

            for i, (key, title) in enumerate(zip(metric_keys, titles)):
                ax = axes[i // 2][i % 2]

                for beta_scheduler in sorted(model_dict.keys()):
                    if beta_scheduler_types and beta_scheduler not in beta_scheduler_types:
                        continue
                    beta_dict = model_dict[beta_scheduler]

                    for beta in sorted(beta_dict.keys()):
                        if beta_max_arr is not None and not np.isclose(beta, beta_max_arr):
                            continue
                        warmup_dict = beta_dict[beta]

                        for warmup in sorted(warmup_dict.keys()):
                            if warmup_epochs_arr and warmup not in warmup_epochs_arr:
                                continue

                            run_metrics = warmup_dict[warmup]
                            y = run_metrics.get(key, None)
                            if y is None:
                                continue
                            label = f"{beta_scheduler}, β={beta:.2f}, warmup={warmup}"
                            line, = ax.plot(range(len(y)), y, label=label)
                            if i == 0:  # collect legend handles once
                                handles.append(line)
                                labels.append(label)

                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel(title)
                ax.grid(True)

            # fig.suptitle(f"{model_type} | Seed {seed}", fontsize=14)
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            if save_dir:
                fname = f"{model_type}_seed{seed}_metrics_overlay"
                save_plot(os.path.join(save_dir,f"seed{seed}", model_type), plot_name=fname, fig=fig)
                plt.close()
            else:
                plt.show()


def plot_final_metrics_vs_x_overlay(metrics_dicts, seeds=None,
                                    beta_scheduler_types=None,
                                    beta_max_arr=None,
                                    warmup_epochs_arr=None,
                                    save_dir=None):
    """
    Plot mean, variance, bias, and mse vs x for each model and seed,
    overlaying different beta schedules. Shared legend and interpolation region.
    """
    seeds = list(metrics_dicts.keys()) if seeds is None else seeds
    for seed in seeds:
        seed_dict = metrics_dicts[seed]
        for model_type, model_dict in seed_dict.items():

            metric_keys = ["final_mean", "final_variance", "final_bias", "final_mse"]
            titles = ["$\mu$ vs $x$", "$\sigma^2$ (dB) vs $x$", "$Bias^2$ (dB) vs $x$", "$MSE$ (dB) vs $x$"]
            ylabels = ["$\mu$", "$\sigma^2$ (dB)", "$Bias^2$ (dB)", "$MSE$ (dB)"]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            handles = [[] for _ in range(4)]
            labels = [[] for _ in range(4)]

            x_left, x_right, x_vals, y_vals = None, None, None, None
            truth_plotted = False  # <- Track if Truth is already plotted

            for beta_scheduler in sorted(model_dict.keys()):
                if beta_scheduler_types and beta_scheduler not in beta_scheduler_types:
                    continue
                beta_dict = model_dict[beta_scheduler]

                for beta in sorted(beta_dict.keys()):
                    if beta_max_arr is not None and not any(np.isclose(beta, b) for b in beta_max_arr):
                        continue
                    warmup_dict = beta_dict[beta]

                    for warmup in sorted(warmup_dict.keys()):
                        if warmup_epochs_arr and warmup not in warmup_epochs_arr:
                            continue

                        run_metrics = warmup_dict[warmup]
                        x_vals = np.array(run_metrics["x_values"])
                        y_vals = run_metrics.get("y_values", None)
                        interp_region = run_metrics.get("interp_region", None)

                        if interp_region:
                            x_left, x_right = interp_region[0], interp_region[1]

                        if beta_scheduler == 'constant':
                            label = f"{beta_scheduler}, β={beta:.2f}"
                        else:
                            label = f"{beta_scheduler}, β={beta:.2f}, warmup={warmup}"

                        for i, key in enumerate(metric_keys):
                            raw_y = np.array(run_metrics.get(key))

                            if key in {"final_variance", "final_mse"}:
                                y = 10 * np.log10(np.maximum(raw_y, 1e-12))
                            elif key == "final_bias":
                                y = 20 * np.log10(np.maximum(np.abs(raw_y), 1e-12))
                            else:
                                y = raw_y

                            ax = axes[i // 2][i % 2]
                            line, = ax.plot(x_vals, y, label=label)
                            handles[i].append(line)
                            labels[i].append(label)

                        # Plot truth on mean subplot only once
                        if not truth_plotted and y_vals is not None:
                            ax_mu = axes[0][0]
                            truth_line, = ax_mu.plot(x_vals, y_vals, color="black", linestyle="--", label="Truth")
                            handles[0].append(truth_line)
                            labels[0].append("Truth")
                            truth_plotted = True

            for i, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel(ylabel)
                ax.grid(True)

                if x_left is not None and x_right is not None:
                    ax.axvspan(x_left, x_right, color='lightcoral', alpha=0.3, label="Interpolation Region")
                    ax.axvline(x_left, color='red', linestyle='--')
                    ax.axvline(x_right, color='red', linestyle='--')

            # Global legend (only from mean subplot)
            fig.legend(handles[0], labels[0], loc="upper center", ncol=3, fontsize=8)
            # fig.suptitle(f"{model_type} | Seed {seed} — Final Metrics vs $x$", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            if save_dir:
                fname = f"{model_type}_seed{seed}_final_metrics_vs_x"
                save_plot(os.path.join(save_dir,f"seed{seed}", model_type), plot_name=fname, fig=fig)
                plt.close()
            else:
                plt.show()






# def plot_final_metrics_vs_x_overlay(metrics_dicts, seeds, save_dir=None,
#                                     beta_scheduler_types=None,
#                                     beta_max_arr=None,
#                                     warmup_epochs_arr=None):
#     """
#     Plot mean, variance, bias, and mse vs x for each model and seed,
#     overlaying different beta schedules.
#     """
#     for seed in seeds:
#         seed_dict = metrics_dicts[seed]
#         for model_type, model_dict in seed_dict.items():

#             metric_keys = ["final_mean", "final_variance", "final_bias", "final_mse"]
#             titles = ["$\mu$ vs $x$", "$\sigma^2$ (dB) vs $x$", "$Bias^2$ (dB) vs $x$", "$MSE$ (dB) vs $x$"]
#             ylabels = ["$\mu$", "$\sigma^2$ (dB)", "$Bias^2$ (dB)", "$MSE$ (dB)"]

#             fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#             handles = [[] for _ in range(4)]
#             labels = [[] for _ in range(4)]

#             for scheduler in sorted(model_dict.keys()):
#                 if beta_scheduler_types and scheduler not in beta_scheduler_types:
#                     continue
#                 beta_dict = model_dict[scheduler]

#                 for beta in sorted(beta_dict.keys()):
#                     if beta_max_arr is not None and not any(np.isclose(beta, b) for b in beta_max_arr):
#                         continue
#                     warmup_dict = beta_dict[beta]

#                     for warmup in sorted(warmup_dict.keys()):
#                         if warmup_epochs_arr and warmup not in warmup_epochs_arr:
#                             continue

#                         run_metrics = warmup_dict[warmup]
#                         interp_region = run_metrics['interp_region']
#                         x_left = interp_region[0]
#                         x_right = interp_region[1]
#                         y_vals = run_metrics['y_values']
#                         x_vals = np.array(run_metrics["x_values"])
#                         label = f"{scheduler}, β={beta:.2f}, warmup={warmup}"

#                         for i, key in enumerate(metric_keys):
#                             raw_y = np.array(run_metrics[key])

#                             if key == "final_variance" or key == "final_mse":
#                                 y = 10 * np.log10(np.maximum(raw_y, 1e-12))  # dB scale
#                             elif key == "final_bias":
#                                 y = 20 * np.log10(np.maximum(np.abs(raw_y), 1e-12))  # dB scale
#                             else:
#                                 y = raw_y

#                             ax = axes[i // 2][i % 2]
#                             line, = ax.plot(x_vals, y, label=label)
#                             handles[i].append(line)
#                             labels[i].append(label)

#                             line_truth, = ax.plot(x_vals, y_vals, label='Truth')
#                             handles[i].append(line_truth)
#                             labels[i].append('Truth')

#             for i, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
#                 if x_vals is not None:
#                     ax.axvspan(x_left, x_right, color='lightcoral', alpha=0.3, label="Interpolation Region")
#                     ax.axvline(x_left, color='red', linestyle='--')
#                     ax.axvline(x_right, color='red', linestyle='--')

#                 if i == 0 and x_vals is not None and y_vals is not None:
#                     line_truth, = ax.plot(x_vals, y_vals, color="black", linestyle="--", label="Truth")
#                     handles[i].append(line_truth)
#                     labels[i].append("Truth")

#                 ax.set_title(title)
#                 ax.set_xlabel("x")
#                 ax.set_ylabel(ylabel)
#                 ax.grid(True)
#                 if handles[i]:
#                     ax.legend(handles[i], labels[i], fontsize=7)


#             fig.suptitle(f"{model_type} | Seed {seed} — Final Metrics vs $x$", fontsize=14)
#             plt.tight_layout(rect=[0, 0, 1, 0.95])

#             if save_dir:
#                 fname = f"{model_type}_seed{seed}_final_metrics_vs_x"
#                 save_plot(save_dir, plot_name=fname, fig=fig)
#                 plt.close()
#             else:
#                 plt.show()











