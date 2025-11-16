import torch
import os
import numpy as np
import json
from .general_saver import save_analysis_arrays

def base_experiment_saver(model,
                          trainer,
                          metrics_test,
                          summary_dict,
                          x_train,
                          y_train,
                          x_test,
                          y_test,
                          save_path,
                          x_val=None,
                          y_val=None,
                          metrics_train=None,
                          metrics_val=None,
                          info=None):
    """
    Save all outputs of a base experiment to disk in a structured format.

    Args:
        model: Trained model (nn.Module)
        trainer: Trainer object with .losses, .mses, .kls, .betas
        metric_outputs: Dict of computed metrics
        summary_dict: Dict with additional run info (model type, seed, etc.)
        x_train, y_train, x_test, y_test: Tensors
        save_dir: Root directory to save to
    """
    os.makedirs(os.path.join(save_path, "analysis"), exist_ok=True)

    # === Save model state ===
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))

    # === Save loss curves ===
    def safe_tensor_list(tensor_list):
        return np.array([
            float(t.cpu()) if isinstance(t, torch.Tensor) else float(t)
            for t in tensor_list
        ])

    np.savez(os.path.join(save_path, "analysis", "loss_curve_data.npz"),
             losses=safe_tensor_list(trainer.losses),
             mses=safe_tensor_list(trainer.mses),
             kls=safe_tensor_list(trainer.kls),
             betas=np.array(trainer.betas))

    # === Save training and test data ===
    region = np.array(info.get('region', None))
    region_interp = np.array(info.get('region_interp', None))

    if x_val != None and y_val != None:
        np.savez(os.path.join(save_path, "analysis", "data.npz"),
                x_train=x_train.cpu().numpy(),
                y_train=y_train.cpu().numpy(),
                x_val=x_val.cpu().numpy(),
                y_val=y_val.cpu().numpy(),
                x_test=x_test.cpu().numpy(),
                y_test=y_test.cpu().numpy(),
                region=region,
                region_interp=region_interp)
    else:
        np.savez(os.path.join(save_path, "analysis", "data.npz"),
                x_train=x_train.cpu().numpy(),
                y_train=y_train.cpu().numpy(),
                x_test=x_test.cpu().numpy(),
                y_test=y_test.cpu().numpy(),
                region=region,
                region_interp=region_interp)
        
    # === Save metrics ===
    save_analysis_arrays(metrics_test, os.path.join(save_path, "analysis"))
    if metrics_train != None:
        save_analysis_arrays(metrics_train, os.path.join(save_path, "train_metrics"))
    if metrics_val != None:
        save_analysis_arrays(metrics_val, os.path.join(save_path, "val_metrics"))
    # with open(os.path.join(save_dir, "metrics.json"), "w") as f:
    #     json.dump(safe_metrics, f, indent=4)

    # === Save summary ===
    with open(os.path.join(save_path, "summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=4)

