import torch
import os
import numpy as np
import json
from .general_saver import save_analysis_arrays

def single_task_saver(metric_outputs, model, trainer, summary, x_train, y_train, x_test, y_test, save_dir):
    # Create directory
    os.makedirs(os.path.join(save_dir, "analysis"), exist_ok=True)

    # Save metrics
    save_analysis_arrays(metric_outputs, os.path.join(save_dir, "analysis"))
    np.savez(os.path.join(save_dir, "analysis", "loss_curve_data.npz"),
                        losses=trainer.losses,
                        mses=trainer.mses,
                        kls=trainer.kls,
                        betas=trainer.betas)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save test and training data
    np.savez(os.path.join(save_dir, "analysis", "data.npz"),
             x_train=x_train.cpu().numpy(),
             y_train=y_train.cpu().numpy(),
             x_test=x_test.cpu().numpy(),
             y_test=y_test.cpu().numpy())

    # Save summary
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)



