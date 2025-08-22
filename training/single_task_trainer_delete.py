import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import numpy as np
import random

class SingleTaskTrainer:
    def __init__(self, model, optimizer=None, lr=1e-3, device=None):
        self.model = model.double()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.kl_models = {'IC_FDNet', 'LP_FDNet', 'BayesNet', 'GaussHyperNet'}
        self.losses, self.mses, self.kls, self.betas = [], [], [], []

        # Handle optimizer(s)
        if model.__class__.__name__ == 'DeepEnsembleNet':
            self.optimizers = [
                torch.optim.Adam(submodel.parameters(), lr=lr)
                for submodel in model.models
            ]
            self.optimizer = None
        else:
            self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)

        # Early stop state
        self._best_val_mse = np.inf
        self._best_state = None
        self._epochs_no_improve = 0

    # ---------- utils ----------
    def _model_has_args(self, *names):
        if not hasattr(self.model, 'forward'):
            return False
        argnames = self.model.forward.__code__.co_varnames
        return all(name in argnames for name in names)

    def _model_forward(self, x, return_kl=True, sample=True):
        # Fix: check each arg separately (not ('return_kl' and 'sample' in ...))
        if self._model_has_args('return_kl', 'sample'):
            return self.model(x, return_kl=return_kl, sample=sample)
        else:
            return self.model(x), torch.tensor(0.0, device=x.device, dtype=torch.double)

    @torch.no_grad()
    def _val_pass(self, x_val, y_val, beta=0.0, MC=1):
        """
        Returns: (val_loss, val_mse, val_kl) averaged over val set.
        Uses MC for stochastic models; ensemble averages members.
        """
        if x_val is None or y_val is None:
            return None

        self.model.eval()
        x_val = x_val.double().to(self.device)
        y_val = y_val.double().to(self.device)
        N = x_val.shape[0]

        if self.model.__class__.__name__ in self.kl_models and MC > 1:
            mse_total, kl_total = 0.0, 0.0
            for _ in range(MC):
                y_pred, kl = self._model_forward(x_val, return_kl=True, sample=True)
                mse = F.mse_loss(y_pred, y_val)
                mse_total += mse.item()
                kl_total  += kl.item()
            mse = mse_total / MC
            kl  = kl_total / MC
        elif self.model.__class__.__name__ == 'DeepEnsembleNet':
            # Average MSE across members (each member predicts deterministically)
            mses = []
            for submodel in self.model.models:
                y_pred = submodel(x_val)
                mses.append(F.mse_loss(y_pred, y_val).item())
            mse = float(np.mean(mses))
            kl = 0.0
        else:
            y_pred, kl_t = self._model_forward(x_val, return_kl=True, sample=True)
            mse = F.mse_loss(y_pred, y_val).item()
            kl = float(kl_t.item())

        val_loss = mse + float(beta) * kl
        return val_loss, mse, kl

    def _maybe_checkpoint(self, path, extra=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "losses": self.losses,
            "mses": self.mses,
            "kls": self.kls,
            "betas": self.betas,
            "extra": extra or {},
        }
        # For ensembles, saving the container state_dict captures members too
        torch.save(payload, path)

    # ---------- training ----------
    def train(
        self,
        x_train, y_train,
        epochs=1000,
        beta_param_dict=None,
        x_val=None, y_val=None,
        print_every=100,
        batch_size=10,
        grad_clip=True,
        MC=1,
        # Early stop / checkpointing
        patience=50,
        min_delta=0.0,
        restore_best_weights=True,
        best_ckpt_path=None,          # e.g. "checkpoints/best.pt"
        save_every=None,              # e.g. 100 to also save periodic checkpoints
        periodic_ckpt_dir=None        # e.g. "checkpoints/epochs"
    ):
        x_train, y_train = x_train.double().to(self.device), y_train.double().to(self.device)
        N_train = x_train.shape[0]
        if N_train == 0:
            raise ValueError("Empty context set provided.")

        # Default beta schedule
        if beta_param_dict is None:
            beta_param_dict = {
                "beta_scheduler": "linear",
                "warmup_epochs": round(epochs / 2),
                "beta_max": 1.0
            }

        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Reset early stop state for this run
        self._best_val_mse = np.inf
        self._epochs_no_improve = 0
        self._best_state = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_kl = 0.0
            total_mse = 0.0
            beta = self._compute_beta(epoch, beta_param_dict)

            for x_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                if self.model.__class__.__name__ in self.kl_models:
                    mse_total, kl_total = None, None
                    for _ in range(MC):
                        y_pred, kl = self._model_forward(x_batch, return_kl=True, sample=True)
                        mse = F.mse_loss(y_pred, y_batch)
                        mse_total = mse if mse_total is None else mse_total + mse
                        kl_total  = kl  if kl_total  is None else kl_total  + kl
                    mse = mse_total / MC
                    kl  = kl_total  / MC
                    loss = mse + beta * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    if grad_clip:
                        clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                elif self.model.__class__.__name__ == 'DeepEnsembleNet':
                    mse_accum = 0.0
                    for submodel, subopt in zip(self.model.models, self.optimizers):
                        subopt.zero_grad()
                        y_pred_i = submodel(x_batch)
                        sub_loss = F.mse_loss(y_pred_i, y_batch)
                        sub_loss.backward()
                        if grad_clip:
                            clip_grad_norm_(submodel.parameters(), 1.0)
                        subopt.step()
                        mse_accum += sub_loss.item()

                    mse = torch.tensor(mse_accum / len(self.model.models), device=self.device, dtype=torch.double)
                    kl  = torch.tensor(0.0, device=self.device, dtype=torch.double)
                    loss = mse  # no KL for ensemble members

                else:
                    y_pred, kl = self._model_forward(x_batch, return_kl=True, sample=True)
                    mse = F.mse_loss(y_pred, y_batch)
                    loss = mse + beta * kl

                    self.optimizer.zero_grad()
                    loss.backward()
                    if grad_clip:
                        clip_grad_norm_((self.model.parameters()), 1.0)
                    self.optimizer.step()

                # Accumulate stats (scale by batch size for epoch mean)
                bs = x_batch.shape[0]
                total_loss += loss.item() * bs
                total_mse  += mse.item()  * bs
                total_kl   += kl.item()   * bs

            # Normalize
            total_loss /= N_train
            total_mse  /= N_train
            total_kl   /= N_train

            self.losses.append(total_loss)
            self.mses.append(total_mse)
            self.kls.append(total_kl)
            self.betas.append(beta)

            # Periodic checkpoint (optional)
            if save_every is not None and periodic_ckpt_dir is not None and (epoch % save_every == 0):
                path = os.path.join(periodic_ckpt_dir, f"epoch_{epoch}.pt")
                self._maybe_checkpoint(path, extra={"epoch": epoch})

            # Validation + early stop
            if x_val is not None and y_val is not None:
                val_stats = self._val_pass(x_val, y_val, beta=beta, MC=max(MC, 10))
                if val_stats is not None:
                    val_loss, val_mse, val_kl = val_stats

                    improved = (self._best_val_mse - val_mse) > float(min_delta)
                    if improved:
                        self._best_val_mse = val_mse
                        self._epochs_no_improve = 0
                        # Save the absolute best state
                        self._best_state = {
                            "model_state": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
                            "epoch": epoch,
                            "val_mse": val_mse
                        }
                        if best_ckpt_path is not None:
                            self._maybe_checkpoint(best_ckpt_path, extra={"epoch": epoch, "val_mse": val_mse})
                    else:
                        self._epochs_no_improve += 1

                    if epoch % print_every == 0:
                        print(f"[Epoch {epoch}] TrainLoss {total_loss:.4f} | TrainMSE {total_mse:.4f} | KL {total_kl:.4f} | β {beta:.3f} || ValMSE {val_mse:.4f}")

                    if self._epochs_no_improve >= patience:
                        if restore_best_weights and self._best_state is not None:
                            self.model.load_state_dict(self._best_state["model_state"])
                        return  # stop training early
            else:
                if epoch % print_every == 0:
                    print(f"[Epoch {epoch}] Loss {total_loss:.4f} | MSE {total_mse:.4f} | KL {total_kl:.4f} | β {beta:.3f}")

        # Finished all epochs; restore best if requested and available
        if x_val is not None and y_val is not None and restore_best_weights and self._best_state is not None:
            self.model.load_state_dict(self._best_state["model_state"])

    # kept same API
    def early_stop(self, x_val, y_val):
        # No-op: early stopping is integrated into train() above.
        return False

    @torch.no_grad()
    def evaluate(self, x, num_samples=30, sample=True):
        self.model.eval()
        x = x.double().to(self.device)
        N = x.shape[0]
        preds = []

        if hasattr(self.model, 'forward') and ('return_kl' and 'sample' in self.model.forward.__code__.co_varnames):
            y_pred = [self._model_forward(x=x, return_kl=False, sample=sample).squeeze(0).detach().cpu().numpy() for _ in range(num_samples)]
        elif self.model.__class__.__name__ == 'DeepEnsembleNet':
            y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
        else:
            y_pred = self._model_forward(x=x)[0].squeeze(0).detach().cpu().numpy()
            y_pred = [y_pred for _ in range(num_samples)]

        preds = np.stack(y_pred)  # shape: (num_samples, batch_size, output_dim)

        return preds

    def _compute_beta(self, epoch, beta_param_dict):
        beta_scheduler = beta_param_dict["beta_scheduler"]
        beta_max = float(beta_param_dict["beta_max"])
        warmup_epochs = int(beta_param_dict["warmup_epochs"])

        if beta_scheduler == "constant":
            return beta_max

        progress = min(epoch / max(warmup_epochs, 1), 1.0)

        if beta_scheduler == "linear":
            return beta_max * progress
        elif beta_scheduler == "cosine":
            return beta_max * (1 - np.cos(np.pi * progress)) / 2
        elif beta_scheduler == "sigmoid":
            slope = 12
            midpoint = 0.5
            sigmoid_val = 1 / (1 + np.exp(-slope * (progress - midpoint)))
            sigmoid_norm = sigmoid_val / (1 / (1 + np.exp(-slope * (1 - midpoint))))
            return beta_max * sigmoid_norm
        elif beta_scheduler == "zero":
            return 0.0
        else:
            raise ValueError(f"Unsupported beta scheduler: {beta_scheduler}")



if __name__=='__main__':
    from models.fdnet import LP_FDNet, IC_FDNet
    from models.hypernet import HyperNet
    from models.bayesnet import BayesNet
    from models.gausshypernet import GaussianHyperNet
    from models.mlpnet import MLPNet
    from models.deepensemblenet import DeepEnsembleNet
    from data.toy_functions import generate_meta_task
    
    # Parameters
    input_dim = 1               
    hidden_dim = 32
    epochs = 500
    print_every = 20
    sample = True
    seed = 10
    model_type = 'DeepEnsembleNet'

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Train and test data
    x_train, y_train, x_val, y_val, x_test, y_test, desc = generate_meta_task(n_train=10, n_val=5, n_test=100, seed=seed)
    
    # Create model
    if model_type == 'LP_FDNet':
        model = LP_FDNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'IC_FDNet':
        model = IC_FDNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'HyperNet':
        model = HyperNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64)
    elif model_type == 'BayesNet':
        model = BayesNet(input_dim, hidden_dim, input_dim, prior_std=1.0)
    elif model_type == 'GaussHyperNet':
        model = GaussianHyperNet(input_dim, hidden_dim, input_dim, hyper_hidden_dim=64, latent_dim=10, prior_std=1.0)
    elif model_type == 'MLPNet':
        model = MLPNet(input_dim, hidden_dim, input_dim, dropout_rate=0.1)
    elif model_type == 'DeepEnsembleNet':
        model = DeepEnsembleNet(
            network_class=MLPNet,
            num_models=5,
            seed_list=[0, 1, 2, 3, 4],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            dropout_rate=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print('Running', type(model).__name__, '======================================================================')
    
    # Create training class instance
    trainer = SingleTaskTrainer(model)
    # Train
    trainer.train(
    x_train, y_train,
    epochs=1000,
    beta_param_dict={"beta_scheduler":"linear","warmup_epochs":500,"beta_max":1.0},
    x_val=x_val, y_val=y_val,
    print_every=50,
    batch_size=32,
    MC=4,                          # MC for stochastic models (train & val)
    patience=100,                  # early stop patience
    min_delta=0.0,                 # require this much improvement
    restore_best_weights=True,     # load best at the end
    best_ckpt_path="checkpoints/best.pt",
    save_every=200,
    periodic_ckpt_dir="checkpoints/periodic"
)
    # Evaluate
    preds = trainer.evaluate(x=x_test)



