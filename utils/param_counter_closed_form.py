
"""
param_closed_form.py

Closed-form parameter counts for your one-hidden-layer architectures.
Each function returns an integer count of trainable parameters.

Conventions:
- d_in = input_dim, H = hidden_dim, d_out = output_dim
- Hh = hyper_hidden_dim, L = latent_dim, E = num_models
"""

from __future__ import annotations

def _T(n_in: int, n_out: int) -> int:
    """Weights+biases to emit for Linear(n_in -> n_out)."""
    return n_out * n_in + n_out

def mlp_params(d_in: int, H: int, d_out: int) -> int:
    """MLP (Linear d_in->H, Linear H->d_out)."""
    return H * d_in + H + d_out * H + d_out

def mlp_dropout_params(d_in: int, H: int, d_out: int) -> int:
    """Same topology as MLP; dropout adds no parameters."""
    return mlp_params(d_in, H, d_out)

def bayesnet_params(d_in: int, H: int, d_out: int) -> int:
    """Bayesian MLP with (mu, rho) per weight/bias -> 2x deterministic."""
    return 2 * mlp_params(d_in, H, d_out)

def hypernet_params(d_in: int, H: int, d_out: int, Hh: int) -> int:
    """
    HyperNet with two hyperlayers:
      L1: cond=d_in, emit T(d_in, H)
      L2: cond=H,   emit T(H, d_out)
    Each hyperlayer is Linear(cond->Hh) + Linear(Hh->T) (with bias).
    """
    T1 = _T(d_in, H)
    T2 = _T(H, d_out)
    L1 = Hh * d_in + Hh + T1 * Hh + T1
    L2 = Hh * H    + Hh + T2 * Hh + T2
    return L1 + L2

def ic_fdnet_params(d_in: int, H: int, d_out: int, Hh: int) -> int:
    """
    IC_FDNet: Gaussian W,b -> emit (mu, log_sigma) => 2*T; both layers condition on d_in.
    """
    T1 = 2 * _T(d_in, H)
    T2 = 2 * _T(H, d_out)
    L1 = Hh * d_in + Hh + T1 * Hh + T1
    L2 = Hh * d_in + Hh + T2 * Hh + T2
    return L1 + L2

def lp_fdnet_params(d_in: int, H: int, d_out: int, Hh: int) -> int:
    """
    LP_FDNet: Gaussian W,b -> 2*T; layer 1 cond=d_in, layer 2 cond=H.
    """
    T1 = 2 * _T(d_in, H)
    T2 = 2 * _T(H, d_out)
    L1 = Hh * d_in + Hh + T1 * Hh + T1
    L2 = Hh * H    + Hh + T2 * Hh + T2
    return L1 + L2

def gausshypernet_params(d_in: int, H: int, d_out: int, Hh: int, L: int) -> int:
    """
    GaussHyperNet: each layer conditions on latent z (dim L);
    hyperlayer: Linear(L->Hh) + Linear(Hh->2*T) + learnable z (length L) per layer.
    """
    T1 = 2 * _T(d_in, H)
    T2 = 2 * _T(H, d_out)
    L1 = Hh * L + Hh + T1 * Hh + T1 + L
    L2 = Hh * L + Hh + T2 * Hh + T2 + L
    return L1 + L2

def deep_ensemble_mlp_params(d_in: int, H: int, d_out: int, E: int) -> int:
    """Deep ensemble of E MLP members (wrapper overhead ignored)."""
    return E * mlp_params(d_in, H, d_out)

# Convenience: compute all at once
def all_model_params(d_in: int, H: int, d_out: int, Hh: int, L: int, E: int):
    return {
        "MLPNet": mlp_params(d_in, H, d_out),
        "MLPDropoutNet": mlp_dropout_params(d_in, H, d_out),
        "BayesNet": bayesnet_params(d_in, H, d_out),
        "HyperNet": hypernet_params(d_in, H, d_out, Hh),
        "IC_FDNet": ic_fdnet_params(d_in, H, d_out, Hh),
        "LP_FDNet": lp_fdnet_params(d_in, H, d_out, Hh),
        "GaussHyperNet": gausshypernet_params(d_in, H, d_out, Hh, L),
        "DeepEnsemble(MLP base)": deep_ensemble_mlp_params(d_in, H, d_out, E),
    }

if __name__ == "__main__":
    # sanity check with example dims
    H = 32
    Hh = 32
    L = 10
    E = 5
    print(all_model_params(d_in=1, H=H, d_out=1, Hh=Hh, L=L, E=5))

    print(f"mlp count = {mlp_params(d_in=1, H=32, d_out=1)}")

    print(f"bayes count = {bayesnet_params(d_in=1, H=16, d_out=1)}")

    print(f"hyper count = {hypernet_params(d_in=1, H=4, d_out=1, Hh=4)}")

    print(f"ic fdn count = {ic_fdnet_params(d_in=1, H=3, d_out=1, Hh=3)}")

    print(f"lp fdn count = {lp_fdnet_params(d_in=1, H=3, d_out=1, Hh=3)}")

    print(f"gauss hyper count = {gausshypernet_params(d_in=1, H=3, d_out=1, Hh=3, L=1)}")

    print(f"de count = {deep_ensemble_mlp_params(d_in=1, H=3, d_out=1, E=10)}")

    print(f"mlp count = {mlp_params(d_in=1, H=770, d_out=1)}")

    print(f"ic fdn count = {ic_fdnet_params(d_in=1, H=16, d_out=1, Hh=10)}")

    print(f"lp fdn count = {lp_fdnet_params(d_in=1, H=15, d_out=1, Hh=9)}")
