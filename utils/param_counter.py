
from __future__ import annotations
import importlib
import sys
from pathlib import Path
from typing import Dict, Any
import torch.nn as nn

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

mlpnet = importlib.import_module("models.mlpnet")
mlpdropoutnet = importlib.import_module("models.mlpdropoutnet")
bayesnet = importlib.import_module("models.bayesnet")
hypernet = importlib.import_module("models.hypernet")
gausshypernet = importlib.import_module("models.gausshypernet")
fdnet = importlib.import_module("models.fdnet")
deepensemblenet = importlib.import_module("models.deepensemblenet")

def count_parameters_of(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def build_model(model_name: str, **kwargs) -> nn.Module:
    name = model_name.lower()
    if name == "mlpnet":
        return mlpnet.MLPNet(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs["output_dim"],
            dropout_rate=kwargs.get("dropout_rate", 0.0),
        )
    elif name == "mlpdropoutnet":
        return mlpdropoutnet.MLPDropoutNet(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs.get("output_dim", 1),
            dropout_rate=kwargs.get("dropout_rate", 0.1),
        )
    elif name == "bayesnet":
        return bayesnet.BayesNet(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs["output_dim"],
            prior_std=kwargs.get("prior_std", 1.0),
        )
    elif name == "hypernet":
        return hypernet.HyperNet(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs["output_dim"],
            hyper_hidden_dim=kwargs["hyper_hidden_dim"],
        )
    elif name == "gausshypernet":
        return gausshypernet.GaussHyperNet(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs["output_dim"],
            hyper_hidden_dim=kwargs["hyper_hidden_dim"],
            latent_dim=kwargs.get("latent_dim", 10),
            prior_std=kwargs.get("prior_std", 1.0),
        )
    elif name == "ic_fdnet":
        return fdnet.IC_FDNet(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs["output_dim"],
            hyper_hidden_dim=kwargs["hyper_hidden_dim"],
        )
    elif name == "lp_fdnet":
        return fdnet.LP_FDNet(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs["output_dim"],
            hyper_hidden_dim=kwargs["hyper_hidden_dim"],
        )
    elif name == "deepensemblenet":
        base_model_name = kwargs.get("base_model_name", None)
        num_models = kwargs.get("num_models", 5)
        base_kwargs = kwargs.get("base_kwargs", None)
        if base_model_name is None or base_kwargs is None:
            raise ValueError("deepensemblenet requires 'base_model_name' and 'base_kwargs'.")

        base_cls_lookup = {
            "mlpnet": mlpnet.MLPNet,
            "mlpdropoutnet": mlpdropoutnet.MLPDropoutNet,
            "bayesnet": bayesnet.BayesNet,
            "hypernet": hypernet.HyperNet,
            "gausshypernet": gausshypernet.GaussHyperNet,
            "ic_fdnet": fdnet.IC_FDNet,
            "lp_fdnet": fdnet.LP_FDNet,
        }
        base_cls = base_cls_lookup.get(base_model_name.lower())
        if base_cls is None:
            raise ValueError(f"Unsupported base_model_name '{base_model_name}'.")

        return deepensemblenet.DeepEnsembleNet(
            network_class=base_cls,
            num_models=num_models,
            **base_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}'.")

def count_params(model_name: str, **kwargs) -> int:
    model = build_model(model_name, **kwargs)
    return count_parameters_of(model)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count parameters for a given model and args.")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--output_dim", type=int, default=None)
    parser.add_argument("--hyper_hidden_dim", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--prior_std", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--num_models", type=int, default=5)
    parser.add_argument("--base_model_name", type=str, default=None)
    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if k not in {"model"} and v is not None}
    if args.model.lower() == "deepensemblenet":
        base_keys = dict(kwargs)
        for k in ["num_models", "base_model_name"]:
            base_keys.pop(k, None)
        kwargs["base_kwargs"] = base_keys
    n = count_params(args.model, **kwargs)
    print(n)
