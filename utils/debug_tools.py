def debug_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, grad is None? {param.grad is None}")

def get_param_and_grad_dict(model, detach=True, clone=True):
    """
    Collects parameter values and gradients from a model.

    Args:
        model (nn.Module): The PyTorch model.
        detach (bool): Detach from graph (useful for logging/debugging).
        clone (bool): Clone tensors to prevent mutation after optimizer.step().

    Returns:
        dict: {
            'param_name': {'value': tensor, 'grad': tensor or None}, ...
        }
    """
    param_info = {}
    for name, param in model.named_parameters():
        value = param.data
        grad = param.grad

        if detach:
            value = value.detach()
            grad = grad.detach() if grad is not None else None
        if clone:
            value = value.clone()
            grad = grad.clone() if grad is not None else None

        param_info[name] = {
            'value': value.cpu(),
            'grad': grad.cpu() if grad is not None else None
        }

    return param_info


