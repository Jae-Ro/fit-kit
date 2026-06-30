import torch


def get_device(preference: str = "auto") -> torch.device:
    """Utility function to get correcto pytorch device for host system

    :param preference: preferred device string (e.g., "cpu", "cuda:0"), defaults to "auto"
    :return: correct torch.device
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
