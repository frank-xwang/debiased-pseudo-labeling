from torch import nn


def get_norm(norm):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "BN1d": nn.BatchNorm1d,
            "SyncBN": nn.SyncBatchNorm,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "IN": lambda channels: nn.InstanceNorm2d(channels, affine=True),
            "None": None,
        }[norm]
    return norm
