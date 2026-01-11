import torch

def deg2radian(deg):
    """
    Converts degrees to radians

    Parameter
    _________
    deg: Union[torch.tensor, float]
        Degree value.
    """
    return torch.tensor(torch.pi*(deg/180))