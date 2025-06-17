import torch

from src.models.unet import UNet

def load_model(weights_path: str, device: str = "cpu"):
    """Загрузка модели с весами"""
    model = UNet(n_classes=1)
    model.load_state_dict(torch.load(
        (weights_path),
        map_location=torch.device(device)
    ))
    model.eval()
    return model