import cv2 
import torch
import numpy as np

from src.models.unet import ResNet34_UNet
from src.data.transforms import val_transform


def predict(image_path, model_path, device='cuda'):
    model = ResNet34_UNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    augmented = val_transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask = model(image_tensor)
        mask = mask.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (original_size[1], original_size[0]))
    
    return mask