
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Для тренировочных данных
train_transform = A.Compose([
    # A.RandomRotate90(p=0.5),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
    # A.Resize(256, 256),
    A.Normalize(
        mean=[0.3456, 0.3881, 0.3476],  # Inria-специфичные значения
        std=[0.2037, 0.1886, 0.1815], 
        max_pixel_value=255.0
    ),
    ToTensorV2(),
])

# Для валидации (только нормализация)
val_transform = A.Compose([
    # A.Resize(256, 256),
    A.Normalize(
        mean=[0.3456, 0.3881, 0.3476],  # Inria-специфичные значения
        std=[0.2037, 0.1886, 0.1815], 
        max_pixel_value=255.0
    ),
    ToTensorV2(),
])