import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

class InriaDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Compose = None,
        patch_size: int = None,
        overlap: int = 0,
        is_train: bool = True,
    ):
        """
        Args:
            images_dir (str): Путь к папке с изображениями.
            masks_dir (str): Путь к папке с масками.
            transform (albumentations.Compose): Аугментации и препроцессинг.
            patch_size (int, optional): Размер патча для нарезки (если None - без нарезки).
            overlap (int): Перекрытие между патчами (пиксели).
            is_train (bool): Флаг для тренировочного/валидационного режима.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.patch_size = patch_size
        self.overlap = overlap
        self.is_train = is_train

        # Получаем список файлов
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))

        # Проверка соответствия изображений и масок
        assert len(self.image_paths) == len(self.mask_paths), "Количество изображений и масок не совпадает!"
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            assert os.path.basename(img_path) == os.path.basename(mask_path), "Имена файлов не совпадают!"

        # Предварительно вычисляем координаты патчей (если patch_size задан)
        self.patches_info = []
        if self.patch_size is not None:
            self._precompute_patches()

    def _precompute_patches(self):
        """Вычисляет координаты всех патчей для всех изображений."""
        for img_idx in range(len(self.image_paths)):
            image = cv2.imread(self.image_paths[img_idx], cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            stride = self.patch_size - self.overlap

            # Количество патчей по высоте и ширине
            num_y = (h - self.overlap) // stride + (1 if (h - self.overlap) % stride != 0 else 0)
            num_x = (w - self.overlap) // stride + (1 if (w - self.overlap) % stride != 0 else 0)

            # Генерация координат
            for y in range(num_y):
                for x in range(num_x):
                    y_start = y * stride
                    x_start = x * stride
                    y_end = min(y_start + self.patch_size, h)
                    x_end = min(x_start + self.patch_size, w)
                    self.patches_info.append((img_idx, x_start, y_start, x_end, y_end))

    def __len__(self):
        """Возвращает общее количество патчей или изображений."""
        return len(self.patches_info) if self.patch_size else len(self.image_paths)

    def __getitem__(self, idx):
        """Загружает и возвращает патч (image, mask) по индексу."""
        if self.patch_size is not None:
            # Загрузка патча из большого изображения
            img_idx, x_start, y_start, x_end, y_end = self.patches_info[idx]
            image = cv2.imread(self.image_paths[img_idx], cv2.IMREAD_COLOR)
            mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
            
            # Вырезаем патч
            image_patch = image[y_start:y_end, x_start:x_end]
            mask_patch = mask[y_start:y_end, x_start:x_end]

            # Дополнение до нужного размера (если патч меньше)
            if image_patch.shape[0] < self.patch_size or image_patch.shape[1] < self.patch_size:
                pad_h = max(0, self.patch_size - image_patch.shape[0])
                pad_w = max(0, self.patch_size - image_patch.shape[1])
                image_patch = np.pad(image_patch, ((0, pad_h), (0, pad_w)), mode="constant")
                mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode="constant")
        else:
            # Загрузка полного изображения
            image_patch = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
            mask_patch = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Нормализация маски (0-1)
        mask_patch = (mask_patch > 0).astype(np.float32)

        # Аугментации (Albumentations)
        if self.transform:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_patch = transformed["image"]
            mask_patch = transformed["mask"]

        # Конвертация в тензоры PyTorch (если аугментации не включают ToTensorV2)
        if not isinstance(image_patch, torch.Tensor):
            image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
            mask_patch = torch.from_numpy(mask_patch).unsqueeze(0).float()

        return image_patch, mask_patch

def get_loaders(train_dir, val_dir, batch_size, train_transform, val_transform):
    train_dataset = InriaDataset(
        images_dir=f"{train_dir}/images",
        masks_dir=f"{train_dir}/gt",
        transform=train_transform,
        patch_size=512,
        overlap=8
    )
    val_dataset = InriaDataset(
        images_dir=f"{val_dir}/images",
        masks_dir=f"{val_dir}/masks",
        transform=val_transform,
        patch_size=512,
        overlap=8
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader
