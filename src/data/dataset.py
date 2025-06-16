import glob
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader


class InriaDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.float() / 255.
    

# def get_loaders(train_dir, val_dir, batch_size, train_transform, val_transform):
#     train_dataset = InriaDataset(
#         images_dir=f"{train_dir}/images",
#         masks_dir=f"{train_dir}/masks",
#         transform=train_transform,
#     )
#     val_dataset = InriaDataset(
#         images_dir=f"{val_dir}/images",
#         masks_dir=f"{val_dir}/masks",
#         transform=val_transform,
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#     )
#     return train_loader, val_loader