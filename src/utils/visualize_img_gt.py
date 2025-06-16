import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_images_and_masks(images, pred_masks, gt_masks, n1=0, n2=1):
    """
    Визуализирует нормализованные изображения, предсказанные маски и GT маски в 2 строки.
    Каждая строка содержит: исходное изображение, pred маску, GT маску.
    
    Параметры:
        images: тензор изображений [B, 3, H, W] (минимум 2 изображения)
        pred_masks: тензор предсказанных масок [B, 1, H, W]
        gt_masks: тензор GT масок [B, 1, H, W]
    """
    fig, axes = plt.subplots(2, 3, figsize=(5, 3))  # 2 строки, 3 колонки
    
    # Переносим параметры нормализации на то же устройство, что и images
    device = images.device
    mean = torch.tensor([0.3456, 0.3881, 0.3476], device=device).view(3, 1, 1)
    std = torch.tensor([0.2037, 0.1886, 0.1815], device=device).view(3, 1, 1)
    
    # Обратная нормализация (на том же устройстве)
    with torch.no_grad():
        denorm_images = images * std + mean  # [B, 3, H, W]
        denorm_images = denorm_images.permute(0, 2, 3, 1)  # [B, H, W, 3]
        denorm_images = torch.clamp(denorm_images, 0, 1)    # Обрезаем значения
        denorm_images_np = denorm_images.cpu().numpy()      # Переводим в numpy
        
        # Обработка масок
        pred_masks_np = torch.sigmoid(pred_masks).cpu().numpy()  # [B, 1, H, W] -> [0, 1]
        gt_masks_np = gt_masks.cpu().numpy()  # [B, 1, H, W]
    
    # Первая строка (первый пример)
    axes[0, 0].imshow(denorm_images_np[n1])
    axes[0, 0].set_title('Image 1', fontsize='small')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_masks_np[n1].squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Pred Mask 1', fontsize='small')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt_masks_np[n1].squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('GT Mask 1', fontsize='small')
    axes[0, 2].axis('off')
    
    # Вторая строка (второй пример)
    axes[1, 0].imshow(denorm_images_np[n2])
    axes[1, 0].set_title('Image 2', fontsize='small')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_masks_np[n2].squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Pred Mask 2', fontsize='small')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(gt_masks_np[n2].squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('GT Mask 2', fontsize='small')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()