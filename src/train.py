import torch
import random
from src.utils.visualize_img_gt import visualize_images_and_masks
from configs.base import *

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, metrics, device):
    model.eval()
    val_loss = 0.0

    # Визуализируем случайные две картинки из val_loader'а
    i, j = random.sample(range(BATCH_SIZE//2), 2) 
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            
            outputs = model(images)
            metrics.update(outputs, masks)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

        visualize_images_and_masks(images, outputs, masks, n1=i, n2=j) 
        
    return val_loss / len(dataloader), metrics.compute()