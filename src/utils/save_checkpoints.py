import os
import torch
import csv 
from configs.base import *

def save_checkpoint(epoch, model, optimizer, history, val_loss, best_val_loss):
    try:
        model_name = model.__class__.__name__
        checkpoint_dir = os.path.join(PROJECT_PATH, "experiments/checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,  # Вся история обучения (все эпохи)
            'val_loss': val_loss
        }

        # Всегда сохраняем последнее состояние
        last_path = os.path.join(checkpoint_dir, f"{model_name}_last.pth")
        torch.save(checkpoint, last_path)

        # Сохраняем лучшую модель (если текущая лучше)
        if val_loss < best_val_loss:
            history[best_epoch] = epoch
            best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
            torch.save(checkpoint, best_path)
            best_val_loss = val_loss
      

        return best_val_loss

    except Exception as e:
        print(f"Ошибка при сохранении чекпоинта: {e}")
        return best_val_loss  # В случае ошибки возвращаем старое значение:

    

def log_to_csv(epoch_data, history, filename='training_log.csv'):
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ['epoch', 'train_loss', 'val_loss'] + list(history['val']['metrics'].keys())
                writer.writerow(header)
            row = [epoch_data['epoch'], epoch_data['train_loss'], epoch_data['val_loss']]
            row += [epoch_data['metrics'][k] for k in history['val']['metrics'].keys()]
            writer.writerow(row)
    except Exception as e:
        print(f"Ошибка при записи в лог-файл: {e}")

def load_checkpoint(model, optimizer=None):
    """
    Загружает последнее состояние обучения из last.pth или создаёт новое.
    Возвращает:
        - model
        - optimizer (если передан)
        - history
        - best_val_loss
        - initial_epoch
    """
    model_name = model.__class__.__name__
    checkpoint_dir = os.path.join(PROJECT_PATH, "experiments/checkpoints")
    last_path = os.path.join(checkpoint_dir, f"{model_name}_last.pth")
    
    try:
        checkpoint = torch.load(last_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        history = checkpoint['history']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        initial_epoch = checkpoint['epoch'] + 1
        
        print(f"Загружено последнее состояние. Продолжаем с эпохи {initial_epoch}")
        return model, optimizer, history, best_val_loss, initial_epoch
    
    except FileNotFoundError:
        history = {
            'train': {'loss': []},
            'val': {
                'loss': [],
                'metrics': {
                    'IoU': [],
                    'Dice': [],
                    'Precision': [],
                    'Recall': []
                }
            },
            'best_epoch': -1
        }
        print("Обучение начинается с нуля")
        return model, optimizer, history, float('inf'), 0