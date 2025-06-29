import torch
from torchmetrics.classification import BinaryJaccardIndex, BinaryRecall, BinaryPrecision
from torchmetrics.segmentation import DiceScore

class SegmentationMetrics:
    def __init__(self, device='cuda', threshold=0.5):
        """
        Инициализация метрик для бинарной семантической сегментации.
        
        Параметры:
            device: устройство для вычислений ('cuda' или 'cpu')
            threshold: порог бинаризации предсказаний
        """
        self.threshold = threshold
        self.device = device
        
        # Используем специализированные бинарные версии метрик
        self.iou = BinaryJaccardIndex(threshold=threshold).to(device)
        self.dice = DiceScore(num_classes=1, average='micro').to(device)
        self.precision = BinaryPrecision(threshold=threshold).to(device)
        self.recall = BinaryRecall(threshold=threshold).to(device)

    def update(self, preds, targets):
        """
        Обновление состояния метрик для нового батча
        
        Параметры:
            preds: предсказания модели (логиты) [B, 1, H, W]
            targets: ground truth маски [B, 1, H, W]
        """
        # Проверка размерностей
        if preds.dim() != 4 or targets.dim() != 4:
            raise ValueError("Input tensors must be 4D: [B, C, H, W]")
            
        # Бинаризация с порогом
        preds = (torch.sigmoid(preds) > self.threshold).float()
        
        # Обновление всех метрик
        self.iou.update(preds, targets)
        self.dice.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)

    def compute(self):
        """
        Вычисление всех метрик и сброс состояния
        
        Возвращает:
            Словарь с вычисленными метриками
        """
        results = {
            "IoU": self.iou.compute(),
            "Dice": self.dice.compute(),
            "Precision": self.precision.compute(),
            "Recall": self.recall.compute(),
            "F1": 2 * (self.precision.compute() * self.recall.compute()) / 
                 (self.precision.compute() + self.recall.compute() + 1e-6)
        }
        self.reset()
        return results

    def reset(self):
        """Сброс состояния всех метрик"""
        self.iou.reset()
        self.dice.reset()
        self.precision.reset()
        self.recall.reset()