import torch
import torchmetrics

class SegmentationMetrics:
    def __init__(self, device='cuda'):
        self.iou = JaccardIndex(task='binary').to(device)
        self.dice = Dice(average='micro').to(device)
        self.precision = Precision(task='binary').to(device)
        self.recall = Recall(task='binary').to(device)

    def update(self, preds, targets):
        preds = (torch.sigmoid(preds) > 0.5).float()
        self.iou.update(preds, targets)
        self.dice.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)

    def compute(self):
        results = {
            "IoU": self.iou.compute(),
            "Dice": self.dice.compute(),
            "Precision": self.precision.compute(),
            "Recall": self.recall.compute()
        }
        self.reset()  # Автоматический сброс после вычислений
        return results

    def reset(self):
        self.iou.reset()
        self.dice.reset()
        self.precision.reset()
        self.recall.reset()

