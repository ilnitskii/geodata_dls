
import matplotlib.pyplot as plt
import os
from configs.base import *

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Losses
    ax1.plot(history['train']['loss'], label='Train')
    ax1.plot(history['val']['loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.legend()
    
    # Metrics
    for metric, values in history['val']['metrics'].items():
        ax2.plot(values, label=metric)
    ax2.set_title('Validation Metrics')
    ax2.legend()
    
    plt.savefig(os.path.join(PROJECT_PATH, 'experiments/plots/training_plot.png'))
    plt.close()

