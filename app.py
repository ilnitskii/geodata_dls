import streamlit as st
from configs.base import *
from src.models.unet import UNet
from src.app.viz import show_results
import os
import torch



def load_model(weights_path, device="cpu"):
    """Загрузка модели с весами"""
    checkpoint = torch.load(weights_path, map_location=device)
    model = UNet(n_classes=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(
    #     (weights_path),
    #     map_location=torch.device(device)
    # ))
    model.to(device)
    model.eval()
    return model

# APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model_weights_path = os.path.join(APP_PATH, 'experiments/checkpoints/UNet_best.pth')

model_weights_path = os.path.join('experiments', 'checkpoints', 'UNet_best.pth')

# Интерфейс
st.title("Сегментация зданий")
uploaded_file = st.file_uploader("Загрузите спутниковый снимок", type=["jpg", "png", "tif"])

if uploaded_file:
    DEVICE = torch.device('cpu')
    model = load_model(
        weights_path=model_weights_path,
        device=DEVICE
    )
    show_results(model, uploaded_file)