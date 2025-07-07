import streamlit as st
from configs.base import *
from src.app.model_utils import load_model
from src.app.viz import show_results
import os
import torch


APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_weights_path = os.path.join(APP_PATH, 'experiments/checkpoints/unet_resnet34_buildings_best.pth')

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