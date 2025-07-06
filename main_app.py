import streamlit as st
from configs.base import *
from src.app.model_utils import load_model
from src.app.viz import show_results
import os
import torch


model_weights_path = os.path.join(PROJECT_PATH, 'experiments/checkpoints/unet_resnet34_buildings_best.pth')

# Интерфейс
st.title("Сегментация зданий")
uploaded_file = st.file_uploader("Загрузите спутниковый снимок", type=["jpg", "png", "tif"])

if uploaded_file:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        weights_path=model_weights_path,
        device=device
    )
    show_results(model, uploaded_file)