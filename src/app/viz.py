import numpy as np
import streamlit as st
from PIL import Image
import torch

def preprocess(image, size=512):
    """Подготовка изображения для модели"""
    # ... ваш код препроцессинга ...

def show_results(model, uploaded_file):
    """Отображение результатов"""
    with st.spinner("Обработка..."):
        image = Image.open(uploaded_file)
        tensor = preprocess(image)
        
        with torch.no_grad():
            mask = model(tensor)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Исходное изображение")
        with col2:
            st.image(
                mask.squeeze().numpy() > 0.5, 
                caption="Сегментация зданий"
            )