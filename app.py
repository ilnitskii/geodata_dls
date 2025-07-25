import streamlit as st
from configs.base import *
from src.models.unet import UNet
from src.app.viz import show_results
import os
import torch
import requests
# import gdown

MODEL_DIR = "experiments/checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "UNet_weights.pth")


# DROPBOX
@st.cache_resource
def download_model():
    """Скачивает модель с Dropbox"""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            

            url = "https://www.dropbox.com/scl/fi/4xa1vcfuuk2hjxtp2kyc7/UNet_weights.pth?rlkey=7jphy7glcxoo35qw9iuew6514&st=zuw9jeqi&dl=1"
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
            else:
                st.error(f"Ошибка при скачивании модели: {response.status_code}")
                st.stop()

        return MODEL_PATH

    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.stop()

# MODEL_NAME = "UNet_best.pth"
# MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
# GOOGLE_DRIVE_ID = "15ptjzR8jtS5VsyaXrGmHts0YOrHD3md3"
# GOOGLE DRIVE
# @st.cache_resource(ttl=3600)  # Кэш на 1 час (можно обновить при необходимости)
# def download_model():
#     """Скачивает модель с Google Drive при необходимости"""
#     try:
#         os.makedirs(MODEL_DIR, exist_ok=True)
        
#         if not os.path.exists(MODEL_PATH):
#             st.info("Загрузка модели... Это может занять несколько минут")
#             url = f"https://drive.google.com/uc?id={file_id}"
#             gdown.download(url, MODEL_PATH, quiet=False)
            
#             # Проверка, что файл скачался
#             if not os.path.exists(MODEL_PATH):
#                 st.error("Не удалось загрузить модель!")
#                 st.stop()
                
#         return MODEL_PATH
    
#     except Exception as e:
#         st.error(f"Ошибка загрузки модели: {str(e)}")
#         st.stop()

def load_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Загружает модель с весами"""
    try:
        # Проверяем доступность файла
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"File of model not found: {weights_path}")
        
        # Загружаем чекпоинт
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Инициализируем модель
        model = UNet()
        
        # Совместимость с разными форматами чекпоинтов
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

def dual_text(en, ru, level=1):
    st.markdown(f"""
    <h{level} style='margin-bottom: 0'>{en}</h{level}>
    <p style='font-size: 0.9em; color: rgba(255, 255, 255, 0.5); margin-top: 0'>
    {ru}
    </p>
    """, unsafe_allow_html=True)

# Основной интерфейс
dual_text("Segmentation of buildings using satellite images",
          "Сегментация зданий на спутниковых снимках", level=1)

# Скачиваем модель (если нужно)
model_path = download_model()

# Загружаем модель в память (кэшируем)
@st.cache_resource
def get_model():
    return load_model(model_path, device='cpu')

# Загрузка изображения
uploaded_file = st.file_uploader(
    label="Upload your satellite image / Загрузите спутниковый снимок",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file:
    model = get_model()
    show_results(model, uploaded_file)
        
st.markdown("""
    <hr style="margin-top: 50px;">
    <div style='text-align: center; color: grey; font-size: 14px;'>
        © 2025 Created by <b>Evgenii Ilnitski</b> |
        <a href="https://github.com/ilnitskii/geodata_dls" target="_blank">GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)  