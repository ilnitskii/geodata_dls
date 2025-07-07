import streamlit as st
from configs.base import *
from src.models.unet import UNet
from src.app.viz import show_results
import os
import torch
import gdown

# Константы путей
MODEL_DIR = "experiments/checkpoints"
MODEL_NAME = "UNet_best.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
GOOGLE_DRIVE_ID = "15ptjzR8jtS5VsyaXrGmHts0YOrHD3md3"

@st.cache_resource(ttl=3600)  # Кэш на 1 час (можно обновить при необходимости)
def download_model():
    """Скачивает модель с Google Drive при необходимости"""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if not os.path.exists(MODEL_PATH):
            st.info("Загрузка модели... Это может занять несколько минут")
            url = f"https://drive.google.com/uc?export=download&confirm=pbef&id={GOOGLE_DRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
            # Проверка, что файл скачался
            if not os.path.exists(MODEL_PATH):
                st.error("Не удалось загрузить модель!")
                st.stop()
                
        return MODEL_PATH
    
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.stop()

def load_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Загружает модель с весами"""
    try:
        # Проверяем доступность файла
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл модели не найден: {weights_path}")
        
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
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.stop()

# Основной интерфейс
st.title("Сегментация зданий по спутниковым снимкам")

# Скачиваем модель (если нужно)
model_path = download_model()

# Загружаем модель в память (кэшируем)
@st.cache_resource
def get_model():
    return load_model(model_path, device='cpu')

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Загрузите спутниковый снимок", 
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Поддерживаются форматы JPG, PNG, TIFF"
)

if uploaded_file:
    # Показываем индикатор загрузки
    with st.spinner("Обработка изображения..."):
        model = get_model()
        show_results(model, uploaded_file)
        
        
        
# import streamlit as st
# from configs.base import *
# from src.models.unet import UNet
# from src.app.viz import show_results
# import os
# import torch
# import gdown

# MODEL_DIR = "experiments/checkpoints"
# MODEL_PATH = os.path.join(MODEL_DIR, "UNet_best.pth")

# @st.cache_resource  # Кэшируем загрузку модели
# def load_pth():
#     # Создаем директорию, если её нет
#     os.makedirs(MODEL_DIR, exist_ok=True)
    
#     # Если файл уже существует - не качаем снова
#     if not os.path.exists(MODEL_PATH):
#         # ID вашего файла на Google Drive
#         file_id = "1HQaNpXAZhSHS0J9kdKYXDQa3muZMdfd4"
#         gdrive_url = f"https://drive.google.com/uc?id={file_id}"

#         # Скачиваем файл
#         gdown.download(gdrive_url, MODEL_PATH, quiet=False)
#         st.success("Модель успешно загружена!")
#     else:
#         st.info("Модель уже загружена")
    
#     return MODEL_PATH

# def load_model(weights_path, device="cpu"):
#     """Загрузка модели с весами"""
#     checkpoint = torch.load(weights_path, map_location=device)
#     model = UNet(n_classes=1)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     # model.load_state_dict(torch.load(
#     #     (weights_path),
#     #     map_location=torch.device(device)
#     # ))
#     model.to(device)
#     model.eval()
#     return model

# # APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # model_weights_path = os.path.join(APP_PATH, 'experiments/checkpoints/UNet_best.pth')

# model_weights_path = load_pth()

# # Интерфейс
# st.title("Сегментация зданий")
# uploaded_file = st.file_uploader("Загрузите спутниковый снимок", type=["jpg", "png", "tif"])

# if uploaded_file:
#     DEVICE = torch.device('cpu')
#     model = load_model(
#         weights_path=model_weights_path,
#         device=DEVICE
#     )
#     show_results(model, uploaded_file)