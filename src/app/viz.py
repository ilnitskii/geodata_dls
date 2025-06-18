import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2
from src.data.transforms import *

def split_to_patches(image_array, patch_size=512, overlap=64):
    """Разбивает изображение на перекрывающиеся патчи"""
    h, w = image_array.shape[:2]
    stride = patch_size - overlap
    patches = []
    coords = []
    
    for y in range(0, h - overlap, stride):
        for x in range(0, w - overlap, stride):
            x_end = min(x + patch_size, w)
            y_end = min(y + patch_size, h)
            
            patch = image_array[y:y_end, x:x_end]
            
            # Дополнение если патч меньше нужного размера
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(patch,
                              ((0, patch_size - patch.shape[0]),
                               (0, patch_size - patch.shape[1]),
                               (0, 0)),
                              mode='reflect')
            
            patches.append(patch)
            coords.append((x, y, x_end, y_end))
    
    return patches, coords

def merge_patches(patches, coords, original_shape):
    """Собирает предсказания из патчей в полное изображение"""
    full_mask = np.zeros(original_shape[:2], dtype=np.float32)
    count = np.zeros(original_shape[:2], dtype=np.float32)
    
    for patch, (x, y, x_end, y_end) in zip(patches, coords):
        h, w = y_end - y, x_end - x
        full_mask[y:y_end, x:x_end] += patch[:h, :w]
        count[y:y_end, x:x_end] += 1
    
    return full_mask / count

def preprocess_patch(patch):
    """Препроцессинг отдельного патча"""
    transformed = val_transform(image=patch)
    return transformed["image"].unsqueeze(0) 

def show_results(model, uploaded_file):
    """Обработка и отображение результатов с делением на патчи"""
    with st.spinner("Обработка большого изображения..."):
        # Загрузка изображения
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Разбиение на патчи
        patches, coords = split_to_patches(img_array, patch_size=512, overlap=8)
        
        # Прогноз для каждого патча
        masks = []
        progress_bar = st.progress(0)
        for i, patch in enumerate(patches):
            tensor = preprocess_patch(patch).to(next(model.parameters()).device)
            with torch.no_grad():
                pred = model(tensor).squeeze().cpu().numpy()
            masks.append(pred)
            progress_bar.progress((i + 1) / len(patches))
        
        # Сборка полного изображения
        full_mask = merge_patches(masks, coords, img_array.shape)
        binary_mask = (full_mask > 0.5).astype(np.uint8) * 255
        
        # Визуализация
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Исходное изображение")
        with col2:
            st.image(binary_mask, caption="Сегментация зданий", 
                    clamp=True)
        
        # Опция скачивания результата
        # result_img = Image.fromarray(binary_mask)
        # with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        #     result_img.save(tmp.name)
        #     st.download_button(
        #         label="Скачать маску",
        #         data=tmp.read(),
        #         file_name="building_mask.png",
        #         mime="image/png"
        #     )