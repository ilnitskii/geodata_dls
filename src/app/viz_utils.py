import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from src.data.transforms import *

def split_to_patches(image_array, patch_size=512, overlap=64):
    """Разбивает изображение на патчи"""
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

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    """Наложение маски на изображение с прозрачностью"""
    mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay

def calculate_mask_stats(image_array, mask):
    """Вычисляет статистику по маске"""
    # Общее количество пикселей в изображении
    total_pixels = image_array.shape[0] * image_array.shape[1]
    
    # Количество пикселей маски (где mask == True)
    mask_pixels = np.sum(mask)
    
    # Процент покрытия
    coverage_percent = (mask_pixels / total_pixels) * 100
    
    return {
        "total_pixels": total_pixels,
        "mask_pixels": mask_pixels,
        "coverage_percent": coverage_percent
    }

def get_scale_from_user():
    """Интерактивный запрос масштаба у пользователя"""
    st.markdown("**Для расчета масштаба изображения укажите:**")
    known_length = st.number_input(
        "Длина объекта (м):",
        min_value=0.1,
        value=1.0,
        step=0.1
    )
    object_pixels = st.number_input(
        "Его длина в пикселях:",
        min_value=1,
        value=100,
        step=1
    )
    calculated_ppm = object_pixels / known_length
    st.write(f"Рассчитано: {calculated_ppm:.1f} пикселей/метр")
    return calculated_ppm

def calculate_area_m2(mask_pixels, scale_ppm):
    """Вычисляет площадь в квадратных метрах, если задан масштаб"""
    area_m2 = None
    if scale_ppm is not None:
        area_m2 = mask_pixels / (scale_ppm ** 2)
    return area_m2

def add_pixel_ruler(image, tick_step=100):
    """Добавляет пиксельные шкалы по осям изображения"""
    img = image.copy()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    width, height = img.size
    
    # Настройки шкалы
    ruler_size = 40  # Размер области со шкалой
    font_size = 15
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bg_color = (20, 20, 20, 255)  # Цвет фона шкалы с прозрачностью
    text_color = (180, 220, 255)

    # Создаем новое изображение с увеличенными размерами для шкал
    new_width = width + ruler_size
    new_height = height + ruler_size
    new_img = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
    new_img.paste(img, (ruler_size, 0))
    
    
    
    # Добавляем серый фон для шкал
    overlay = Image.new('RGBA', new_img.size, bg_color)
    # Стираем центральную часть (где основное изображение)
    clear_area = (ruler_size, 0, new_width, height)
    overlay.paste((0,0,0,0), clear_area)
    new_img = Image.alpha_composite(new_img, overlay)
    
    # Теперь создаем ImageDraw для нового изображения
    draw = ImageDraw.Draw(new_img)
    
    # Рисуем вертикальную шкалу (ось Y)
    for y in range(0, height, tick_step):
        y_pos = y
        # Линия деления
        draw.line([(ruler_size-10, y_pos), (ruler_size, y_pos)], fill=text_color, width=1)
        # Подпись
        draw.text((5, y_pos-5), str(y), fill=text_color, font=font)
    
    # Рисуем горизонтальную шкалу (ось X)
    for x in range(0, width, tick_step):
        x_pos = x + ruler_size
        # Линия деления
        draw.line([(x_pos, height), (x_pos, height+10)], fill=text_color, width=1)
        # Подпись
        draw.text((x_pos-10, height+15), str(x), fill=text_color, font=font)
        
    return new_img
