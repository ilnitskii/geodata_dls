import streamlit as st
import numpy as np
from PIL import Image
import io
import torch
from src.app.viz_utils import *



def dual_text(en, ru, level=2):
    st.markdown(f"""
    <h{level} style='margin-bottom: 0'>{en}</h{level}>
    <p style='font-size: 0.9em; color: rgba(255, 255, 255, 0.5); margin-top: 0'>
    {ru}
    </p>
    """, unsafe_allow_html=True)

def show_results(model, uploaded_file):
    """Обработка и отображение результатов с делением на патчи"""

    with st.spinner("Processing large image... / Обработка большого изображения..."):
        # Загрузка изображения
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        # Разбиение на патчи
        patches, coords = split_to_patches(img_array, patch_size=512, overlap=64)
        
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
        
        # Создание наложенного изображения
        overlay = overlay_mask(img_array, binary_mask)

        dual_text("Segmentation Results", "Результаты сегментации", level=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original image / Исходное изображение", use_container_width=True)
        with col2:
            st.image(binary_mask, caption="Building segmentation / Сегментация зданий", use_container_width=True, clamp=True)

        # Пиксельная шкала
        show_ruler = st.checkbox("Show pixel ruler / Показать пиксельную шкалу", False)
        if show_ruler:
            tick_step = st.slider("Tick step (pixels) / Шаг шкалы (пикселей)", 20, 200, 50)
            overlay_with_ruler = add_pixel_ruler(overlay, tick_step)

        # Отображение наложенного изображения
        st.image(
            overlay_with_ruler if show_ruler else overlay,
            caption="Overlay with mask / Наложение маски на исходное изображение",
            width=800
        )

        dual_text("Scale Settings", "Настройки масштаба", level=2)
        use_scale = st.checkbox("Calculate area in m² / Рассчитать площадь в м²", value=False)
        scale_ppm = get_scale_from_user() if use_scale else None   

        # Скачивание результата
        buf = io.BytesIO()
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(buf, format='PNG')
        byte_im = buf.getvalue()
        original_name = uploaded_file.name.split('.')[0]
        file_name = f"{original_name}_overlay.png"

        st.download_button(
            label="Download result with mask (PNG) / Скачать изображение с наложенной маской (PNG)",
            data=byte_im,
            file_name=file_name,
            mime="image/png"
        )

        # Статистика
        mask = full_mask > 0.5
        stats = calculate_mask_stats(img_array, mask)

        dual_text("Mask Statistics", "Статистика маски", level=2)
        st.write(f"🔵 Total pixels / Общее количество пикселей: {stats['total_pixels']:,}")
        st.write(f"🟢 Mask pixels / Пикселей маски: {stats['mask_pixels']:,}")
        st.write(f"📊 Coverage / Процент покрытия: {stats['coverage_percent']:.2f}%")

        if scale_ppm is not None:
            area_m2 = calculate_area_m2(stats['mask_pixels'], scale_ppm)
            st.write(f"📏 Area: {area_m2:.2f} m² / Площадь: {area_m2:.2f} м² (проверьте указанный масштаб в левой части интерфейса)")


# def show_results(model, uploaded_file):
#     """Обработка и отображение результатов с делением на патчи"""

#     with st.spinner("Обработка большого изображения..."):
#         # Загрузка изображения
#         image = Image.open(uploaded_file).convert("RGB")
#         img_array = np.array(image)
        
#         # Разбиение на патчи
#         patches, coords = split_to_patches(img_array, patch_size=512, overlap=64)
        
#         # Прогноз для каждого патча
#         masks = []
#         progress_bar = st.progress(0)
#         for i, patch in enumerate(patches):
#             tensor = preprocess_patch(patch).to(next(model.parameters()).device)
#             with torch.no_grad():
#                 pred = model(tensor).squeeze().cpu().numpy()
#             masks.append(pred)
#             progress_bar.progress((i + 1) / len(patches))
        
#         # Сборка полного изображения
#         full_mask = merge_patches(masks, coords, img_array.shape)
#         binary_mask = (full_mask > 0.5).astype(np.uint8) * 255
        
#         # Создание наложенного изображения
#         overlay = overlay_mask(img_array, binary_mask)
        
#         st.subheader("Результаты сегментации")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption="Исходное изображение", 
#                     use_container_width=True)
#         with col2:
#             st.image(binary_mask, caption="Сегментация зданий", 
#                     use_container_width=True,
#                     clamp=True)
            
#         show_ruler = st.checkbox("Показать пиксельную шкалу", False)
#         if show_ruler:
#             tick_step = st.slider("Шаг шкалы (пикселей)", 20, 200, 50)
#             overlay_with_ruler = add_pixel_ruler(overlay, tick_step)

#         # Наложение маски
#         st.image(overlay_with_ruler if show_ruler else overlay, 
#                 caption="Наложение маски на исходное изображение", 
#                 width=800)
        
#         st.subheader("Настройки масштаба")
#         use_scale = st.checkbox("Рассчитать площадь в м²", value=False)
#         scale_ppm = get_scale_from_user() if use_scale else None   

#         # Скачивание результата
#         buf = io.BytesIO()  # Создаем буфер в памяти
#         overlay_pil = Image.fromarray(overlay)  # Конвертация в PIL Image
#         overlay_pil.save(buf, format='PNG')  # Сохраняем изображение в буфер
#         byte_im = buf.getvalue()  # Получаем байты
        
#         original_name = uploaded_file.name.split('.')[0]
#         file_name = f"{original_name}_overlay.png"

#         st.download_button(
#             label="Скачать изображение с наложенной маской (PNG)",
#             data=byte_im,
#             file_name=file_name,
#             mime="image/png"
#         )
        
#         # Статистика
#         mask = full_mask > 0.5
#         stats = calculate_mask_stats(img_array, mask)
#         st.subheader("Статистика маски")
#         st.write(f"🔵 Общее количество пикселей: {stats['total_pixels']:,}")
#         st.write(f"🟢 Пикселей маски: {stats['mask_pixels']:,}")
#         st.write(f"📊 Процент покрытия: {stats['coverage_percent']:.2f}%")

#         if scale_ppm is not None:
#             area_m2 = calculate_area_m2(stats['mask_pixels'], scale_ppm)
#             st.write(f"📏 Площадь маски: {area_m2:.2f} м² (проверьте указанный масштаб изображения в левой части интерфейса)")
            
    