
import numpy as np
from PIL import Image
import os
from glob import glob
from configs.base import PROJECT_PATH

def split_image_into_patches(image_path, output_dir, patch_size=512, overlap=64):
    """
    Разбивает изображение на патчи с заданным перекрытием.
    
    Параметры:
        image_path (str): Путь к исходному изображению
        output_dir (str): Директория для сохранения патчей
        patch_size (int): Размер патча (квадратный)
        overlap (int): Перекрытие между патчами в пикселях
    """

    # Получаем название файла
    filename_with_ext = os.path.basename(image_path)
    filename = os.path.splitext(filename_with_ext)[0]

    # Загрузка изображения
    image = Image.open(image_path)
    img_array = np.array(image)
    
    # Проверка размеров изображения
    height, width = img_array.shape[:2]
    # print(f"Исходное изображение: {width}x{height} пикселей")
    
    # Создание директории для сохранения (если не существует)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Вычисление шага (stride)
    stride = patch_size - overlap
    
    # Подсчет количества патчей по ширине и высоте
    num_patches_x = (width - overlap) // stride
    num_patches_y = (height - overlap) // stride
    
    # Корректировка, если изображение не делится ровно
    if (width - overlap) % stride != 0:
        num_patches_x += 1
    if (height - overlap) % stride != 0:
        num_patches_y += 1
    
    # print(f"Будет создано {num_patches_x}x{num_patches_y} = {num_patches_x*num_patches_y} патчей")
    
    # Генерация и сохранение патчей
    patch_num = 0
    for y in range(0, height - overlap, stride):
        for x in range(0, width - overlap, stride):
            # Определение координат
            x_start = x
            y_start = y
            x_end = min(x_start + patch_size, width)
            y_end = min(y_start + patch_size, height)
            
            # Извлечение патча
            patch = img_array[y_start:y_end, x_start:x_end]
            
            # Если патч меньше нужного размера, дополняем нулями
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_height = patch_size - patch.shape[0]
                pad_width = patch_size - patch.shape[1]
                
                if len(img_array.shape) == 3:  # Цветное изображение
                    patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
                else:  # Градации серого
                    patch = np.pad(patch, ((0, pad_height), (0, pad_width)), mode='constant')
            
            # Сохранение патча
            patch_img = Image.fromarray(patch)
            output_path = os.path.join(output_dir, f"{filename}_patch_{patch_num:04d}.tif")
            patch_img.save(output_path)
            patch_num += 1
    
    # print(f"Сохранено {patch_num} патчей в директорию {output_dir}")


def create_patches_to_disc(images_path, gt_path, images_patches_path, gt_patches_path):
    for img_name in os.listdir(images_path):
        if img_name.endswith('.tif'):
            mask_name = img_name  # Предполагаем одинаковые имена
            img_path = os.path.join(images_path, img_name)
            mask_path = os.path.join(gt_path, mask_name)
            split_image_into_patches(img_path, images_patches_path, patch_size=512, overlap=8)
            split_image_into_patches(mask_path, gt_patches_path, patch_size=512, overlap=8)

    # список путей к картинкам
    image_paths = sorted(glob(os.path.join(images_patches_path, "*.tif")))  
    gt_paths = sorted(glob(os.path.join(gt_patches_path, "*.tif"))) 
    return image_paths, gt_paths
        
