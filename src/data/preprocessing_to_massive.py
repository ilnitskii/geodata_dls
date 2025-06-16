
import ....

def image_to_patches(image_path, patch_size, overlap):
    """
    Разбивает изображение на патчи с заданным перекрытием и возвращает массив патчей.
    
    Параметры:
        image_path (str): Путь к исходному изображению
        patch_size (int): Размер патча (квадратный)
        overlap (int): Перекрытие между патчами в пикселях
    
    Возвращает:
        list: Список numpy массивов с патчами
    """
    # Загрузка изображения
    image = Image.open(image_path)
    img_array = np.array(image)
    
    # Проверка размеров изображения
    height, width = img_array.shape[:2]
    
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
    
    # Список для хранения патчей
    patches = []
    
    # Генерация патчей
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
                    patch = np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), 
                                mode='constant', constant_values=0)
                else:  # Градации серого
                    patch = np.pad(patch, ((0, pad_height), (0, pad_width)), 
                                mode='constant', constant_values=0)
            
            # Добавление патча в список
            patches.append(patch)
    
    # Преобразование списка в numpy массив для удобства
    patches_array = np.array(patches)
    
    # Возвращаем массив патчей и размерность сетки
    return patches_array


def create_patches_to_array(dataset_path, patch_size, overlap):
    """
    Функция создает массивы патчей из исходных изображений
    На вход подается путь к датасету (внутри должны находиться папки images и gt)

    Параметр:
        dataset_path - путь к папке с датасетом
    Выход:
        Массив патчей картинки и массив патчей gt
    """
    images_folder_path = os.path.join(dataset_path, 'images')
    gt_folder_path = os.path.join(dataset_path, 'gt')

    all_image_patches_list = []
    all_gt_patches_list = []
    for img_name in os.listdir(images_folder_path):
        if img_name.endswith('.tif'):
            img_path = os.path.join(images_folder_path, img_name)
            gt_path = os.path.join(gt_folder_path, img_name)

            image_patches = image_to_patches(img_path, patch_size, overlap)
            gt_patches = image_to_patches(gt_path, patch_size, overlap) / 255
            
            all_image_patches_list.append(image_patches)
            all_gt_patches_list.append(gt_patches)

    all_image_patches = np.vstack(all_image_patches_list)
    all_gt_patches = np.vstack(all_gt_patches_list)

    return all_image_patches, all_gt_patches


