# 🏙️ Semantic Segmentation of Buildings from Aerial Imagery

**Final project for Deep Learning School @ MIPT (ФМПИ, МФТИ)**  
Semantic segmentation of buildings using satellite images from the **Inria Aerial Image Labeling Dataset**.

## 📌 Project Overview

This project tackles the task of segmenting building footprints from high-resolution aerial imagery.  
We use a custom U-Net model with a ResNet-34 encoder trained **from scratch** (no pretrained weights).

> 🔍 Goal: generate accurate binary masks of buildings from satellite images.

## 🧠 Model Details

- **Architecture**: U-Net + ResNet-34 encoder
- **Loss Function**: Combined Dice Loss + Binary Cross Entropy (BCE)
- **Trained From Scratch**: No ImageNet weights used
- **Environment**: Google Colab, Kaggle

## 📊 Evaluation Metrics

- **IoU (Intersection over Union)**
- **F1-score**
📈 *(to be added below asap)*

## 🖼️ Example Predictions

| Input Image |    Mask    | Input + Mask |
|-------------|------------|--------------|
| ![](https://drive.google.com/file/d/1Mj-gG1QZkvH86kGn5LOonVZPdqNU6K3i/view?usp=drive_link) | ![](https://drive.google.com/file/d/1MBF2gEA_C9Qp-hZrl3uRoKvFBQEiffIH/view?usp=sharing) | ![](https://drive.google.com/file/d/14J4dO76fW8vtHbUh4zN4ZyDltiRMiWVG/view?usp=sharing) |

## 🚀 Try it yourself

You can test the model online:  
🔗 **[Streamlit Demo](https://geodatadls.streamlit.app/)** — upload your own image and get building masks in real time.

## 📁 Repository Structure

GEODATA_DLS/
├── configs/ # Конфигурации и базовые параметры
├── experiments/ # Результаты обучения (веса, логи, предсказания)
├── src/ # Основной код проекта
│ ├── app/ # Streamlit-интерфейс и визуализация
│ ├── data/ # Загрузка, препроцессинг и аугментации данных
│ ├── models/ # Архитектура модели и функция потерь
│ └── utils/ # Метрики, сохранение, визуализация обучения
├── train.py # Скрипт для обучения модели
├── app.py # Точка входа в Streamlit-приложение
├── main_train.ipynb # Jupyter-ноутбук для локального обучения
├── main_train_kaggle.ipynb # Ноутбук под Kaggle-среду
├── requirements.txt # Зависимости проекта
├── README.md # Описание проекта



## 💬 Citation / Credits

- **Dataset**: [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
- **Course**: [Deep Learning School, MIPT (ФМПИ МФТИ)](https://dls.samcs.ru/)
- **Author**: Evgenii Ilnitski