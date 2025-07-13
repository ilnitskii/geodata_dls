# 🏙️ Semantic Segmentation of Buildings from Aerial Imagery

**Final project for Deep Learning School @ MIPT (ФМПИ, МФТИ)**  
Semantic segmentation of buildings using satellite images from the **Inria Aerial Image Labeling Dataset**.

## 📌 Project Overview

This project tackles the task of segmenting building footprints from high-resolution aerial imagery.  
We use a custom U-Net model with a ResNet-34 encoder trained **from scratch** (no pretrained weights).

> 🔍 Goal: generate accurate binary masks of buildings from satellite images.

## 🚀 Try it yourself

You can test the model online:  
🔗 **[Streamlit Demo](https://geodatadls.streamlit.app/)** — upload your own image and get building masks in real time.

## 🖼️ Example Predictions

| Input Image |    Mask    | Input + Mask |
|-------------|------------|--------------|
| ![](experiments/predictions/input.jpg) | ![](experiments/predictions/mask.jpg) | ![](experiments/predictions/inp+mask.jpg) |

## 🧠 Model Details

- **Architecture**: U-Net + ResNet-34 encoder
- **Loss Function**: Combined Dice Loss + Binary Cross Entropy (BCE)
- **Trained From Scratch**: No ImageNet weights used
- **Environment**: Google Colab, Kaggle

## 📊 Evaluation Metrics

- **IoU (Intersection over Union)**
- **F1-score**

[](experiments/predictions/metrics.png)

## 📁 Project Structure

app.py — Launching the Streamlit application  
main_train.ipynb — Training in Colab  
main_train_kaggle.ipynb — Training in Kaggle  
requirements.txt — Dependencies  
README.md — Project description  
configs/ — Project configurations  
experiments/ — Checkpoints, logs, predictions  
src/app/ — Streamlit interface and visualization  
src/data/ — Data loading and processing  
src/models/ — Model and loss function  
src/utils/ — Metrics, saving, plots  
src/train.py — Model training   


## 💬 Citation / Credits

- **Dataset**: [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
- **Course**: [Deep Learning School, MIPT (ФМПИ МФТИ)](https://dls.samcs.ru/)
- **Author**: Evgenii Ilnitski