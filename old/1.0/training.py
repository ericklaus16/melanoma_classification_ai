import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

# Configurações básicas
IMG_SIZE = 128  # Tamanho das imagens (128x128 pixels)
BATCH_SIZE = 32  # Tamanho do lote
EPOCHS = 20  # Número de épocas de treinamento

# Carregar a planilha
labels_df = pd.read_csv('./../../data/ISIC_2019_Training_GroundTruth.csv')

# Função para carregar e processar imagens
def load_images(image_dir, img_size):
    images = []
    labels = []
    class_names = os.listdir(image_dir)  # Lista de categorias (pastas)

    # Iterar sobre cada categoria
    for label in class_names:
        category_path = os.path.join(image_dir, label)
        if os.path.isdir(category_path):  # Verifica se é um diretório
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                # Ler a imagem
                image = cv2.imread(img_path)
                if image is not None:
                    print(f"Carregando: {img_path}")
                    # Redimensionar a imagem
                    image = cv2.resize(image, (img_size, img_size))
                    images.append(image)
                    
                    # Adicionando o rótulo (baseado no índice da classe)
                    label_index = class_names.index(label)  # Obtém o índice da classe
                    labels.append(label_index)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Carregar as imagens e rótulos
image_dir = '.\\data\\training_images\\'
images, labels = load_images(image_dir, IMG_SIZE)

# Normalizar as imagens (valores entre 0 e 1)
images = images / 255.0

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Construir o modelo RNC
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(9, activation='softmax')  # 9 saídas para MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# Avaliar o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Teste Acurácia: {test_acc}')

# Salvar o modelo treinado para usar posteriormente
model.save('melanoma_model_v5.keras')