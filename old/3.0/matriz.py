import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
IMG_SIZE = 128  # Tamanho das imagens (128x128 pixels)

# Carregar a planilha
labels_df = pd.read_csv('./data/ISIC_2019_Training_GroundTruth.csv')

# Função para carregar e processar imagens
def load_images(image_dir, labels_df, img_size):
    images = []
    labels = []
    for _, row in labels_df.iterrows():
        img_path = os.path.join(image_dir, row['image'] + '.jpg')  # Adicionando extensão da imagem
        image = cv2.imread(img_path)
        if image is not None:
            # Redimensionar a imagem
            image = cv2.resize(image, (img_size, img_size))
            images.append(image)
            # Adicionando os rótulos (a partir da segunda coluna, que contém os valores MEL, NV, etc.)
            labels.append(row[1:].values.astype(float))  # Acessar diretamente os valores como float
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Carregar as imagens e rótulos
image_dir = './data/ISIC_2019_Training_Input'
images, labels = load_images(image_dir, labels_df, IMG_SIZE)

# Normalizar as imagens (valores entre 0 e 1)
images = images / 255.0

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Carregar o modelo treinado
model = load_model('melanoma_model_v5.keras')

# Fazer previsões no conjunto de teste
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Aplicando um threshold de 0.5

# Se o seu conjunto de dados tem múltiplas classes, você pode usar np.argmax
# para obter as classes de uma matriz de saídas
y_test_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Visualizar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'], 
            yticklabels=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])
plt.xlabel('Classes Previstas')
plt.ylabel('Classes Reais')
plt.title('Matriz de Confusão')
plt.show()

# Exibir o relatório de classificação
print(classification_report(y_test_classes, y_pred_classes, target_names=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']))
