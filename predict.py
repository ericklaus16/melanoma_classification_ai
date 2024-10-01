import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf

# Função para processar a imagem e prepará-la para o modelo
def preprocess_image(image_path, img_size):
    image = cv2.imread(image_path)  # Carregar a imagem
    if image is None:
        raise ValueError(f'Não foi possível carregar a imagem: {image_path}')
    
    image = cv2.resize(image, (img_size, img_size))  # Redimensionar para o tamanho esperado
    image = image / 255.0  # Normalizar os valores da imagem (0-1)
    image = np.expand_dims(image, axis=0)  # Adicionar uma dimensão para o lote (batch size = 1)
    
    return image

# Função para prever a classe de uma imagem
def predict_image(model, image_path, img_size):
    # Processar a imagem
    image = preprocess_image(image_path, img_size)
    
    # Fazer a previsão
    predictions = model.predict(image)
    
    # Definir os rótulos das classes
    class_labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
    # Interpretar a previsão (retorna probabilidades para cada classe)
    predicted_label = class_labels[np.argmax(predictions[0])]
    predicted_probabilities = predictions[0]
    
    return predicted_label, predicted_probabilities

# Carregar o modelo treinado
model = tf.keras.models.load_model('melanoma_model_v2.keras')

# Caminho da imagem que você deseja testar
image_path = './data/ISIC_2019_Training_Input/ISIC_0000004.jpg'

# Prever a classe da imagem
predicted_label, predicted_probabilities = predict_image(model, image_path, 128)

# Exibir o resultado
print(f'Classe prevista: {predicted_label}')
print(f'Probabilidades: {predicted_probabilities}')