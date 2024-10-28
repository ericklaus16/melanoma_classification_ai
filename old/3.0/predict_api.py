import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Carregar o modelo treinado de detecção de melanoma
model = tf.keras.models.load_model('60%-accuracy.keras')

# Função para processar a imagem e prepará-la para o modelo
def preprocess_image(image_path, img_size=256):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Não foi possível carregar a imagem: {image_path}')
    
    # Pré-processamento: redimensionar e normalizar
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    return image

# Função para prever a classe de uma imagem
def predict_image(model, image_path, img_size=256):
    # Processar a imagem
    image = preprocess_image(image_path, img_size)
    
    # Fazer a previsão
    predictions = model.predict(image)
    
    # Definir os rótulos das classes (verifique se são as mesmas do treinamento)
    class_labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    
    # Interpretar a previsão
    predicted_label = class_labels[np.argmax(predictions[0])]
    predicted_probabilities = predictions[0]
    
    return predicted_label, predicted_probabilities

@app.get('/test')
def test():
    print("Requisição recebida")
    return jsonify({"msg": "Sucesso"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    print("Requisição recebida")
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo vazio'}), 400
    
    # Salvar a imagem temporariamente
    image_path = './temp_image.jpg'
    file.save(image_path)
    
    try:
        # Fazer a previsão
        predicted_label, predicted_probabilities = predict_image(model, image_path)
        
        # Retornar a resposta
        return jsonify({
            'predicted_label': predicted_label,
            'predicted_probabilities': predicted_probabilities.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Remover a imagem temporária
        os.remove(image_path)

if __name__ == '__main__':
    app.run(debug=True)
