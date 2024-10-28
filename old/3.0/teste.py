import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ocultando mensagens de aviso do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras_tuner import RandomSearch
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Carregando imagens do diretório
data = tf.keras.utils.image_dataset_from_directory('./data/sorted_images')

# Normalizando as imagens
data = data.map(lambda x, y: (x / 255.0, y))

# Definindo tamanhos para treino, validação e teste
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Convertendo os rótulos para categóricos
train = train.map(lambda x, y: (x, to_categorical(y, num_classes=8)))
val = val.map(lambda x, y: (x, to_categorical(y, num_classes=8)))

# Função para criar o modelo
def create_model(hp):
    model = Sequential()
    
    # Otimização do número de filtros na primeira camada
    model.add(Conv2D(hp.Int('filters1', min_value=16, max_value=64, step=16), (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    
    # Otimização do número de filtros na segunda camada
    model.add(Conv2D(hp.Int('filters2', min_value=32, max_value=128, step=32), (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    # Otimização do número de unidades na camada densa
    model.add(Dense(hp.Int('units', min_value=128, max_value=512, step=128), activation='relu'))
    
    # Otimização da taxa de dropout
    model.add(Dropout(hp.Float('dropout_rate', 0, 0.5, step=0.1)))
    
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Inicializando o Keras Tuner
tuner = RandomSearch(
    create_model,
    objective='val_accuracy',
    max_trials=10,  # Número máximo de combinações de hiperparâmetros a serem testadas
    executions_per_trial=1,
    directory='my_dir',
    project_name='cancer_model_tuning'
)

# Executando a busca
tuner.search(train, validation_data=val, epochs=10)

# Obtendo os melhores hiperparâmetros
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Melhores hiperparâmetros: {best_hyperparameters.values}")

# Criando e treinando o modelo com os melhores hiperparâmetros
model = tuner.hypermodel.build(best_hyperparameters)
history = model.fit(train, validation_data=val, epochs=10)
