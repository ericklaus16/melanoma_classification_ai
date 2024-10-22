import os
import shutil
import random

# Diretório original onde as imagens estão organizadas por classe
origem = './data/sorted_images'

# Diretórios de destino para treino, validação e teste
dest_train = './data/training_images'
dest_val = './data/validation_images'
dest_test = './data/test_images'

# Cria os diretórios principais se não existirem
os.makedirs(dest_train, exist_ok=True)
os.makedirs(dest_val, exist_ok=True)
os.makedirs(dest_test, exist_ok=True)

# Lista das classes e suas respectivas quantidades
classes = {
    'MEL': 4522,
    'NV': 12875,
    'BCC': 3323,
    'AK': 867,
    'BKL': 2624,
    'DF': 239,
    'VASC': 253,
    'SCC': 628,
    'UNK': 0  # Se necessário, pode-se ignorar essa classe
}

# Proporções para treino, validação e teste
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Percorre cada classe e faz a divisão proporcional
for classe, total_images in classes.items():
    if total_images == 0:
        continue  # Pula a classe se não houver imagens

    # Cria subdiretórios para cada classe em treino, validação e teste
    os.makedirs(os.path.join(dest_train, classe), exist_ok=True)
    os.makedirs(os.path.join(dest_val, classe), exist_ok=True)
    os.makedirs(os.path.join(dest_test, classe), exist_ok=True)

    # Lista todas as imagens da classe
    imagens = os.listdir(os.path.join(origem, classe))
    
    # Embaralha as imagens para garantir uma distribuição aleatória
    random.shuffle(imagens)

    # Calcula quantidades para cada conjunto
    train_limit = int(train_ratio * total_images)
    val_limit = int(val_ratio * total_images)
    test_limit = total_images - train_limit - val_limit  # O que restar vai para teste

    # Move as imagens para os diretórios correspondentes
    for i, imagem in enumerate(imagens):
        src_path = os.path.join(origem, classe, imagem)
        
        if i < train_limit:
            shutil.move(src_path, os.path.join(dest_train, classe, imagem))
        elif i < train_limit + val_limit:
            shutil.move(src_path, os.path.join(dest_val, classe, imagem))
        else:
            shutil.move(src_path, os.path.join(dest_test, classe, imagem))

print('Divisão proporcional concluída!')
