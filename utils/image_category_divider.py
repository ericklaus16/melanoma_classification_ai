import csv
import os
import shutil

# Inicializando contadores para cada classe
counts = {
    'MEL': 0,
    'NV': 0,
    'BCC': 0,
    'AK': 0,
    'BKL': 0,
    'DF': 0,
    'VASC': 0,
    'SCC': 0,
    'UNK': 0
}

input_dir = './data/ISIC_2019_Training_Input/'
output_dir = './data/sorted_images/'


os.makedirs(output_dir, exist_ok=True)
for classe in counts.keys():
    os.makedirs(os.path.join(output_dir, classe), exist_ok=True)

# Lendo o arquivo CSV
with open('./data/ISIC_2019_Training_GroundTruth.csv', 'r') as file:
    reader = csv.reader(file)
    
    # Ignorando o cabeçalho
    next(reader)
    
    # Processando cada linha
    for row in reader:
        image_name = row[0]
        # Convertendo os valores de string para float
        values = list(map(float, row[1:]))
        
        # Atualizando contadores
        if values[0] == 1.0:
            counts['MEL'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'MEL', image_name + '.jpg'))
        if values[1] == 1.0:
            counts['NV'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'NV', image_name + '.jpg'))
        if values[2] == 1.0:
            counts['BCC'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'BCC', image_name + '.jpg'))
        if values[3] == 1.0:
            counts['AK'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'AK', image_name + '.jpg'))
        if values[4] == 1.0:
            counts['BKL'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'BKL', image_name + '.jpg'))
        if values[5] == 1.0:
            counts['DF'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'DF', image_name + '.jpg'))
        if values[6] == 1.0:
            counts['VASC'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'VASC', image_name + '.jpg'))
        if values[7] == 1.0:
            counts['SCC'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'SCC', image_name + '.jpg'))
        if values[8] == 1.0:
            counts['UNK'] += 1
            shutil.move(os.path.join(input_dir, image_name + ".jpg"), os.path.join(output_dir, 'UNK', image_name + '.jpg'))

# Exibindo os resultados
for classe, count in counts.items():
    print(f'{classe}: {count}')
