'''
José E. Maese Álvarez. 
TFM: Uso de redes neuronales para identificación de señales de tráfico.
Funciones de creacion de base de datos 
'''

import os
import random
from shutil import copyfile

def split_data(input_folder, output_train, output_valid, output_test, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
    # Verifica que las proporciones sumen 1.0
    assert train_ratio + valid_ratio + test_ratio == 1.0, "Las proporciones deben sumar 1.0"

    # Obtén la lista de archivos en la carpeta de imágenes
    image_files = [f for f in os.listdir(os.path.join(input_folder, 'images')) if f.endswith('.jpg')]

    # Baraja aleatoriamente la lista de archivos
    random.shuffle(image_files)

    # Calcula las divisiones de los conjuntos de datos
    total_files = len(image_files)
    train_split = int(train_ratio * total_files)
    valid_split = int(valid_ratio * total_files)

    # Divide los archivos en conjuntos de entrenamiento, validación y prueba
    train_set = image_files[:train_split]
    valid_set = image_files[train_split:train_split + valid_split]
    test_set = image_files[train_split + valid_split:]

    # Copia los archivos a las carpetas correspondientes
    copy_files(train_set, os.path.join(input_folder, 'images'), output_train)
    copy_files(valid_set, os.path.join(input_folder, 'images'), output_valid)
    copy_files(test_set, os.path.join(input_folder, 'images'), output_test)

def copy_files(file_list, source_folder, destination_folder):
    for file in file_list:
        source_path = os.path.join(source_folder, file)
        dest_path = os.path.join(destination_folder, 'images', file)
        copyfile(source_path, dest_path)

        # También copia el archivo de etiquetas correspondiente si existe
        label_file = os.path.splitext(file)[0] + '.txt'
        source_label_path = os.path.join(input_folder, 'labels', label_file)
        dest_label_path = os.path.join(destination_folder, 'labels', label_file)
        
        if os.path.exists(source_label_path):
            copyfile(source_label_path, dest_label_path)

# Rutas de entrada y salida
input_folder = r'C:\Users\josen\Archivos TFM\crear_dataset\Datasets\Mapillary_YOLOv8_version\test'
output_train = r'C:\Users\josen\Archivos TFM\crear_dataset\Datasets\Mapillary_DS\train'
output_valid = r'C:\Users\josen\Archivos TFM\crear_dataset\Datasets\Mapillary_DS\valid'
output_test = r'C:\Users\josen\Archivos TFM\crear_dataset\Datasets\Mapillary_DS\test'

# Crea las carpetas de salida si no existen
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_valid, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

# Llama a la función para dividir los datos
split_data(input_folder, output_train, output_valid, output_test)

