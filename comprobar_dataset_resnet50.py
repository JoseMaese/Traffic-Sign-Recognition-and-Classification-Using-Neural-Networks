import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import random

# Configuración de los parámetros
BATCH_SIZE = 64
IMG_SIZE = (180, 180)

# Configurar variable de entorno para evitar el error de OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Definir las clases
clases = {
    0: '100', 1: '120', 2: '20', 3: '30', 4: '40', 5: '50', 6: '60', 7: '70',
    8: '80', 9: 'no_overtaking', 10: 'no_truck_overtaking', 11: 'prohibited_driving',
    12: 'truck_access_denied'
}

# Cargar el conjunto de validación
directory = "Datasets/ClassificationTrafficSigns/"
validation_dataset = image_dataset_from_directory(directory,
                                                  shuffle=True,
                                                  color_mode='rgb',
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  seed=42)

# Cargar el modelo
model = tf.keras.models.load_model("ResNet_Model_local_v2.h5")

# Convertir el dataset a una lista de numpy arrays
validation_data = list(validation_dataset.take(1).as_numpy_iterator())[0]
images, labels = validation_data

# Seleccionar 16 imágenes aleatorias del conjunto de validación
indices = random.sample(range(len(images)), 16)
random_images = images[indices]
random_labels = labels[indices]

# Hacer predicciones sobre las imágenes seleccionadas
predictions = model.predict(np.array(random_images))

# Crear una figura de 4x4
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for img, ax, pred, label in zip(random_images, axes, predictions, random_labels):
    ax.imshow(img.astype("uint8"))
    ax.axis('off')
    pred_label = np.argmax(pred)
    pred_class = clases[pred_label]
    true_class = clases[label]
    color = 'green' if pred_label == label else 'red'
    ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)

plt.tight_layout()
plt.savefig("predicciones_4x4.png")
plt.show()
