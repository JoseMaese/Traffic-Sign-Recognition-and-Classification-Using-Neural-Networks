import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Cargar el modelo YOLOv8
model = YOLO("Modelos/modelo004.pt")
class_names = ['Danger', 'Mandatory', 'Other', 'Prohibitory']

# Definir colores para cada clase
colors = {
    'Danger': (18, 156, 243),  # Azul claro
    'Mandatory': (219, 152, 52),  # Naranja
    'Other': (60, 76, 231),  # Azul oscuro
    'Prohibitory': (42, 32, 23)  # Marrón oscuro
}

# Cargar las imágenes del dataset
dataset_dir = "Datasets/YOLOv7_signal_detection/test/images"
image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

# Seleccionar 4 imágenes aleatorias
selected_images = random.sample(image_files, 4)

# Función para dibujar los bounding boxes y etiquetas en la imagen
def draw_boxes(image, boxes, classes, scores, class_names):
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(cls)]
        label = f"{class_name}: {score:.2f}"
        color = colors[class_name]
        # Dibujar el rectángulo del texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        # Escribir el texto en blanco
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Dibujar el bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

# Crear una figura de 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for img_path, ax in zip(selected_images, axes):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Realizar la predicción
    results = model(img_rgb)
    
    # Obtener las predicciones
    boxes = results[0].boxes.xyxy.numpy()
    classes = results[0].boxes.cls.numpy()
    scores = results[0].boxes.conf.numpy()
    
    # Dibujar las predicciones
    img_with_boxes = draw_boxes(img_rgb, boxes, classes, scores, class_names)
    
    ax.imshow(img_with_boxes)
    ax.axis('off')

plt.tight_layout()
plt.savefig("predicciones_yolo_2x2.png")
plt.show()
