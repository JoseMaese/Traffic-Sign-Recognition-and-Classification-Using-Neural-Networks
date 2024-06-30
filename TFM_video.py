'''
José E. Maese Álvarez. 
TFG: Uso de redes neuronales para identificación de matrículas.
Codigo completo de deteccion e identificacion de señales de trafico a partir de video.
'''

import cv2
from ultralytics import YOLO
import os
import numpy as np
from tensorflow.keras.models import load_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the YOLOv8 model
model = YOLO("Modelos/modelo004.pt")
class_names = open('signal.names').read().strip().split('\n')

# Load ResNet50 model
modelo_resnet = load_model("Modelos/Clasificador/ResNet_Model_local.h5")


# Open the video file
cap = cv2.VideoCapture("Imagenes_prueba/video_prueba_2.mp4")

# Get video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Obtener la resolución original
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir la nueva resolución deseada (por ejemplo, la mitad de la resolución original)
new_width = original_width // 1
new_height = original_height // 1

# Get video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/output_video_2_con_clasificador_v1.mp4', fourcc, fps, (new_width, new_height))

# Dibujamos los bounding boxes correspondientes
def dibuja_frame(frame, box, class_name, conf):  
    color = (96, 174, 39)   # BGR
    conf = "{:.3f}".format(conf)
    if id is not None:
        if class_name == 0:
            clase = 'Danger'
            color = (18, 156, 243)

        if class_name == 1:
            clase = 'Mandatory'
            color = (219, 152, 52)

        if class_name == 2:
            # clase = 'Other'
            clase = analisis_velocidad(frame, box)
            color = (60, 76, 231)

        if class_name == 3:
            clase = 'Prohibitory'
            color = (42, 32, 23)

        text = f"ID {id}: {clase} {conf}"
        
        # Obtener dimensiones del texto
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # Dibujar fondo
        cv2.rectangle(frame, (box[0], box[1] - 4 - text_height), (box[0] + text_width, box[1]), color, cv2.FILLED)
        # Escribir texto en blanco
        cv2.putText(frame, text, (box[0], box[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Dibujar recuadro
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)   
    
    
    
    
    
def analisis_velocidad(signal, box):
    tam = 180
    # Aumentar el tamaño de la región de interés (ROI) en un 10% en cada dirección
    width_increase = int((box[2] - box[0]) * 0.1)
    height_increase = int((box[3] - box[1]) * 0.1)

    # Definir los nuevos límites del cuadro delimitador
    new_box = [max(0, box[0] - width_increase),
           max(0, box[1] - height_increase),
           min(signal.shape[1], box[2] + width_increase),
           min(signal.shape[0], box[3] + height_increase)]

    # Región de interés con el nuevo tamaño
    roi = signal[new_box[1]:new_box[3], new_box[0]:new_box[2]]
    
    # Region of interest
    # roi = signal[box[1]:box[3], box[0]:box[2]]
    
    # Preprocesa la imagen recortada para que sea compatible con la entrada de la red ResNet
    roi_resized = cv2.resize(roi, (tam, tam))
    np_img = np.array(roi_resized)
    signal_preprocessed = np.reshape(np_img, [-1, tam, tam, 3])
    
    # Utiliza el modelo ResNet para predecir la clase de la señal de tráfico
    resultado = modelo_resnet.predict(signal_preprocessed)
    
    # Reducir la importancia de las últimas 4 clases
    resultado[0][-4:] *= 0.3
    
    class_index = resultado.argmax(axis=1)
    
    clases = {
       0: '100', 1: '120', 2: '20', 3: '30', 4: '40', 5: '50', 6: '60', 7: '70',
       8: '80', 9: 'no_overtaking', 10: 'no_truck_overtaking', 11: 'prohibited_driving',
       12: 'truck_access_denied'
   }
    
    # Devuelve la clase predicha
    return clases[class_index[0]]

    
# Loop through the video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If frame is read correctly, proceed
    if ret:
        
        frame = cv2.resize(frame, (new_width, new_height))
        results = model.track(frame, persist=True)

        # Check if there are any detections
        if results[0].boxes is not None:
            # Extract IDs if they exist
            ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
            class_index = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes.cls is not None else []
            confianzas = results[0].boxes.conf.cpu().numpy().astype(float) if results[0].boxes.conf is not None else []

            # Annotate frame with boxes and IDs
            for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy().astype(int)):
                # Asigna el ID correspondiente a la detección actual, si existen IDs disponibles. 
                # Si no hay IDs disponibles, id se establece en None.
                id = ids[i] if len(ids) > 0 else None
                # Obtenemos la clase predicha para esta detección y su confianza
                class_name = class_index[i] if len(class_index) > 0 else None
                conf = confianzas[i] if len(confianzas) > 0 else None
                
                dibuja_frame(frame, box, class_name, conf)
                        
        
        out.write(frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached or frame is not read correctly
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()