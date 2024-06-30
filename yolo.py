from ultralytics import YOLO

model = YOLO('Modelos/modelo002.pt')  # initialize model

results = model('Imagenes_prueba/signal.png', save=True)  # perform inference
results = model('Imagenes_prueba/signal1.jpg', save=True)  
results = model('Imagenes_prueba/signal2.jpg', save=True) 
results = model('Imagenes_prueba/signal3.jpg', save=True)  
results = model('Imagenes_prueba/signal4.jpg', save=True)  
results = model('Imagenes_prueba/signal5.jpg', save=True) 
results = model('Imagenes_prueba/signal6.jpg', save=True) 
results = model('Imagenes_prueba/signal7.jpg', save=True) 
results = model('Imagenes_prueba/signal8.png', save=True) 
results[:]