# YOLO object detection
import cv2 as cv
import numpy as np
import time

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None

# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
#net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
#net = cv.dnn.readNetFromDarknet('yolov2-tiny.cfg', 'yolov2-tiny.weights')
#net = cv.dnn.readNetFromDarknet('yolov3-tiny.cfg', 'yolov3-tiny.weights')

net = cv.dnn.readNetFromDarknet('YOLO\yolov7.cfg', 'YOLO\yolov7-tiny.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def load_image(video_frame):
    # global img, img0, outputs, ln

    img0 = video_frame
    img0 = cv.resize(img0, (900,600), interpolation=cv.INTER_AREA)
    img = img0.copy()
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    #############
    # r = blob[0, 0, :, :]

    # cv.imshow('blob', r)
    # text = f'Blob shape={blob.shape}'
    # cv.displayOverlay('blob', text)
    # cv.waitKey(1)
    #############
    
    net.setInput(blob)

    outputs = net.forward(ln)

    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    outputs = np.vstack(outputs)

    # post_process(img, outputs, 0.5)
    return outputs
    # cv.imshow('window',  img)
    # cv.displayOverlay('window', f'forward propagation time={t:.3}')
    # cv.waitKey(0)

def post_process(img, outputs, conf):
    
    H, W = img.shape[:2]
    
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

# def trackbar(x):
#     global img
#     conf = x/100
#     img = img0
    
#     post_process(img, outputs, conf)
    # cv.displayOverlay('window', f'confidence level={conf}')
    # cv.imshow('window', img)

# cv.namedWindow('window')
# cv.createTrackbar('confidence', 'window', 50, 101, trackbar)

# load_image('images/horse.jpg')
# load_image('images/food.jpg')
# load_image('images/autopista.jpg')
# load_image('images/cows.jpg')
# load_image('images/zafari.png')
# load_image('images/airport.jpg')
# load_image('images/tennis.jpg')
# load_image('images/wine.jpg')
# load_image('images/bicycle.jpg')

cv.destroyAllWindows()