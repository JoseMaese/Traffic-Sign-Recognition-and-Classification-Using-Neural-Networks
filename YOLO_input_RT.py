import cv2
from YOLO_model_RT import load_image, post_process

win_name = 'Detection'
cv2.namedWindow(win_name)

labels = open('coco.names').read().strip().split('\n')

video_reader = cv2.VideoCapture(0)
 
while True:
    _, image = video_reader.read()
    # print("Tipo:", type(image))
    boxes = load_image(image)
    # print("Flag 1")
    image = post_process(image, boxes, 0.2)
    # print("Flag 2")
        
    cv2.imshow(win_name, image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
 
cv2.destroyAllWindows()
video_reader.release()






