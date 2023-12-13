import cv2
from ultralytics import YOLO


coco_model = YOLO('yolov8n.pt')
coco_model.confidence_threshold = 0.75

classLabels=[]
file_name='Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

cap = cv2.VideoCapture(0)
Barang=[0, 24, 26 , 28]
while True:
    ret, frame = cap.read()
    detections = coco_model(frame)[0]

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in Barang:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # label = f"{coco_model.names[int(class_id)]}: {score:.2f}"
                label=classLabels[int(class_id)]
                cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print(label)


        
  
    cv2.imshow('coba',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()