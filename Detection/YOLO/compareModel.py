from ultralytics import YOLO
import cv2

videoPath = "./data/video/cars.mp4"
model1 = YOLO('./models/yolo/yolov8n.pt').to('cuda')
model2 = YOLO('./models/yolo/yolo11n.pt').to('cuda')
models_name = ("yolov8n", "yolo11n")
y_points = (30, 75)

color = [(0,200,0),(0,0,200)]

cap = cv2.VideoCapture(videoPath)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        detect = [ model1(frame)[0] , model2(frame)[0]]
        for i in range(len(detect)):        
            cv2.putText(frame,models_name[i], (10,y_points[i]), cv2.FONT_HERSHEY_COMPLEX, 1, color[i], 2)
            for box in detect[i].boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), color[i], 2)

                cv2.putText(frame,f'{box.conf[0]:.2}',(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.7, (255,255,255),1)
        
        cv2.imshow("compare models", cv2.resize(frame,(1080,720),interpolation=cv2.INTER_AREA))
        if cv2.waitKey(100)&0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()