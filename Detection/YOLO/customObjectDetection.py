from ultralytics import YOLO
import cv2

model = YOLO('./models/custom/licensePlate_yolo11.pt').to('cuda')
video = cv2.VideoCapture('./data/video/cars.mp4')

while video.isOpened():
    ret,frame = video.read()
    if ret:
        detect = model(frame)[0]
        for box in detect.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,210,0), 2)

        cv2.imshow("detect licensePlate",frame)
        if cv2.waitKey(24)&0xFF == ord('q'):
            break
    else:
        video.release()
        cv2.destroyAllWindows()


        
