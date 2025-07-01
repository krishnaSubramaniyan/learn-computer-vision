import cv2
from ultralytics import YOLO

yolo = YOLO('./models/yolo/yolo11n.pt').to('cuda')
video = cv2.VideoCapture("./data/video/people.avi")

def getColours(cls_num):
    base_colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(base_colors[color_index])

while video.isOpened():
    isTrue,frame = video.read()
    if isTrue:
        frame = cv2.resize(frame,(1080,600))
        track = yolo(frame)[0]
        for box in track.boxes:
            classID = int(box.cls[0])
            color = getColours(classID)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
            cv2.putText(frame,f'{track.names[classID]} {box.conf[0]:.2f}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        
        cv2.imshow("object detection video",frame)
        if cv2.waitKey(24)&0xFF == ord('q'):
            break
    else:
        video.release()
        cv2.destroyAllWindows()

