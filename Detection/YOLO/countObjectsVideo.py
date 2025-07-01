import cv2
from ultralytics import YOLO


def getColor(cls_num):
    if cls_num == 0:
        return (200,0,0)
    else:
        return (0,0,200)

def printObjectCount(objectCount,img,x,y,color) -> None:
    for key,value in objectCount.items():
        cv2.putText(img,f'{track.names[key]} : {value}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        y += 30

videoPath = './data/video/people.avi'
yolo = YOLO('./models/yolo/yolo11n.pt').to('cuda')
video = cv2.VideoCapture(videoPath)
objectCount = {0:0, 2:0}  # classID:objectCount

while video.isOpened():
    isTrue,frame = video.read()
    if not(isTrue):
        break
    frame = cv2.resize(frame,(1080,600))
    
    track = yolo(frame,classes=[0,2])[0]
    for box in track.boxes:
        classID = int(box.cls[0])
        objectCount[classID] += 1
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        color = getColor(classID)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
        cv2.putText(frame,f'{track.names[classID]} {box.conf[0]:.2f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.90, color, 1)

    printObjectCount(objectCount,frame,30,30,(0,0,220))
    cv2.imshow("objects count",frame)
    
    for key in objectCount.keys():
        objectCount[key] = 0
    
    if(cv2.waitKey(20)&0xFF == ord('q')):
        break

video.release()
cv2.destroyAllWindows()