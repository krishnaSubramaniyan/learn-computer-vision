import cv2
from ultralytics import YOLO

yolo = YOLO("./models/yolo/yolo11n.pt").to('cuda')
imagePath = "./data/image/multiObject.jpg"
result = yolo.track(imagePath)[0]

def writeObjectCount(count :dict, result, img,x,y) -> None:
    for key,value in count.items():
        className = result.names[key]
        cv2.putText(img, f'{className} : {value}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,10,220), 2, cv2.LINE_AA )
        y += 30

img = cv2.imread(imagePath,cv2.IMREAD_COLOR)
color = (255,0,0)
count = dict()
for box in result.boxes:
    #object count
    classID = int(box.cls[0])
    if(count.get(classID) == None):
        count[classID] = 1
    else:
        count[classID] += 1
     
    x1,y1,x2,y2 = [int(i) for i in box.xyxy[0]]
    cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
    cv2.putText(img,f'{result.names[classID]} {box.conf[0]:.2f}',(x1,y1),cv2.FONT_HERSHEY_PLAIN,2,color,1)
    
writeObjectCount(count,result,img,20,40)
cv2.imshow("count objects",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
