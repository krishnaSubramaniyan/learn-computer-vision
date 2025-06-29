import cv2
from ultralytics import YOLO

yolo = YOLO("./models/yolo/yolo11n.pt").to('cuda')
imagePath = "./data/image/car.jpg"
img = cv2.imread(imagePath,cv2.IMREAD_COLOR)

result = yolo.track(imagePath,classes=[2])
result = result[0]
color = (200,50,80)

for box in result.boxes:
    x1,y1,x2,y2 = [int(i) for i in box.xyxy[0]]
    cv2.rectangle(img,(x1,y1),(x2,y2), color,2)
    cv2.putText(img,f'{result.names[int(box.cls[0])]} {box.conf[0]:.2f}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

cv2.imshow("detect object",img)
cv2.waitKey(0)
cv2.destroyAllWindows()