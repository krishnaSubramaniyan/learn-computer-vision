from ultralytics import YOLO
import cv2

def getColours(cls_num):
    base_colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(base_colors[color_index])

yolo = YOLO("./models/yolo/yolo11n.pt").to('cuda')
img = cv2.imread("./data/image/multiObject.jpg")
result = yolo(img)[0]

for box in result.boxes:
    cls = int(box.cls[0])
    className = result.names[cls]
    [x1,y1,x2,y2]  = box.xyxy[0]
    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    color =  getColours(cls)
    cv2.rectangle(img,(x1,y1), (x2,y2), color, 2)
    cv2.putText(img,f'{className} {box.conf[0]:.2f}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
    print("box = ",box)
    print("object = ",className)

cv2.imshow("Multi Objects Detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()