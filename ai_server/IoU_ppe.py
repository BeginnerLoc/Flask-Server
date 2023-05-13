# box1 = [2 , 3 , 12 , 15] #ground truth
# box2 = [8 , 10 , 20 , 23] #predicted

# #intersection

# xl = max(box1[0] , box2[0])
# yl = max(box1[1] , box2[1])
# xr = min(box1[2] , box2[2])
# yr = min(box1[3] , box2[3])

# area_intersection = (xr - xl) * (yr - yl)

# #union

# width_box1 = box1[2] - box1[0]
# height_box1 = box1[3] - box1[1]
# width_box2 = box2[2] - box2[0]
# height_box2 = box2[3] - box2[1]

# area_box1 = width_box1 * height_box1
# area_box2 = width_box2 * height_box2

# union_area = area_box1 + area_box2 - area_intersection

# iou = area_intersection / union_area

# print(iou)


from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.yolo.utils.plotting import Annotator

# model1 = YOLO('ai_model\hole.pt')
# model1.conf = 0.7

model2 = YOLO('ai_model\ppe_model.pt')
# model2.conf = 0.7

cap = cv2.VideoCapture(0)

#Make the resolutions to 480p
# cap.set(3, 640)
# cap.set(4, 480)
cap.set(3, 1920)
cap.set(4, 1080)

#play the bounding boxes with the label and the score :
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    #Define COCO Labels
    if labels == []:
        labels = {0: u'__background__', 1: u'Hardhat', 2: u'Mask', 3: u'NO-Hardhat', 
                  4: u'NO-Mask', 5: u'NO-Safety Vest', 6: u'Person', 
                  7: u'Safety Cone', 8: u'Safety Vest', 9: u'machinery', 10: u'vehicle'}
# 'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
# 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
    #Define colors
        colors = [(0,255,0),(0,255,0),(0,0,255),
                  (0,0,255),(0,0,255),(123,63,0),
                  (123,63,0),(0,255,0),(123,63,0),(123,63,0)]

    #plot each boxes
    for box in boxes:
        #add score in label if score=True
        if score :
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else :
            label = labels[int(box[-1])+1]
        #filter every box under conf threshold if conf threshold setted
        if conf :
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
            else:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)

while True:
    _, frame = cap.read()

    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = np.asarray(frame)

    results1 = model2.predict(image)

    #annotator = Annotator(frame)

    plot_bboxes(image, results1[0].boxes.data, score=False, conf=0.85)


    #frame = annotator.result()
    cv2.imshow('YOLO V8 Detection', image)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()