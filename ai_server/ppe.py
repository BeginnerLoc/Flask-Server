from ultralytics import YOLO
import cv2


def box_label(
    employee_data, image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
):
    requirements = []
    if (
        employee_data is not None
        and employee_data["position"] == "Manager"
    ):
        requirements = ["NO-Mask"]

    if label != "Person" and label not in requirements:
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(
            image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA
        )
        if label:
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(
                label, 0, fontScale=lw / 3, thickness=tf
            )[0]
            outside = p1[1] - h >= 3
            p2 = (
                p1[0] + w,
                p1[1] - h - 3 if outside else p1[1] + h + 3,
            )
            cv2.rectangle(
                image, p1, p2, color, -1, cv2.LINE_AA
            )
            cv2.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
            
def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    if labels == []:
        labels = {
            0: "__background__",
            1: "Hardhat",
            2: "Mask",
            3: "NO-Hardhat",
            4: "NO-Mask",
            5: "NO-Safety Vest",
            6: "Person",
            7: "Safety Cone",
            8: "Safety Vest",
            9: "machinery",
            10: "vehicle",
        }
        
        colors = [
            (0, 255, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 0, 255),
            (0, 0, 255),
            (123, 63, 0),
            (123, 63, 0),
            (0, 255, 0),
            (123, 63, 0),
            (123, 63, 0),
        ]
    
    for box in boxes:
        if score:
            label = (
                labels[int(box[-1]) + 1]
                + " "
                + str(round(100 * float(box[-2]), 1))
                + "%"
            )
        else:
            label = labels[int(box[-1]) + 1]

        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
            else:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)    
