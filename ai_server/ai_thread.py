import face_recognition
import cv2
import numpy as np
from pymongo.mongo_client import MongoClient
from ultralytics import YOLO
import threading
from datetime import datetime 

class AiThread():
    def __init__(
        self,
        employee_data=None,
        uri="mongodb+srv://loctientran235:PUp2XTv7tkArDjJB@c290.5lmj4xh.mongodb.net/?retryWrites=true&w=majority",
    ):
        super().__init__()
        self.db_client = MongoClient(uri)
        self.db = self.db_client["construction"]
        self.collection = self.db["encodings"]
        self.video_capture = cv2.VideoCapture(0)
        self.stop_event = threading.Event()
        self.employee_data = employee_data

    def train_encoding(self, image_url):
        image = face_recognition.load_image_file(image_url)
        name = image_url.split(".")[0]
        encoding = face_recognition.face_encodings(image)[0]
        object_encoding = encoding.astype(object)
        result = np.insert(object_encoding, 0, name)
        return result

    def save_encodings(self, encoding):
        encoding_list = encoding.tolist()
        self.collection.insert_one({"encode": encoding_list})

    def retrieve_encoding(self):
        encoding_data = self.collection.find()
        encoding_list = []
        for encoding in encoding_data:
            num_vals = encoding["encode"][1:]
            detect_name = encoding["encode"][0]
            encode_np = np.array(num_vals)
            encoding_list.append((detect_name, encode_np))
        return encoding_list

    def update_employee(self, name):
        collection = self.db["workers"]
        self.employee_data = collection.find_one({"name": name})
        print(
            "=========================== "
            + self.employee_data["name"]
            + " is a "
            + self.employee_data["position"]
            + "================================"
        )

    def search_data_thread(self, name):
        thread = threading.Thread(target=self.update_employee, args=(name,))
        thread.start()
        return thread

    def recognition(self):
        encoding_list = self.retrieve_encoding()
        model2 = YOLO("C:\\FYP\\Flask-Server\\ai_server\\ai_model\\ppe_model.pt") 

        checkin_recorded = set() # Initializes an empty set called checkin_recorded. This set will be used to keep track of the names of employees who have already checked in to avoid duplicates.

        known_face_encodings = [encoding[1] for encoding in encoding_list]
        known_face_names = [encoding[0] for encoding in encoding_list]

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True 

        while True:
            ret, frame = self.video_capture.read()

            
            
            if process_this_frame:
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding
                    )
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                    else:
                        face_distances = face_recognition.face_distance(
                            known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                    if (
                        self.employee_data is None
                        or self.employee_data["name"] != name
                    ):
                        thread = self.search_data_thread(name)
                        self.stop_event.set()
                        thread.join()
                        self.stop_event.clear()
                    face_names.append(name)

                #Live check in:

                # Create a single document for each day to store all the worker check-ins
                    checkin_data = {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "check_ins": []
                    }

                # Check if the detected face is a known employee and insert check-in record
                    if name != "Unknown" and name not in checkin_recorded:
                        collection = self.db["workers"]
                        worker_data = collection.find_one({"name": name})
                        if worker_data is not None:
                            position = worker_data["position"]
                            worker_id = worker_data["worker_id"]
                        else:
                            position = None
                            worker_id = None 

                        collection2 = self.db["checkin"]

                        checkin_entry = {
                            "name": name,
                            "position": position,
                            "worker_id": worker_id,
                            "time": datetime.now()
                        }
                        checkin_data["check_ins"].append(checkin_entry)
                        checkin_recorded.add(name)

                    # Insert the check-in data for the day into the MongoDB collection
                    collection2.update_one({"date": checkin_data["date"]}, {"$push": {"check_ins": {"$each": checkin_data["check_ins"]}}}, upsert=True)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names): 
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if name != "Unknown":
                    results1 = model2.predict(frame, verbose=False)
                    self.plot_bboxes(
                        frame, results1[0].boxes.data, score=False, conf=0.85
                    )

                    cv2.rectangle(
                        frame, (left, top), (right, bottom), (0, 255, 0), 2
                    )
                    cv2.rectangle(
                        frame,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(
                        frame,
                        name,
                        (left + 6, bottom - 6),
                        font,
                        1.0,
                        (255, 255, 255),
                        1,
                    )
            cv2.imshow("Video", frame)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
        
    def box_label(
    self, image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
):
        requirements = []
        if (
            self.employee_data is not None
            and self.employee_data["position"] == "Manager"
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

    def plot_bboxes(self, image, boxes, labels=[], colors=[], score=True, conf=None):
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
                    self.box_label(image, box, label, color)
                else:
                    color = colors[int(box[-1])]
                    self.box_label(image, box, label, color)