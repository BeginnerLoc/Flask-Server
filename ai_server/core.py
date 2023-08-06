import face_recognition
import cv2
import numpy as np
from pymongo.mongo_client import MongoClient
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import collections
from collections import Counter
import os
import asyncio
from telegram import Bot
import httpx
import base64

async def send_message(bot_token, chat_id, name, breach, image, location, breach_id):
    base64_image = base64.b64encode(image).decode('utf-8')  # Decode base64 to string
    
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=5.0)) as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                    data={"chat_id": chat_id},
                    files={"photo": ("image.jpg", image, "image/jpeg")},
                    params = {
                        "caption": (
                            f"Breach number:{breach_id} \n\n"
                            f"{name} was not wearing a {breach}\n"
                            f"Location of breach: {location}\n"
                            f"Time of breach: {datetime.now()}"
                        )
                    }
                )
                print(f"Response Status Code: {response.status_code}")
                break  # Break the loop if the request succeeds
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
            print(f"Failed attempt {attempt}/{retries}: {exc}")
            if attempt == retries:
                print("Request failed after maximum retries")
                break
            else:
                print("Retrying...") 
                continue 

#Telegram Bot token and Chat ID (Astro's Chat ID)
bot_token = '6060060457:AAGRyic-1HVFcUy1dSEsdLMJo0rB9Mvz0y0'
chat_id  = '443723632'

uri = "mongodb+srv://loctientran235:PUp2XTv7tkArDjJB@c290.5lmj4xh.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
db_client = MongoClient(uri)
db = db_client["construction"]
collection = db["encodings_test3"]
collection2 = db["db_breaches_2"]

# Send a ping to confirm a successful connection
try:
    db_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

stop_event = threading.Event() 

employee_data = None

ai_model = YOLO('ai_model\\ppe_model.pt')
# ai_model = YOLO('D:/Workspace/Flask-Server/ai_server/ai_model/ppe_model.pt')

photo_path = "UI_photos\\"
# photo_path = 'D:/Workspace/Flask-Server/ai_server/UI_photos/'

imgBackground = cv2.imread(photo_path + 'background.png')
model = cv2.imread(photo_path + 'pageA.png')
clear_text = cv2.imread(photo_path + 'clear.png')
clear_text2 = cv2.imread(photo_path + 'clear2.png')

#avaList
folderAvaPath = photo_path + 'Ava'
avaPathList = os.listdir(folderAvaPath)
imgAvaList = []
for path in avaPathList:
    imgAvaList.append(cv2.imread(os.path.join(folderAvaPath, path)))

#ppeList
folderPpePath = photo_path + 'ppe'
ppePathList = os.listdir(folderPpePath)
imgPpeList = []
for path in ppePathList:
    imgPpeList.append(cv2.imread(os.path.join(folderPpePath, path)))

def train_encoding(image_url):
    image = face_recognition.load_image_file(image_url)
    name = image_url.split(".")[0]
    encoding = face_recognition.face_encodings(image)[0]
    object_encoding = encoding.astype(object)
    result = np.insert(object_encoding, 0, name)
    return result

def save_encodings(encoding):
    # convert the array to BSON
    encoding_list = encoding.tolist()
    collection.insert_one({"encode": encoding_list})

def retrieve_encoding():
    # retrieve all the documents from the collection
    encoding_data = collection.find()

    # create a list to store the encodings
    encoding_list = []
    for encoding in encoding_data:
        num_vals = encoding['encode'][1:]
        detect_name = encoding['encode'][0]
        encode_np = np.array(num_vals)
        encoding_list.append((detect_name, encode_np))
        
    return encoding_list   

def update_employee(name):
    collection = db["workers"]
    global employee_data
    employee_data = collection.find_one({"name": name})
    print(employee_data)
       
def search_data_thread(name):
    thread = threading.Thread(target=update_employee, args=(name,))
    thread.start()
    return thread

def alert_process(breach_ppe, most_frequent_name, worker_breaches):
    alert_message = "[ALERT]\nWorker Name:" + most_frequent_name + " is not wearing the proper PPE!\nTimestamp: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    #asyncio.run(send_message(bot_token, chat_id, alert_message))
    
    # Check the length of the list before appending
    initial_length = len(worker_breaches)

    new_breach = {"Workername": most_frequent_name, "Breach": breach_ppe}

    # # Check if the new breach already exists for the same workername with any breach
    # if any(breach["Workername"] == new_breach["Workername"] for breach in worker_breaches):
    #     existing_breaches = {breach["Breach"] for breach in worker_breaches if breach["Workername"] == new_breach["Workername"]}
    #     print(existing_breaches)
        
    #     # Check if the new breach is a duplicate for the same workername
    #     if not new_breach["Breach"] in existing_breaches:
    #         worker_breaches.append(new_breach)
    # else:
    #     worker_breaches.append(new_breach)

    # Check if workername already exists
    if not any(breach["Workername"] == new_breach["Workername"] for breach in worker_breaches):
        worker_breaches.append(new_breach)

    # Check the length of the list after appending
    updated_length = len(worker_breaches)

    # Check if a new entry has been added
    if updated_length > initial_length:
        print("New entry has been added to the list.")
        print(worker_breaches[-1])

        #Capture the frame with the plotting boxes for breach images
        frame_capture = imgBackground[158:158 + 480, 52:52 + 640]

        # Capture the frame as an image
        _, buffer = cv2.imencode(".jpg", frame_capture)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        # Find the document with the highest breach ID
        largest_breach = collection2.find_one(sort=[("breach_id", -1)])

        if largest_breach is None:
            next_breach_id = 1
        else:
            # Determine the next breach ID
            next_breach_id = largest_breach["breach_id"] + 1

        Location = "Entrance A"

        worker_breach_name = worker_breaches[-1]["Workername"]
        worker_breach_description = worker_breaches[-1]["Breach"]

        # Save the encoded image in MongoDB
        collection2.insert_one({
            "datetime": datetime.now(),
            "worker_name": worker_breach_name,
            "description": worker_breach_description,
            "breach_id": next_breach_id,
            "severity": "",
            "evidence_photo": encoded_image,
            "location": Location,
            "case_resolved": False,
            "case_resolution": None,
            "case_resolved_time": None
        })

        description = worker_breach_description.split(",")
        description_array = [item.strip().replace("no-", "") for item in description]
        tele_description = ", ".join(description_array)

        capture_image = frame_capture.copy()
        retval, buffer = cv2.imencode('.jpg', capture_image)
        image_bytes = buffer.tobytes()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(send_message(bot_token, chat_id, worker_breach_name, tele_description, image_bytes, Location, next_breach_id))

def plot_bboxes(draw_box_ppe, image, boxes, labels=[], colors=[], score=True, conf=None):
    output = []
    
    # #Define COCO Labels
    # if labels == []:
    #     labels = {0: u'__background__', 1: u'helmet', 2: u'no-helmet', 3: u'no-vest', 4: u'vest'}
    # #Define colors
    #     colors = [(0,255,0),(0,0,255),(0,0,255),(0,255,0)]

    #Define COCO Labels
    if labels == []:
        labels = {0: u'__background__', 1: u'helmet', 2: u'mask', 3: u'no-helmet', 
                  4: u'no-mask', 5: u'no-vest', 6: u'Person', 
                  7: u'Safety Cone', 8: u'vest', 9: u'machinery', 10: u'vehicle'}
    # 'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
    # 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
    #Define colors
        colors = [(0,255,0),(0,255,0),(0,0,255),
                  (0,0,255),(0,0,255),(123,63,0),
                  (123,63,0),(0,255,0),(123,63,0),(123,63,0)]

    label_list = []
    exclude_labels = ['Person', 'machinery', 'vehicle', 'Safety Cone']

    if(employee_data is not None):
        role = employee_data["position"]
        if role == "Supervisor":
            exclude_labels = ['Person', 'machinery', 'vehicle', 'Safety Cone', 'no-mask']

    #plot each boxes
    for box in boxes:
        #add score in label if score=True
        if score :
            label = labels[int([-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else :
            label = labels[int(box[-1])+1]
        #filter every box under conf threshold if conf threshold setted
        if conf :
            if box[-2] > conf:
                color = colors[int(box[-1])]
            else:
                color = colors[int(box[-1])]
            
            #box label
            if draw_box_ppe:
                if label not in label_list and label not in exclude_labels:
                    label_list.append(label)

                    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                    tf = max(lw - 1, 1)  # font thickness
                    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                    outside = p1[1] - h >= 3
                    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image,
                                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                0,
                                lw / 3,
                                color=(255, 255, 255),
                                thickness=tf,
                                lineType=cv2.LINE_AA)
                    output.append(label)
    return output

def facial_recognition(frame, draw_box_face, known_face_encodings, known_face_names, face_names, checkin_recorded):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        else:
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
        face_names.append(name)

        #Live check in:

        # Create a single document for each day to store all the worker check-ins
        checkin_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "check_ins": []
        }

        # Check if the detected face is a known employee and insert check-in record
        if name != "Unknown" and name not in checkin_recorded:
            collection = db["workers"]
            worker_data = collection.find_one({"name": name})
            if worker_data is not None:
                position = worker_data["position"]
                worker_id = worker_data["worker_id"]
                supervisor = worker_data["supervisor"]
            else:
                position = None
                worker_id = None

            # collection = db["checkin_1"]
            collection = db["checkin_2"] 


            print("=====================================================================================")
            checkin_entry = {
                "name": name,
                "worker_id": worker_id,
                "position": position,
                "supervisor": supervisor,
                "time": datetime.now()
            }
            checkin_data["check_ins"].append(checkin_entry)
            checkin_recorded.add(name)

            # Insert the check-in data for the day into the MongoDB collection
            collection.update_one({"date": checkin_data["date"]}, 
                                    {"$push": {"check_ins": {"$each": checkin_data["check_ins"]}}}, 
                                    upsert=True)
            
    #Draw Box for Face
    if draw_box_face:
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            if name != "Unknown":
                cv2.rectangle(imgBackground[158:158 + 480, 52:52 + 640], (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(imgBackground[158:158 + 480, 52:52 + 640], (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgBackground[158:158 + 480, 52:52 + 640], name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

def main():
    encoding_list = retrieve_encoding()

    checkin_recorded = set() # Initializes an empty set called checkin_recorded. This set will be used to keep track of the names of employees who have already checked in to avoid duplicates.
    
    # Create a worker_breaches_list to keep track of workers with PPE breaches
    worker_breaches = []

    item_number = 0
    counter = 0

    known_face_encodings = [encoding[1] for encoding in encoding_list]
    known_face_names = [encoding[0] for encoding in encoding_list]
    
    face_names = []
    ppe_list = []
    draw_box_face = True
    draw_box_ppe = False

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read() 

        imgBackground[158:158 + 480, 52:52 + 640] = frame
        imgBackground[30:30 + 674, 800:800 + 440] = model
        ROI = imgBackground[158:158 + 480, 52 + 160 :52 + 480]

        facial_recognition(frame, draw_box_face, known_face_encodings, known_face_names, face_names, checkin_recorded)

        cv2.putText(imgBackground, "Authenticating...", (910, 655), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        cv2.imshow('SAFETY CONSTRUCTION SYSTEM', imgBackground)

        results = ai_model.predict(ROI, verbose=False)
        #PPE_list - 3 Dimensions Array
        ppe_item = plot_bboxes(draw_box_ppe, ROI, results[0].boxes.data, score=False, conf=0.85)

        ppe_list.append(ppe_item)
        #print(ppe_list)
        print(ppe_item)
        print(face_names[-10:])

        #Find the most face detect
        if len(face_names) > 10:
            draw_box_face = False
            draw_box_ppe = True
            #face_names = face_names[-15:]
            
            most_frequent_name = Counter(face_names[:10]).most_common(1)[0][0]
            print("Session Holder: " + most_frequent_name)

            if most_frequent_name != "Unknown" and most_frequent_name != None:
                if employee_data == None or employee_data['name'] != most_frequent_name:      
                    thread = search_data_thread(most_frequent_name)
                    # Set the stop_event object to stop the thread
                    stop_event.set()
                    # Wait for the thread to finish
                    thread.join()
                    # Reset the stop_event object for the next iteration
                    stop_event.clear()

                    role = employee_data["position"]
                
                cv2.rectangle(imgBackground, (52 + 160, 158), (52 + 480, 640), (0,255,0), 1, cv2.LINE_AA) 
                cv2.putText(imgBackground, "Hi, " + most_frequent_name, (875, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(imgBackground, "PPE Require:", (830, 290), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(imgBackground, "Your Role is:", (835, 210), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

                if most_frequent_name == "Astro":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[0]
                    cv2.putText(imgBackground, role, (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                    
                elif most_frequent_name == "Chris":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[1]
                    cv2.putText(imgBackground, role, (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                    
                elif most_frequent_name == "Daren":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[2]
                    cv2.putText(imgBackground, role, (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

                elif most_frequent_name == "Loc":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[3]
                    cv2.putText(imgBackground, role, (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

                ppe_helmet = None
                ppe_vest = None
                ppe_mask = None

                if len(ppe_list) > 10:
                    ppe_list = ppe_list[-10:]
                    #if PPE existing more than 5 frames is True, less than 2 frames is as False
                    if role != "Supervisor":
                        if sum(sublist.count("no-helmet") for sublist in ppe_list) <= 3:
                            if sum(sublist.count("helmet") for sublist in ppe_list) >= 3:
                                    imgBackground[300:300 + 137, 865:865 + 125] = imgPpeList[0]
                                    ppe_helmet = True
                            elif sum(sublist.count("helmet") for sublist in ppe_list) <= 2:
                                imgBackground[300:300 + 137, 865:865 + 125] = imgPpeList[3]
                                ppe_helmet = False
                        else:
                            imgBackground[300:300 + 137, 865:865 + 125] = imgPpeList[3]
                            ppe_helmet = False
                        
                        if sum(sublist.count("no-vest") for sublist in ppe_list) <= 3:
                            if sum(sublist.count("vest") for sublist in ppe_list) >= 3:
                                    imgBackground[440:440 + 137, 865:865 + 125] = imgPpeList[1]
                                    ppe_vest = True
                            elif sum(sublist.count("vest") for sublist in ppe_list) <= 2:
                                imgBackground[440:440 + 137, 865:865 + 125] = imgPpeList[4]
                                ppe_vest = False
                        else:
                            imgBackground[440:440 + 137, 865:865 + 125] = imgPpeList[4]
                            ppe_vest = False
                            
                        if sum(sublist.count("no-mask") for sublist in ppe_list) <= 3:
                            if sum(sublist.count("mask") for sublist in ppe_list) >= 3:
                                    imgBackground[300:300 + 137, 1040:1040 + 125] = imgPpeList[2]
                                    ppe_mask = True
                            elif sum(sublist.count("mask") for sublist in ppe_list) <= 2:
                                imgBackground[300:300 + 137, 1040:1040 + 125] = imgPpeList[5]
                                ppe_mask = False
                        else:
                            imgBackground[300:300 + 137, 1040:1040 + 125] = imgPpeList[5]
                            ppe_mask = False
                    else:
                        ppe_mask = True
                        if sum(sublist.count("no-helmet") for sublist in ppe_list) <= 3:
                            if sum(sublist.count("helmet") for sublist in ppe_list) >= 3:
                                    imgBackground[300:300 + 137, 865:865 + 125] = imgPpeList[0]
                                    ppe_helmet = True
                            elif sum(sublist.count("helmet") for sublist in ppe_list) <= 2:
                                imgBackground[300:300 + 137, 865:865 + 125] = imgPpeList[3]
                                ppe_helmet = False
                        else:
                            imgBackground[300:300 + 137, 865:865 + 125] = imgPpeList[3]
                            ppe_helmet = False
                        
                        if sum(sublist.count("no-vest") for sublist in ppe_list) <= 3:
                            if sum(sublist.count("vest") for sublist in ppe_list) >= 3:
                                    imgBackground[300:300 + 137, 1040:1040 + 125] = imgPpeList[1]
                                    ppe_vest = True
                            elif sum(sublist.count("vest") for sublist in ppe_list) <= 2:
                                imgBackground[300:300 + 137, 1040:1040 + 125] = imgPpeList[4]
                                ppe_vest = False
                        else:
                            imgBackground[300:300 + 137, 1040:1040 + 125] = imgPpeList[4]
                            ppe_vest = False
                            
                    #Set the Message if both PPE detected
                    if ppe_helmet and ppe_vest and ppe_mask:
                        imgBackground[635:635 + 35, 900:900 + 300] = clear_text
                        cv2.putText(imgBackground, "You are good to go. Stay Safe!", (905, 655), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)
                    else:
                        imgBackground[635:635 + 35, 900:900 + 300] = clear_text
                        cv2.putText(imgBackground, "Please wear PPE!!", (910, 655), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)


                #REFRESH SESSIONS
                if ppe_item:
                    item_number += 1

                if item_number % 5 == 0 and most_frequent_name != "Unknown":
                    counter += 1
                    print("Session Time: " + str(counter))
                    imgBackground[120:120 + 23, 50:50 + 440] = clear_text2               
                    cv2.putText(imgBackground, "Session Ends in " + str(11 - counter), (50, 140), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)

                #Alert Before SESSION ENDS
                if counter > 10:
                    #ALERT WHEN BREACH HAPPENED
                    breach_ppe = ""
                    if not ppe_helmet:
                        breach_ppe += "no-helmet , "
                    if not ppe_vest:
                        breach_ppe += "no-vest , "
                    if not ppe_mask:
                        breach_ppe += "no-mask , "

                    breach_ppe = breach_ppe[:-3]
                    print(breach_ppe)

                    if breach_ppe != "":
                        alert_process(breach_ppe, most_frequent_name, worker_breaches)

                    #RESET SESSIONS
                    imgBackground[120:120 + 23, 50:50 + 440] = clear_text2 
                    face_names.clear()
                    counter = 0
                    draw_box_face = True
                    draw_box_ppe = False
                
            if most_frequent_name == "Unknown":
                #RESET SESSIONS IF NO ONE DETECTED
                imgBackground[120:120 + 23, 50:50 + 440] = clear_text2 
                face_names.clear()
                counter = 0
                draw_box_face = True
                draw_box_ppe = False

        # Display the results
        cv2.imshow('SAFETY CONSTRUCTION SYSTEM', imgBackground)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# encoding = train_encoding("Loc.1.jpg")
# save_encodings(encoding)
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

main()