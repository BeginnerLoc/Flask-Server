import face_recognition
import cv2
import numpy as np
from pymongo.mongo_client import MongoClient
from ultralytics import YOLO
import threading
import time
import collections
from collections import Counter
import os


uri = "mongodb+srv://loctientran235:PUp2XTv7tkArDjJB@c290.5lmj4xh.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
db_client = MongoClient(uri)
db = db_client["construction"]
collection = db["encodings"]
# Send a ping to confirm a successful connection
try:
    db_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

model2 = YOLO('ai_server\\ai_model\\ppe_model.pt')

stop_event = threading.Event()

employee_data = None

imgBackground = cv2.imread('ai_server\\UI_photos\\background.png')
model = cv2.imread('ai_server\\UI_photos\\pageA.png')
clear_text = cv2.imread('ai_server\\UI_photos\\clear.png')

#statusList
folderStatusPath = 'ai_server\\UI_photos\\status'
statusPathList = os.listdir(folderStatusPath)
imgStatusList = []
for path in statusPathList:
    imgStatusList.append(cv2.imread(os.path.join(folderStatusPath, path)))

#avaList
folderAvaPath = 'ai_server\\UI_photos\\Ava'
avaPathList = os.listdir(folderAvaPath)
imgAvaList = []
for path in avaPathList:
    imgAvaList.append(cv2.imread(os.path.join(folderAvaPath, path)))

#ppeList
folderPpePath = 'ai_server\\UI_photos\\ppe'
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

    print("=========================== " + employee_data["name"] + " is a " + employee_data["position"] + "================================")
       
def search_data_thread(name):
    thread = threading.Thread(target=update_employee, args=(name,))
    thread.start()
    return thread

def recognition():
    encoding_list = retrieve_encoding()
    
    known_face_encodings = [encoding[1] for encoding in encoding_list]
    known_face_names = [encoding[0] for encoding in encoding_list]
    
    face_locations = []
    face_encodings = []
    face_names = []
    ppe_list = []
    #process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        imgBackground[158:158 + 480, 52:52 + 640] = frame
        imgBackground[30:30 + 674, 800:800 + 440] = model

        #Only process every other frame of video to save time
        #if process_this_frame:

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        cv2.putText(imgBackground, "Authenticating...", (910, 655), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        cv2.imshow('Video', imgBackground)
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
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
                    
            if employee_data == None or employee_data['name'] != name:        
                thread = search_data_thread(name)
                # Set the stop_event object to stop the thread
                stop_event.set()
                # Wait for the thread to finish
                thread.join()
                # Reset the stop_event object for the next iteration
                stop_event.clear()

            face_names.append(name)
            print(name)

        #process_this_frame = not process_this_frame

        #Draw Box for Face
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            if name != "Unknown":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        results1 = model2.predict(frame, verbose=False)
        #PPE_list
        ppe_item = plot_bboxes(imgBackground[158:158 + 480, 52:52 + 640], results1[0].boxes.data, score=False, conf=0.85)
        ppe_list.append(ppe_item)
        print(ppe_item)

        #Find the most face detect
        if len(face_names) > 10:
            most_frequent_name = Counter(face_names).most_common(1)[0][0]

            if most_frequent_name != "Unknown" and most_frequent_name != None:
                cv2.putText(imgBackground, "Hi, " + most_frequent_name, (875, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(imgBackground, "PPE Require:", (830, 290), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(imgBackground, "Your JobScopes Today is:", (835, 210), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

                if most_frequent_name == "Astro":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[0]
                    cv2.putText(imgBackground, "Laborer: remove the shattered class", (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
                    
                elif most_frequent_name == "Chris":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[1]
                    cv2.putText(imgBackground, "Painter: paint the walls", (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
                    
                elif most_frequent_name == "Daren":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[2]
                    cv2.putText(imgBackground, "Electrician: install the power distribution", (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)

                elif most_frequent_name == "Loc":
                    imgBackground[50:50 + 108, 1105:1105 + 108] = imgAvaList[3]
                    cv2.putText(imgBackground, "Welder: operate welding equipment", (845, 235), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
                
                #PPE Require - 0-Helmet and 1-Vest
                imgBackground[330:330 + 125, 865:865 + 105] = imgPpeList[0]
                #140 spaces
                imgBackground[470:470 + 125, 865:865 + 105] = imgPpeList[1]

                #Default Status - all not detected
                # imgBackground[320:320 + 130, 1030:1030 + 152] = imgStatusList[3]
                # #140 spaces
                # imgBackground[460:460 + 130, 1030:1030 + 152] = imgStatusList[2]

                if len(ppe_list) > 20:
                    ppe_list = ppe_list[-20:]
                    #if PPE existing more than 5 times is True, less than 2 is as False
                    if ppe_list.count("Hardhat") >= 5:
                        status1 = imgStatusList[0]
                    elif ppe_list.count("Hardhat") <= 2:
                        status1 = imgStatusList[1]
                    
                    if ppe_list.count("Safety vest") >= 5:
                        status2 = imgStatusList[2]
                    elif ppe_list.count("Safety vest") <= 2:
                        status2 = imgStatusList[3]
                    
                    imgBackground[320:320 + 130, 1030:1030 + 152] = status1
                    imgBackground[460:460 + 130, 1030:1030 + 152] = status2

                    if ppe_list.count("Safety Vest") >= 2 and ppe_list.count("Hardhat") >= 2:
                        imgBackground[635:635 + 35, 900:900 + 300] = clear_text
                        cv2.putText(imgBackground, "You’re good to go. Stay Safe!", (910, 655), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 2)
                    else:
                        imgBackground[635:635 + 35, 900:900 + 300] = clear_text
                        cv2.putText(imgBackground, "Please wear PPE!!", (910, 655), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                session_start_time = time.time()

                if int(time.time() - session_start_time) > 5:
                    #Check the last 20 names for the most frequent name
                    most_frequent_name_current = Counter(face_names[-10:]).most_common(1)[0][0]
                    if most_frequent_name != most_frequent_name_current:
                        most_frequent_name = most_frequent_name_current

        # Display the results
        cv2.imshow('Video', imgBackground)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


#play the bounding boxes with the label and the score :
# def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
#   #Construction workers are required to wear hat - vest - mask
#   #Managers are required to wear hat - vest
#     #   requirements = []
#     #   if employee_data!= None and employee_data['position'] == 'Manager':    
#     #     requirements = ['NO-Mask']
    
#     #   if label != 'Person':
#     lw = max(round(sum(image.shape) / 2 * 0.003), 2)
#     p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
#     cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
#     if label and label != "Person":
#         tf = max(lw - 1, 1)  # font thickness
#         w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
#         outside = p1[1] - h >= 3
#         p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
#         cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(image,
#                     label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
#                     0,
#                     lw / 3,
#                     txt_color,
#                     thickness=tf,
#                     lineType=cv2.LINE_AA)
        
#         #Status
#         #print(label)

#         return label
        

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    output = ""
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
            else:
                color = colors[int(box[-1])]

            #box label
            if label != 'Person' and label != 'Mask' and label != 'NO-Mask' and label != 'machinery'and label != 'vehicle':
                lw = max(round(sum(image.shape) / 2 * 0.003), 2)
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                if label and label != "Person":
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
                    output = label
    return output


# encoding = train_encoding("Loc.1.jpg")
# save_encodings(encoding)
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)


recognition()