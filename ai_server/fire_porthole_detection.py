from ultralytics import YOLO
from telegram import Bot
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import pymongo
from datetime import datetime, timedelta
import time
import asyncio

# Initialising telegram connection
async def send_message(token, chat_id, message):
    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text=message)

bot_token = '6060060457:AAGRyic-1HVFcUy1dSEsdLMJo0rB9Mvz0y0'
chat_id = '-1001910285228'

# Initialising mongodb connection
client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
db = client["construction"]
collection = db["Incidents"]

#Initialise fire detected time = 0 and porthole detected = False
fire_detected_time = None
fire_detected_this_round = None
fire_detected = False
porthole_detected = False
porthole_post = False 

#Porthole and Fire models
# model1 = YOLO('D:\Workspace\Flask-Server\ai_server\ai_model\hole.pt')
model1 = YOLO('./ai_model/hole.pt')
model1.model.conf_thres = 0.7 # Set detection threshold
model2 = YOLO('./ai_model/fire.pt')
model2.model.conf_thres = 0.7 # Set detection threshold


#Initialise the webcam capture device of index 0 and set its resolution to 640x480
cap = cv2.VideoCapture(0)  
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results1 = model1.predict(img)
    annotator = Annotator(frame)

    #Pothole detection
    for r in results1:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  
            c = box.cls
            annotator.box_label(b, f"{model1.names[int(c)]} {box.conf.item():.2f}")
            print(model1.names[int(c)])
            if model1.names[int(c)] == "0" and box.conf.item() >= 0.7:
                if not porthole_detected: # Check if porthole is not already detected
                    porthole_detected = True # Set the flag to True
                    porthole_detected_time = datetime.now()#Record the time
                else:
                    elapsed_time = datetime.now() - porthole_detected_time
                    print(elapsed_time)
                    if elapsed_time > timedelta(seconds=5) and porthole_post == False: # Check if 5 seconds have passed and the porthole has not been posted yet
                        # Check if porthole is still detected before posting to MongoDB
                        if any(box.cls == 0 and box.conf.item() >= 0.7 for box in boxes):
                            porthole_post = True
                            post = {
                                    "incident_type": "porthole",
                                    "timestamp": datetime.now(),
                                    "confidence": 0.7
                                    }
                            collection.insert_one(post) # Add an entry to MongoDB

    # Fire Model detection
    results2 = model2.predict(img)
    fire_detected_round = False # initialize fire_detected_round flag
    for r in results2:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  
            c = box.cls
            annotator.box_label(b, f"{model2.names[int(c)]} {box.conf.item():.2f}")
            if model2.names[int(c)] == 'zfire' and box.conf.item() >= 0.5: # If fire is detected, record the time
                if not fire_detected: 
                    fire_detected_time = datetime.now()
                fire_detected_round = True 
                fire_detected = True
                print('Fire detected')
            else: # If fire is not detected, set fire_detected_round to False
                fire_detected_round = False
        # Check if fire has been detected for more than 8 seconds
        if not fire_detected_round and fire_detected: # If fire is no longer detected, reset the fire_detected_time and fire_detected flag
            fire_detected_time = None
            fire_detected = False
            print('Fire no longer detected')

    if fire_detected_time is not None:
        elapsed_time = datetime.now() - fire_detected_time
        print(elapsed_time)
        if elapsed_time > timedelta(seconds=8): 
            post = { # Add an entry to MongoDB
                    "incident_type": "fire",
                    "timestamp": datetime.now(),
                    "confidence": 0.7
                    }
            message = 'A fire has been detected! \n Time of incident: ' + datetime.now().strftime('%d-%m-%Y %H:%M:%S') + '\n Confidence: ' + str(box.conf.item())
            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_message(bot_token, chat_id, message))
            collection.insert_one(post)
            
            # Reset the fire_detected_time to None
            fire_detected_time = None
            fire_detected = False
        else:
            time.sleep(1) #Delay 1 second
            if elapsed_time is not None and elapsed_time < timedelta(seconds=8) and not fire_detected:
                fire_detected_time = None
                elapsed_time = None
    else:     # Reset elapsed time if no fire has been detected for more than 8 seconds
        elapsed_time = None


    frame = annotator.result()
    cv2.imshow('YOLO V8 Detection', frame)     

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()