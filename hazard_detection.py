import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
from datetime import datetime, timedelta
import time
import pymongo
import math
import base64

async def send_message(bot_token, chat_id, message):
    retries = 3
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=5.0)) as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    json={"chat_id": chat_id, "text": message}
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

#bot_token and chat_id to be used to initialise telegram connection
bot_token = '6060060457:AAGRyic-1HVFcUy1dSEsdLMJo0rB9Mvz0y0'
chat_id = '-1001910285228'

# Initialise model
model = YOLO('Tools.pt')
model.model.conf_thres = 0.7  # Set detection threshold

# Initialise the webcam capture device of index 0 and set its resolution to 640x480
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Define the region of interest (ROI) coordinates
roi_x1 = 150  # Adjust the left border size (in pixels)
roi_y1 = 0    # Set the top border to 0
roi_x2 = int(cap.get(3)) - 150  # Adjust the right border size (in pixels)
roi_y2 = int(cap.get(4))  # Set the bottom border to the full height of the frame

# Variables for object tracking
tracked_object_id = None
tracked_object_position = None
tracked_object_timer = None
capture_time_str = "2023-01-01 00:00:00.123456"
capture_time = datetime.strptime(capture_time_str, "%Y-%m-%d %H:%M:%S.%f")


# Function to calculate the Euclidean distance between two points
def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to store the image in MongoDB as base64
def store_image_in_mongodb(image, item_name):
    # Convert image to base64 format
    retval, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # Store base64 image in MongoDB
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]
    collection = db["hazards_1"]
    post = {"image": base64_image,
            "timestamp": datetime.now(),
            "location": "Walkway",
            "item": item_name
            }
    collection.insert_one(post)
    print("Posting")
    client.close()

while True:
    _, frame = cap.read()

    # Draw the rectangle ROI on the frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotator = Annotator(frame)
    results = model.predict(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            if box.conf.item() > model.model.conf_thres:
                annotator.box_label(b, f"{model.names[int(c)]} {box.conf.item():.2f}")
                x, y, _, _ = b
                if roi_x1 <= x <= roi_x2 and roi_y1 <= y <= roi_y2:
                    if tracked_object_id is None:
                        # Start tracking the object
                        tracked_object_id = int(c)
                        tracked_object_position = (x, y)
                        tracked_object_timer = datetime.now()
                    elif tracked_object_id == int(c):
                        # Object is already being tracked
                        distance = calculate_distance(tracked_object_position, (x, y))
                        if distance < 5:
                            # Object has minor movement, continue tracking
                            elapsed_time = datetime.now() - tracked_object_timer
                            # Print the elapsed time
                            print(f"Object {tracked_object_id} has remained in the same position for: {elapsed_time}")
                            
                            #If object is detected in ROI > 10 seconds, it will check if there is any mongodb entries posted in the last 10 mins
                            if elapsed_time.total_seconds() > 10 and (datetime.now() - capture_time) > timedelta(minutes=10):
                                print("True, More than 10s!")
                                capture_image = frame.copy()
                                capture_time = datetime.now()
                                item_name = model.names[int(c)]
                                location = "Walkway"
                                # if check_mongodb(item_name, location):
                                store_image_in_mongodb(capture_image, item_name)
                        else:
                            # Object has moved significantly, reset tracking variables
                            tracked_object_position = (x, y)
                            tracked_object_timer = datetime.now()
                    else:
                        # Object has changed, update tracking variables
                        tracked_object_id = int(c)
                        tracked_object_position = (x, y)
                        tracked_object_timer = datetime.now()

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()