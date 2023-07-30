import face_recognition
import cv2
import numpy as np
from pymongo.mongo_client import MongoClient
from ultralytics import YOLO
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

import os

uri = "mongodb+srv://loctientran235:PUp2XTv7tkArDjJB@c290.5lmj4xh.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
db_client = MongoClient(uri)
db = db_client["construction"]
collection = db["encodings_test3"]
# Send a ping to confirm a successful connection
try:
    db_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


model2 = YOLO('./ai_model/ppe_model.pt')


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

def get_encodings_from_folder(folder_path):
    # Initialize an empty list to store encodings
    encodings_list = []

    # Define image augmentation sequence
    augmentation = iaa.Sequential([
        # iaa.Resize({"height": 300, "width": 300}),  # Resize images to a fixed size
        iaa.Multiply((1.1))  # Multiply pixel values to increase brightness
    ])

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load the image
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Apply augmentation to the image
            augmented_image = augmentation.augment_image(image)

            # # Display the original and augmented images
            # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # axes[0].set_title("Original Image")
            # axes[1].imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
            # axes[1].set_title("Augmented Image")
            # plt.show()

            # Encode the augmented image and save the encoding to the list
            encoding = face_recognition.face_encodings(augmented_image)[0]
            name = filename.split(".")[0]
            object_encoding = encoding.astype(object)
            result = np.insert(object_encoding, 0, name)
            encodings_list.append(result)

    return encodings_list

folder_path = "C:\\FYP\\Flask-Server\\ai_server\\images"
encodings_list = get_encodings_from_folder(folder_path) 

# Save the list of encodings to the database
for encoding in encodings_list:
    # convert the array to BSON
    encoding_list = encoding.tolist()
    collection.insert_one({"encode": encoding_list})


def retrieve_encoding(): 
    # retrieve all the documents from the collection
    encoding_data = collection.find(sort=[('_id', -1)])

    # create a list to store the encodings
    encoding_list = []
    for encoding in encoding_data:
        num_vals = encoding['encode'][1:]
        detect_name = encoding['encode'][0]
        encode_np = np.array(num_vals)
        encoding_list.append((detect_name, encode_np))
        
    return encoding_list    
    
def recognition():
    encoding_list = retrieve_encoding()
    
    known_face_encodings = [encoding[1] for encoding in encoding_list]
    known_face_names = [encoding[0] for encoding in encoding_list]
    
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
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

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            if name != "Unknown":
                
                results1 = model2.predict(frame)
                plot_bboxes(frame, results1[0].boxes.data, score=False, conf=0.85)
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


#ppe

#play the bounding boxes with the label and the score :
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    
  if label != 'Person':  
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label and label:
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



# encoding = train_encoding("Tai.1.jpg")
# save_encodings(encoding)

video_capture = cv2.VideoCapture(0)
recognition() 