import face_recognition
import cv2
import numpy as np
from pymongo.mongo_client import MongoClient

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




# encoding = train_encoding("Tai.1.jpg")
# save_encodings(encoding)
video_capture = cv2.VideoCapture(0)
recognition()