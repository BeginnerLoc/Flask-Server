import base64
from pymongo import MongoClient
import cv2
import numpy as np

uri = "mongodb+srv://loctientran235:PUp2XTv7tkArDjJB@c290.5lmj4xh.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
db_client = MongoClient(uri)

db = db_client["construction"]
collection2 = db["breach_images"]

# Retrieve the documents from the collection
cursor = collection2.find()

for document in cursor:
    # Get the Base64 encoded image from the document
    base64_image = document["image"]

    # Convert the Base64 encoded image to bytes
    image_bytes = base64.b64decode(base64_image)

    # Convert the bytes to a NumPy array
    np_array = np.frombuffer(image_bytes, np.uint8)

    # Decode the NumPy array as an image
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Display or process the image as needed
    cv2.imshow("Retrieved Image", image)
    cv2.waitKey(0)

# Close the MongoDB connection
db_client.close()
