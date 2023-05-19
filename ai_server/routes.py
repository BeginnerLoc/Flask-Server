from flask import Blueprint, render_template, jsonify
import json
from .extensions import db_client
import datetime
from .ai_thread import AiThread
import threading

main = Blueprint("main", __name__)

@main.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

@main.route("/")
def index():
    return render_template("index.html")


@main.route("/test")
def test():
    ai_thread = AiThread()
    thread = threading.Thread(target=ai_thread.recognition)
    thread.start()
    return render_template("index.html")

# return JSON containing the breach with most cases and the number of cases
@main.route("/api/most_breaches")
def most_breaches():
    
    db = db_client["construction"]
    collection = db["breaches"]
    
    # retrieve documents from the collection
    result = collection.aggregate([
    {"$group": {"_id": "$type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 1}])

    # Get the first document from the result
    doc = next(result, None)
 
    response = json.dumps(doc)
    
    return jsonify(response)

# return JSON containing the today's number of breaches 
@main.route("/api/today_breaches")
def today_breaches():
    
    db = db_client["construction"]
    collection = db["breaches"]
    
    # Get the start and end of today
    start_of_today = datetime.datetime.combine(datetime.datetime.today().date(), datetime.time.min)
    end_of_today = datetime.datetime.combine(datetime.datetime.today().date(), datetime.time.max)

    # Count the number of documents with today's date
    count = collection.count_documents({'datetime': {'$gte': start_of_today, '$lte': end_of_today}})
    
    return jsonify(count)

# return JSON with all the attributes of breaches
@main.route("/api/graph_breaches")
def breaches():
    db = db_client["construction"]
    collection = db["db_breaches"]
    
    # retrieve documents from the collection
    result = collection.find()

    # Convert the result to a list of documents and convert ObjectId to string
    documents = [doc for doc in result]
    for doc in documents:
        doc["_id"] = str(doc["_id"])
    
    # Return the list of documents as a JSON response
    return jsonify(documents)