from flask import Blueprint, render_template, jsonify
import json
from .extensions import db_client
import datetime

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.html")

# return JSON containing the incient with most cases and the number of cases
@main.route("/api/most_incidents")
def most_incidents():
    
    db = db_client["construction"]
    collection = db["incidents"]
    
    # retrieve documents from the collection
    result = collection.aggregate([
    {"$group": {"_id": "$type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 1}])

    # Get the first document from the result
    doc = next(result, None)
 
    response = json.dumps(doc)
    
    return jsonify(response)


@main.route("/api/today_cases")
def today_cases():
    
    db = db_client["construction"]
    collection = db["incidents"]
    
    # Get the start and end of today
    start_of_today = datetime.datetime.combine(datetime.datetime.today().date(), datetime.time.min)
    end_of_today = datetime.datetime.combine(datetime.datetime.today().date(), datetime.time.max)

    # Count the number of documents with today's date
    count = collection.count_documents({'datetime': {'$gte': start_of_today, '$lte': end_of_today}})
    
    return jsonify(count)