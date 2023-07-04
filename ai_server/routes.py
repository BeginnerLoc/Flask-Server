from flask import Blueprint, render_template, jsonify, request
import json
from .extensions import db_client
import datetime
from .ai_thread import AiThread
import threading
from utils.create_pdf import download_pdf

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
@main.route("/api/<project_id>/most_breaches")
def most_breaches(project_id):
    
    db = db_client["construction"]
    collection_name = 'db_breaches_' + project_id
    collection = db["db_breaches"]
    
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
from flask import jsonify

@main.route("/api/<project_id>/today_breaches")
def today_breaches(project_id):
    
    db = db_client["construction"]
    collection_name = "db_breaches_" + project_id
    collection = db[collection_name]
    
    # Get the start and end of today
    start_of_today = datetime.datetime.combine(datetime.datetime.today().date(), datetime.time.min)
    end_of_today = datetime.datetime.combine(datetime.datetime.today().date(), datetime.time.max)

    # Count the number of documents with today's date
    count = collection.count_documents({'datetime': {'$gte': start_of_today, '$lte': end_of_today}})
    
    # Return the count as JSON
    return jsonify({'count': count})


@main.route('/download_pdf', methods=['GET'])
def download_pdf_route():
    return download_pdf()

# return JSON with all the attributes of breaches.
@main.route("/api/<project_id>/graph_breaches")
def breaches(project_id):
    db = db_client["construction"]
    collection_name = "db_breaches_" + project_id
    collection = db[collection_name]
    
    # retrieve documents from the collection
    result = collection.find()

    # Convert the result to a list of documents and convert ObjectId to string
    documents = [doc for doc in result]
    for doc in documents:
        doc["_id"] = str(doc["_id"])
    
    # Return the list of documents as a JSON response
    return jsonify(documents)

# return JSON containing the all the workers details with filter 

@main.route("/api/<project_id>/indiv_breaches")
def get_indiv_breaches(project_id):
    db = db_client["construction"]
    collection_name = "db_breaches_" + project_id
    collection = db[collection_name]
    name_filter = request.args.get("name")

    query = {}
    if name_filter:
        query["worker_name"] = name_filter

    # Retrieve documents from the collection based on the filter
    documents = collection.find(query)

    # Convert documents to a list
    breach_list = list(documents)

    response = json.dumps(breach_list, default=str)
    return jsonify(response)



# return JSON containing the all the workers details
@main.route("/api/<project_id>/all_workers")
def all_workers(project_id):
    db = db_client["construction"]
    collection_name = "workers_" + project_id
    collection = db[collection_name]

    # Retrieve all documents from the collection, excluding the _id field
    documents = collection.find({}, {"_id": 0})

    # Convert documents to a list
    worker_list = [worker for worker in documents]

    return jsonify(worker_list)

# Count the number of workers and display it
@main.route("/api/num_check_in")
def live_checkin():
    db = db_client["construction"]
    collection = db["checkin"]
    
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Retrieve the document for the current date
    document = collection.find_one({"date": current_datetime.date().strftime("%Y-%m-%d")})

    if document:
        # Count the number of worker check-ins
        num_check_ins = len(document["check_ins"])

        response = {"num_check_ins": num_check_ins}
        return jsonify(response)
    else:
        return 0 
    

@main.route("/api/<project_id>/num_hazards")
def num_hazards(project_id):
    db = db_client["construction"]
    collection_name = "incidents_" + project_id
    collection = db[collection_name]

    result = collection.count_documents({})
    
    response = {"num_hazards": 0}
    
    if result:
            # Count the number of hazards
        response = {"num_hazards": result}
 
    return jsonify(response)

    
