from flask import Blueprint, render_template, jsonify, request
import json
from .extensions import db_client
from datetime import datetime, timedelta
from .ai_thread import AiThread
import threading
from utils.create_pdf import download_pdf
import openai
import pymongo
from bson import ObjectId  # Import ObjectId if you need to use it


main = Blueprint("main", __name__)

openai.api_key = "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"

@main.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, PUT, POST, DELETE')
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
    collection = db[collection_name]
    
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
    start_of_today = datetime.combine(datetime.today().date(), datetime.today().min.time())
    end_of_today = datetime.combine(datetime.today().date(), datetime.today().max.time())
    
    print(start_of_today)
    print(end_of_today)
    

    # Count the number of documents with today's date
    count = collection.count_documents({'datetime': {'$gte': start_of_today, '$lte': end_of_today}})
    
    # Return the count as JSON
    return jsonify({'count': count})


@main.route('/download_pdf', methods=['GET'])
def download_pdf_route():
    report_type = request.args.get('report_type')
    start = request.args.get('start')
    end = request.args.get('end')
    project_id = str(request.args.get('project_id'))

    # Connect to the MongoDB database
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]
    
    if report_type == "incidents":
        collection = db["hazards" + str("_"+project_id)]
    elif report_type == "breaches":
        collection = db["db_breaches_2"] #+ str("_"+project_id) 

    # Define the query based on the report type and days
    query = {}

   # Calculate the time difference
    start_date = datetime.fromisoformat(start[:-1]) 
    end_date = datetime.fromisoformat(end[:-1])

    # Access the number of days from the timedelta object
    time_difference = end_date - start_date
    days = time_difference.days

    print("Report Type = " + report_type)

    if report_type == "incidents":
        query = [
            {"$match": {"timestamp": {"$gte": start_date, "$lte": end_date}}},
            {"$project": {"image": 0}}
        ]
    else:
        query = [
            {"$match": {"datetime": {"$gte": start_date, "$lte": end_date}}},
            {"$project": {"evidence_photo": 0}} 
        ]

    data = collection.aggregate(query)
    data_list = list(data)

    # gpt_answer = explain_answer(data_list) #Uncomment when using chatgpt
    gpt_answer = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    #Comment line above when using chatgpt

    return download_pdf(report_type, days, project_id, data_list, gpt_answer, start, end)

# return JSON with all the attributes of breaches.
@main.route("/api/<project_id>/graph_breaches")
def breaches(project_id):
    db = db_client["construction"]
    collection_name = "db_breaches_" + project_id
    collection = db[collection_name]
    
    # retrieve documents from the collection
    result = collection.find({}, {"evidence_photo": 0})

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
    documents = collection.find(query, {"evidence_photo": 0})

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

# Count the number of workers and display it in "Workers Working Today"
@main.route("/api/<project_id>/num_check_in")
def live_checkin(project_id):
    db = db_client["construction"]
    collection_name = "checkin_" + project_id
    collection = db[collection_name]
    
    # Get the current date and time
    current_datetime = datetime.now()

    # Retrieve the document for the current date
    document = collection.find_one({"date": current_datetime.date().strftime("%Y-%m-%d")}, {"evidence_photo": 0})

    if document:
        # Count the number of worker check-ins
        num_check_ins = len(document["check_ins"])

        response = {"num_check_ins": num_check_ins}
        return jsonify(response)
    else:
        return {"num_check_ins": 0}
    

@main.route("/api/<project_id>/num_hazards")
def num_hazards(project_id):
    db = db_client["construction"]
    collection_name = "hazards_" + project_id
    collection = db[collection_name]

    result = collection.count_documents({})
    
    response = {"num_hazards": 0}
    
    if result:
            # Count the number of hazards
        response = {"num_hazards": result}
 
    return jsonify(response)

    
@main.route('/ask_gpt', methods=['POST'])
def ask_gpt():
    data = request.json
    answer = explain_answer(data)
    
    response = {"answer": answer}
    return jsonify(response)

    
def explain_answer(data):
    
    user_prompt= f"""
        <data>{data}</data>
        ####
        Give me the analysis of data and give suggestion for improvement, give 50 words answer
    """
    # Send user prompt to OpenAI and get a response
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=500,  # Adjust the max tokens limit as needed
        temperature=1,  # Adjust the temperature for more or less randomness
        messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": user_prompt}]
    )
    answer = response.choices[0].message.content

    # response = openai.Completion.create(
    #     engine='text-davinci-003',
    #     prompt=user_prompt,
    #     max_tokens=500,  # Adjust the max tokens limit as needed
    #     temperature=0.8 # Adjust the temperature for more or less randomness
    # )
    # answer = response.choices[0].text.strip()

    return answer


@main.route('/ask_gpt_breach_graph', methods=['POST'])
def ask_gpt_breach_graph():
    data = request.json
    answer = explain_top_breach(data)
    
    response = {"answer": answer}
    return jsonify(response)

def explain_top_breach(data):
    
    user_prompt= f"""
        <data>{data}</data>
        ####
        The data is the workers that committed the most number of breaches from a bar chart.
        Give detailed analysis as to why such breaches happedn and suggestions.
        The answer should be in this format: "1. point1 2. point2 ... other points".
        The answer should not excced 50 words
    """
    # Send user prompt to OpenAI and get a response
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=500,  # Adjust the max tokens limit as needed
        temperature=1,  # Adjust the temperature for more or less randomness
        messages=[{"role": "system", "content": "You are an expert Data Analyst."},{"role": "user", "content": user_prompt}]
    )
    answer = response.choices[0].message.content

    return answer

@main.route('/message_chatgpt', methods=['POST'])
def message_chatgpt():
    data = request.json
    messages = data["messages"]
    new_message = data["new_message"]
    # new_message = ""
    
    answer = normal_message(messages, new_message)
    
    response = {"answer": answer}
    return jsonify(response)

def normal_message(previous_prompt, question):
    user_prompt= f"""
        <previous_prompt>{previous_prompt}</previous_prompt>
        ####
        Answer my question based on the previous prompt context. The answer should not excced 50 words
        <question>{question}</question>
    """
    # Send user prompt to OpenAI and get a response
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=200,  # Adjust the max tokens limit as needed
        temperature=1,  # Adjust the temperature for more or less randomness
        messages=[{"role": "system", "content": "You are an expert Data Analyst."},{"role": "user", "content": user_prompt}]
    )
    answer = response.choices[0].message.content
    return answer

# Return the breach image from db_breaches_2 according to the breach ID.
@main.route('/api/<project_id>/get_breach_image/<string:breach_id>', methods=['GET'])
def get_breach_image(project_id, breach_id):
    db = db_client["construction"]
    collection_name = "db_breaches_" + project_id
    collection = db[collection_name]

    try:
        # Find the breach record in the database based on the breach_id
        breach = collection.find_one({'breach_id': int(breach_id)})
        if breach is not None and 'evidence_photo' in breach:
            # Get the base64 encoded image value from the "evidence_photo" field
            base64_image = breach['evidence_photo']

            # Retrieve worker's information from db_breaches
            worker_name = breach['worker_name']
            description = breach['description']
            breach_id = breach['breach_id']
            location = breach['location']

            return jsonify({
                'worker_name': worker_name,
                'description': description,
                'breach_id': breach_id,
                'location': location,
                'image': base64_image 
                            })

        else:
            return jsonify({'error': 'Image not found for the given breach_id'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

