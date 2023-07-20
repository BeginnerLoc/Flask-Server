from flask import Blueprint, render_template, jsonify, request
import json
from .extensions import db_client
import datetime
from .ai_thread import AiThread
import threading
from utils.create_pdf import download_pdf
import openai

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

# Count the number of workers and display it in "Workers Working Today"
@main.route("/api/<project_id>/num_check_in")
def live_checkin(project_id):
    db = db_client["construction"]
    collection_name = "checkin_" + project_id
    collection = db[collection_name]
    
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
        return {"num_check_ins": 0}
    

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
    # print(user_prompt)
    # Send user prompt to OpenAI and get a response
    # response = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo',
    #     max_tokens=500,  # Adjust the max tokens limit as needed
    #     temperature=1,  # Adjust the temperature for more or less randomness
    #     messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": user_prompt}]
    # )
    # answer = response.choices[0].message.content

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=user_prompt,
        max_tokens=500,  # Adjust the max tokens limit as needed
        temperature=0.8 # Adjust the temperature for more or less randomness
    )
    answer = response.choices[0].text.strip()

    print(answer)            
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
        The data is the workers that committed the most number of breaches.
        Give me suggestions for each worker on how to reduce the breaches
    """
    # print(user_prompt)
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

    print(answer)            
    return answer