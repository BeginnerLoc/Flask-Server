from flask import Blueprint, render_template, jsonify

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.html")

# return JSON containing the incient with most cases and the number of cases
@main.route("/api/most_incidents")
def most_incidents():
    response =  {'breach': 'no helmet', 'incident_count': 100}
    return jsonify(response)