from flask import make_response, request
import pymongo
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.graphics.charts.textlabels import Label
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
from io import BytesIO

def download_pdf():
    report_type = request.args.get('report_type')
    days = int(request.args.get('days'))
    project_id = str(request.args.get('project_id'))
    print(report_type)
    print(days)
    print(project_id)
    # Connect to the MongoDB database
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]
    
    if report_type == "incidents":
        collection = db["hazards" + str("_"+project_id)]
        print(collection)
    elif report_type == "breaches":
        collection = db["db_breaches" + str("_"+project_id)]
        print(collection)
        
    # Define the query based on the report type and days
    query = {}

    if days:
        start_date = datetime.today() - timedelta(days=days)
        if report_type == "incidents":
            query["timestamp"] = {"$gte": start_date}
        else:
            query["datetime"] = {"$gte": start_date}


    print(query)
    data = collection.find(query) # Get the data from the MongoDB collection

    # Create a PDF canvas (Letter sized)
    pdf = canvas.Canvas("myreport.pdf", pagesize=letter) 

    # Define the layout and formatting of the PDF report (Page 1 of the report)
    pdf.setFont("Helvetica-Bold", 48) 
    pdf.drawCentredString(4.25 * inch, 7.5 * inch, "Loc Company PTE LTD")
    pdf.setFont("Helvetica-Bold", 36)
    pdf.drawCentredString(4.25 * inch, 6.75 * inch, "Safety Report")

    # Set today's date and time and convert the string and put today's date and time at the bottom of the first page 
    now = datetime.now()
    dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    pdf.setFont("Helvetica", 18)
    pdf.drawCentredString(4.25 * inch, 1 * inch, "Generated on {}".format(dt_string))

    # Start writing the data on the second page
    pdf.showPage()

    # Define the layout and formatting of the PDF report for the remaining pages
    page_size = letter
    left_margin = 0.5 * inch
    top_margin = 10 * inch # Change the top margin to 10 inches
    bottom_margin = 0.5 * inch
    page_number = 2

    hazard_types_count = {}
    resolved_cases = 0
    unresolved_cases = 0

    for d in data:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(4.25 * inch, 10.75 * inch, "Safety Report -- Page {}".format(page_number-1)) # Change the y position to 10.75 inches
        y = top_margin

        

        if "item" in d:
            hazard_type = d["item"]
            hazard_types_count[hazard_type] = hazard_types_count.get(hazard_type, 0) + 1
            resolved = str(d.get("case_resolved"))
            print("Resolved?: " + resolved)
            if resolved == "True":
                resolved_cases += 1
            else:
                unresolved_cases += 1


        print("Data from MongoDB collection:")
        for key, value in hazard_types_count.items():
            print(f"{key}: {value}")


        # Generate the pie chart
        pie_labels_count = list(hazard_types_count.keys())
        pie_sizes_count = list(hazard_types_count.values())

        # past_week_date = datetime.today() - timedelta(days=7)

        if page_number == 2: 
            if report_type == "incidents":
                pdf.drawString(left_margin, y, "Number of hazards within the past {} days: {}".format(days, value))


                create_pie_chart(pie_labels_count, pie_sizes_count, "pie_chart.png", width=5, height=5)
                pie_chart_img = ImageReader("pie_chart.png")
                pdf.drawImage(pie_chart_img, 0.3 * inch, 6 * inch, width=3.75 * inch, height=3.75 * inch)
                pdf.setFont("Helvetica-Bold", 12) 
                pdf.drawString(1.5 * inch, 6.2 * inch, "Type of Hazard")  


                create_pie_chart(["Resolved Cases", "Unresolved Cases"], [resolved_cases, unresolved_cases], "pie_chart_resolved.png", width=5, height=5)
                pie_chart_img = ImageReader("pie_chart_resolved.png")
                pdf.drawImage(pie_chart_img, 4.3 * inch, 6 * inch, width=3.75 * inch, height=3.75 * inch)
                pdf.setFont("Helvetica-Bold", 12) 
                pdf.drawString(5.6 * inch, 6.2 * inch, "Resolved Cases")  


            elif report_type == "breaches":
                past_week_date = datetime.today() - timedelta(days=7)
                past_week_count = collection.count_documents({"datetime": {"$gte": past_week_date}})
                pdf.drawString(left_margin, y, "Number of breaches within the past 7 days: {}".format(past_week_count))
        

    # Draw page number at the bottom of the last page
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(4.25 * inch, bottom_margin, "Safety Report -- Page {}".format(page_number-1))
    pdf.showPage()

    # Save the PDF content to a BytesIO stream
    pdf_stream = BytesIO()
    pdf_stream.write(pdf.getpdfdata())

    # Set the file position back to the beginning of the stream
    pdf_stream.seek(0)

    # Create a Flask response with the PDF content
    response = make_response(pdf_stream.getvalue())

    # Set the appropriate content-disposition headers for attachment download
    response.headers['Content-Disposition'] = 'attachment; filename=myreport.pdf'

    # Set the appropriate content-type header
    response.headers['Content-Type'] = 'application/pdf'

    # Close the PDF canvas
    pdf_stream.close()

    return response

def create_pie_chart(labels, sizes, filename, width=5, height=5):
    print("Pie Chart Data:")
    for label, size in zip(labels, sizes):
        print(f"{label}: {size}")

    plt.figure(figsize=(width, height))
    wedges, labels, autopct = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    for label in labels:
        label.set_fontsize(5)
    plt.savefig(filename)
    plt.close()
