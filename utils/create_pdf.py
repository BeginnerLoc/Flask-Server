from flask import make_response

def download_pdf():
    import pymongo
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from datetime import datetime, timedelta
    from io import BytesIO

    # Connect to the MongoDB database
    client = pymongo.MongoClient("mongodb+srv://Astro:enwVEQqCyk9gYBzN@c290.5lmj4xh.mongodb.net/")
    db = client["construction"]
    collection = db["Incidents"]
    data = collection.find() # Get the data from the MongoDB collection

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
    pdf.setFont("Helvetica", 24)
    pdf.drawCentredString(4.25 * inch, 1 * inch, "Generated on {}".format(dt_string))

    # Start writing the data on the second page
    pdf.showPage()

    # Define the layout and formatting of the PDF report for the remaining pages
    page_size = letter
    left_margin = 0.5 * inch
    top_margin = 10 * inch # Change the top margin to 10 inches
    bottom_margin = 0.5 * inch
    line_height = 14
    max_entries_per_page = 20
    page_number = 2

    # Write the data to the PDF report
    y = top_margin
    entry_count = 0
    for d in data:
        if entry_count == 0:
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawCentredString(4.25 * inch, 10.75 * inch, "Safety Report -- Page {}".format(page_number-1)) # Change the y position to 10.75 inches
            y = top_margin
            
            if page_number == 2:  # Display "Number of incidents within the past 7 days" only on the second page
                # Create a table for the number of incidents within the past 7 days
                past_week_date = datetime.today() - timedelta(days=7)
                past_week_count = collection.count_documents({"timestamp": {"$gte": past_week_date}})
                pdf.setFont("Helvetica-Bold", 12)
                pdf.drawString(left_margin, y, "Number of incidents within the past 7 days: {}".format(past_week_count))
                y -= line_height + 50
            
            pdf.setFont("Helvetica", 12)
            pdf.drawString(left_margin, y, "Incident Type")
            pdf.drawString(left_margin + 2.5 * inch, y, "Timestamp")
            y -= line_height
        
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left_margin, y, "{}".format(d["incident_type"]))
        pdf.setFont("Helvetica", 12)
        pdf.drawString(left_margin + 2.5 * inch, y, "{}".format(d["timestamp"]))
        y -= line_height
        entry_count += 1
        if entry_count == max_entries_per_page:
            entry_count = 0
            # Draw page number at the bottom of each page
            pdf.setFont("Helvetica", 12)
            pdf.drawCentredString(4.25 * inch, bottom_margin, "Safety Report -- Page {}".format(page_number-1))
            pdf.showPage()
            page_number += 1

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