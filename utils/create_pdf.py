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
import textwrap

def download_pdf(report_type, days, project_id, data, gpt_answer, start, end):
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
    top_margin = 10 * inch 
    bottom_margin = 0.5 * inch
    page_number = 2

    hazard_types_count = {}
    # hazard_counts_to_display = [] 
    breach_types_count = {}
    resolved_cases = 0
    unresolved_cases = 0

    for d in data:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawCentredString(4.25 * inch, 10.75 * inch, "Safety Report -- Page {}".format(page_number-1)) 

        if "item" in d:
            hazard_type = d["item"]
            hazard_types_count[hazard_type] = hazard_types_count.get(hazard_type, 0) + 1
            resolved = str(d.get("case_resolved"))
            if resolved == "True":
                resolved_cases += 1
            else:
                unresolved_cases += 1

        if "description" in d:
            breach_type = d["description"]
            breach_id = d["breach_id"]
            # print(breach_type + str(breach_id))
            breach_types_count[breach_type] = breach_types_count.get(breach_type, 0 ) + 1
            resolved = str(d.get("case_resolved"))
            if resolved == "True":
                resolved_cases += 1
            else:
                unresolved_cases += 1

        # Generate the pie chart
        pie_labels_count = list(hazard_types_count.keys())
        pie_sizes_count = list(hazard_types_count.values())
        pie_labels_count_breach = list(breach_types_count.keys())
        pie_sizes_count_breach = list(breach_types_count.values())

    if report_type == "incidents":
        y = top_margin
        count = len(data)
        pdf.drawString(left_margin, y, "Number of hazards from {} to {}".format(start[:10], end[:10]))
        create_pie_chart(pie_labels_count, pie_sizes_count, "pie_chart.png", width=5, height=8)
        pie_chart_img = ImageReader("pie_chart.png")
        pdf.drawImage(pie_chart_img, 0.3 * inch, 6 * inch, width=3.75 * inch, height=3.75 * inch)
        pdf.setFont("Helvetica-Bold", 12) 

        pdf.drawString(1.0 * inch, 6.2 * inch, "Type of Hazard (%)")
        y_position = 5.7 * inch
        for label, size in zip(pie_labels_count, pie_sizes_count):
            text = f"{label} - {size}"
            pdf.drawString(1.0 * inch, y_position, text)
            y_position += 0.2 * inch

        create_pie_chart(["Resolved", "Unresolved"], [resolved_cases, unresolved_cases], "pie_chart_resolved.png", width=5, height=8)
        pie_chart_img = ImageReader("pie_chart_resolved.png")
        pdf.drawImage(pie_chart_img, 4.3 * inch, 6 * inch, width=3.75 * inch, height=3.75 * inch)
        pdf.setFont("Helvetica-Bold", 12) 
        pdf.drawString(5.1 * inch, 6.2 * inch, "Resolved Cases (%)")  

        y_position = 5.7 * inch
        text = f"Resolved Cases - {resolved_cases}"
        pdf.drawString(5.1 * inch, y_position, text)
        text = f"Unresolved Cases - {unresolved_cases}"
        y_position += 0.2 * inch
        pdf.drawString(5.1 * inch, y_position, text)


        pdf.drawString(0.5 * inch, 4.4 * inch, "AI Analysis")
        draw_text_multiline_within_pdf(pdf, gpt_answer, 0.5 * inch, 4 * inch, 9 * inch, 1 * inch)  

    elif report_type == "breaches":
        y = top_margin
        count = len(data)
        pdf.drawString(left_margin, y, "Number of breaches from {} to {}".format(start[:10], end[:10]))
        create_pie_chart(pie_labels_count_breach, pie_sizes_count_breach, "pie_chart.png", width=5, height=8)
        pie_chart_img = ImageReader("pie_chart.png")
        pdf.drawImage(pie_chart_img, 0.3 * inch, 6 * inch, width=3.75 * inch, height=3.75 * inch)
        pdf.setFont("Helvetica-Bold", 12) 
        pdf.drawString(1 * inch, 6.2 * inch, "Type of Breaches (%)")

        y_position = 5.7 * inch #(5.7 --> 5.5 [0.2 inch per line) ])
        count_list = len(list(zip(pie_labels_count_breach, pie_sizes_count_breach)))
        y_position = y_position - (((0.2 * count_list) - 0.4) * inch)

        for label, size in zip(pie_labels_count_breach, pie_sizes_count_breach):
            text = f"{label} - {size}"
            pdf.drawString(1 * inch, y_position, text)
            y_position += 0.2 * inch


        create_pie_chart(["Resolved", "Unresolved"], [resolved_cases, unresolved_cases], "pie_chart_resolved.png", width=5, height=8)
        pie_chart_img = ImageReader("pie_chart_resolved.png")
        pdf.drawImage(pie_chart_img, 4.55 * inch, 6 * inch, width=3.5 * inch, height=3.75 * inch)
        pdf.setFont("Helvetica-Bold", 12) 
        pdf.drawString(5.6 * inch, 6.2 * inch, "Resolved Breaches (%)")  

        y_position = 5.7 * inch
        text = f"Resolved Cases - {resolved_cases}"
        pdf.drawString(5.6 * inch, y_position, text)
        text = f"Unresolved Cases - {unresolved_cases}"
        y_position += 0.2 * inch
        pdf.drawString(5.2 * inch, y_position, text)


        pdf.drawString(0.5 * inch, 4.4 * inch, "AI Analysis")
        draw_text_multiline_within_pdf(pdf, gpt_answer, 0.5 * inch, 4 * inch, 9 * inch, 1 * inch)      

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

def create_pie_chart(labels, sizes, filename, width=5, height=8):
    # print("Pie Chart Data:")
    # for label, size in zip(labels, sizes):
    #     print(f"{label}: {size}")
    plt.figure(figsize=(width, height))
    wedges, _, _ = plt.pie(sizes, labels=None, autopct='%.0f%%', startangle=90)

    # Draw a legend with labels and percentages on the right
    plt.legend(wedges, labels, loc="center left", title="Labels", bbox_to_anchor=(1, 0.5))

    plt.axis('equal')
    plt.gca().set_aspect('equal') 
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def draw_text_multiline_within_pdf(pdf, text, x, y, max_width, max_height):
    lines = textwrap.wrap(text, width=int(max_width / pdf.stringWidth('A', 'Helvetica', 12)))
    line_height = pdf._leading

    for line in lines:
        if y - line_height < 0:
            break
        pdf.drawString(x, y, line)
        y -= line_height