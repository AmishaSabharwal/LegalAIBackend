from flask import Flask, request
from legal_ai.crew import LegalAi
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

app = Flask(__name__)

def draft_file(text, filename="sample.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y_position = height - 50  # Start from the top

    max_width = width - 100  # Adjust to avoid text cutoff

    for line in text.split("\n"):
            words = line.split()
            current_line = ""

            for word in words:
                temp_line = f"{current_line} {word}"
                text_width = stringWidth(temp_line, "Helvetica", 12)

                if text_width <= max_width:
                    current_line = temp_line
                else:
                    c.drawString(50, y_position, current_line)
                    y_position -= 15  # Move to next line
                    current_line = word  # Start new line with the current word

                if y_position < 50:  # Start a new page if needed
                    c.showPage()
                    y_position = height - 50

            c.drawString(50, y_position, current_line)  # Draw remaining text
            y_position -= 12
    c.save()
    
def history(session_id, query, response):
    with open('history_{session_id}', '')

@app.route("/", methods=['POST'])
def run():
    input= request.get_json()
    
    answer = LegalAi().crew().kickoff(inputs=input)
    response = answer.raw
    draft_file(response)
    return response

if __name__ == "__main__":
    app.run(debug=True, port=8000)
