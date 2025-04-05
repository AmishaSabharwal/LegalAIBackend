from flask import Flask, request, jsonify
from src.legal_ai.crew import LegalAi
from docx import Document
from flask_cors import CORS
import json
import os 

app = Flask(__name__)

CORS(app)

def history(query, response):
    messages = []

    # Load existing history from file
    if os.path.exists("history.json"):
        with open("history.json", "r") as file:
            messages = json.load(file)

    # Append the new query-response pair
    messages.append({"query": query, "response": response})

    # Save updated history back to file
    with open('history.json', 'w') as file:
        json.dump(messages, file, indent=4)
    
def extract_text_from_docx(file_path):
    """Extracts text from a .docx file."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

@app.route("/", methods=['POST'])
def run():
    data = request.get_json()
    file_path = data.get('document')
    prompt = data.get('query')

    file_content = extract_text_from_docx(file_path) if file_path else None

    if file_content:
        input_data = {
            'query': prompt,
            'document':file_content
        }
    else: 
        input_data = {'query': prompt}
            
    answer = LegalAi().crew().kickoff(inputs=input_data)
    response = answer.raw
    history(input_data, response)
    return response

if __name__ == "__main__":
    app.run(host="192.168.2.17", debug=True)
