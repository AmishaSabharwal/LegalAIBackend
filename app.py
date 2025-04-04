from flask import Flask, request, jsonify
from legal_ai.crew import LegalAi
from docx import Document
import os, json
app = Flask(__name__)

def history(session_id, query, response):
    messages = []
    if os.path.exists('messages_{session_id}.json'):
        with open('messages_{session_id}.json', 'r') as file:
            messages = json.load(file)

    with open('messages_{session_id}.json', 'w') as file:
        json.dump(messages, file, indent = 4)
        
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

    if not prompt:
        return jsonify({"error": "Query is required"}), 400

    file_content = extract_text_from_docx(file_path) if file_path else None

    input_data = {'query': prompt}
    if file_content:
        input_data = {
            'query': prompt,
            'document':file_content
        }

    answer = LegalAi().crew().kickoff(inputs=input_data)
    response = answer.raw

    return response

if __name__ == "__main__":
    app.run(host="192.168.2.17", port=8000, debug=True)
