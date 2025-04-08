from flask import Flask, request, jsonify
from src.legal_ai.crew import LegalAi
from docx import Document
from flask_cors import CORS
import json
import os 
from pypdf import PdfReader
import pathlib
import tempfile

app = Flask(__name__)

CORS(app)

def history(query, response):
    """
    Appends a new query-response pair to the chat history stored in a JSON file.
    """
    messages = []

    try:
        # Load existing history from file, if available
        if os.path.exists("history.json"):
            with open("history.json", "r") as file:
                messages = json.load(file)
    except (json.JSONDecodeError, IOError):
        # If file is corrupted or unreadable, start fresh
        messages = []

    # Append the latest interaction
    messages.append({"query": query, "response": response})

    # Save updated history
    with open("history.json", "w") as file:
        json.dump(messages, file, indent=4)

def extract_text(file_path):
    """
    Extracts and returns text content from a .docx or .pdf file.

    Args:
        file_path (str): Path to the uploaded document.

    Returns:
        str: Extracted text content.
    """
    file_ext = pathlib.Path(file_path).suffix
    text = ""
    
    if file_ext == '.pdf':
        reader = PdfReader(file_path)
        # Loop through each page in the PDF
        for page in reader.pages:
            page_text = page.extract_text()  # Extract text from current page
            if page_text:
                text += page_text + "\n"  # Append text with newline between pages
    else:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])  # Combine all paragraphs

    return text

@app.route("/", methods=['POST'])
def run():
    """
    Main API endpoint that receives a prompt and optionally a document.
    Passes data to Legal AI Crew for processing and returns the response.
    """
    # data = request.get_json()
    # file_path = data.get('document')  # Path to document (optional)
    # prompt = data.get('query')        # User query
    
    file_path = request.files.get('document')  # Path to document (optional)
    prompt = request.form.get('query')  

    file_content = None

    if file_path:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file_path.filename).suffix) as tmp_file:
            file_path.save(tmp_file.name)
            file_path = tmp_file.name

        file_content = extract_text(file_path)
    # Extract text from the provided document if available
    file_content = extract_text(file_path) if file_path else None
    if not prompt and not file_content:
        return jsonify({"Hello! How can I help you?"}), 400

    input_data = {
        'query': prompt if prompt else None,
        'document': file_content if file_content else None
    }
        
    answer = LegalAi().crew().kickoff(inputs=input_data)
 
    response = answer.raw  # Get raw response text

    # Save the interaction to history
    history(input_data, response)

    return response  # Send back the AI's response

if __name__ == "__main__":
    app.run(host="192.168.2.17", debug=True)