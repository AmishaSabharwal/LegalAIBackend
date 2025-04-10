from flask import Flask, request, Response
from src.legal_ai.crew import LegalAi
from docx import Document
from flask_cors import CORS
import json
import os 
from pypdf import PdfReader
import pathlib
import tempfile
import re

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

    # Check if the request contains form data with files (like from an HTML form upload)
    if request.content_type.startswith('multipart/form-data'):
        file_path = request.files.get('document')
        prompt = request.form.get('query')

    # Check if the request is of type application/json (e.g., from a frontend app)
    elif request.content_type == 'application/json':  
        file_path = request.files.get('document')  # This may often be None for pure JSON requests
        data = json.loads(request.data.decode('utf-8')) # Load and decode the raw JSON payload
        prompt = data.get('query')

    # Initialize a variable to hold the extracted text from the file
    file_content = None

    if file_path:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file_path.filename).suffix) as tmp_file:
            file_path.save(tmp_file.name)
            file_path = tmp_file.name

        file_content = extract_text(file_path)
        
    # Extract text from the provided document if available
    file_content = extract_text(file_path) if file_path else None

    # creating input dict for crew
    input_data = {
        'query': prompt if prompt else None, 
        'document': file_content if file_content else None
    }
        
    # starting the crew to work on its assigned tasks
    answer = LegalAi().crew().kickoff(inputs=input_data)
    
    # Just return the raw response
    raw_response = answer.raw.strip()

    # Remove markdown code fences
    cleaned = re.sub(r"^```(?:[a-zA-Z]*)?|```$", "", raw_response.strip())

    # Save to history
    history(input_data, cleaned)

    # Return plain text response
    return Response(cleaned, mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="192.168.2.17", debug=True)
