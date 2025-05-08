import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, Response
from src.legal_ai.crew import LegalAi
from docx import Document
from flask_cors import CORS
import json
from pypdf import PdfReader
import pathlib
import tempfile
import re
import os
from langchain_community.chat_message_histories import ChatMessageHistory

app = Flask(__name__)
CORS(app)

session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

def history(session_id: str, query: str, document: str, response: str):
    filename = f"memory/history_{session_id}.json"
    messages = []

    # Load existing history if it exists
    if os.path.exists(filename):
        try:
            with open(filename, "r") as file:
                messages = json.load(file)
        except (json.JSONDecodeError, IOError):
            messages = []

    # Append the new message pair
    messages.append({"query": query, "document": document, "response": response})

    # Save updated history
    with open(filename, "w") as file:
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

def detect_intent(query: str) -> str:
    """
    Determine the task type based on keywords in the query.
    Returns: one of ['drafting', 'modifier', 'findings', 'summary', 'success']
    """
    if not query:  # Handles None, empty string, etc.
        return 'summary'

    query_lower = query.lower()

    drafting_keywords = ["draft", "create", "write", "prepare", "generate", "make"]
    modifier_keywords = ["modify", "edit", "revise", "update", "change", "add", "improve", "adjust", "remove", "ammend"]
    findings_keywords = ["compare", "difference", "similarities", "key findings", "insights", "recommendations", "highlight", "contrast", "insights", "strengths", "weaknesses"]
    prediction_keywords = ["predict", "winning", "percentage", "prediction", "evaluate", "probability", "score"]
    summary_keywords = ["summarize", "summary", "brief", "overview", "key points", "explain", "takeaways", "highlight"]

    if any(word in query_lower for word in summary_keywords):
        return 'summary'
    elif any(word in query_lower for word in prediction_keywords):
        return 'success'
    elif any(word in query_lower for word in findings_keywords):
        return 'findings'
    elif any(word in query_lower for word in modifier_keywords):
        return 'modifier'
    elif any(word in query_lower for word in drafting_keywords):
        return 'drafting'
    else:
        return 'summary'  # Default fallback

session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

@app.route("/", methods=['POST'])
def run():
    """
    Main API endpoint that receives a prompt and optionally a document.
    Passes data to Legal AI Crew for processing and returns the response.
    """
    if request.content_type.startswith('multipart/form-data'):
        files = request.files.getlist('document')
        prompt = request.form.get('query')
        session_id = request.form.get('session_id')

    # Check if the request is of type application/json (e.g., from a frontend app)
    elif request.content_type == 'application/json':  
        files = request.files.getlist('document')  # This may often be None for pure JSON requests
        data = json.loads(request.data.decode('utf-8')) # Load and decode the raw JSON payload
        prompt = data.get('query')
        session_id = data.get('session_id')

    file_content = ""
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            text = extract_text(tmp_file.name)
            file_content += f"\n--- {file.filename} ---\n{text}\n"
    
    chat_history = get_session_history(session_id)
    history_text = "\n".join([
        f"{msg.type.upper()}: {msg.content}" for msg in chat_history.messages
    ])
    # Step 4: Build Input for Crew
    input_data = {
        'query': prompt if prompt else None, 
        'document': file_content if file_content else None,
        'history': history_text
    }

    # Step 5: Run the Crew
    task_name = detect_intent(prompt)
    answer = LegalAi().crew(task_name=task_name).kickoff(inputs=input_data)
    # Just return the raw response
    raw_response = answer.raw.strip()

    text = re.sub(r'\\(usepackage|documentclass|begin|end)\{[^}]*\}', '', raw_response)

    # Convert \textbf{} to **bold**
    text = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', text)

    # Convert \textit{} or \emph{} to *italic*
    text = re.sub(r'\\(textit|emph)\{([^}]*)\}', r'*\2*', text)

    # Convert \item to Markdown list
    text = re.sub(r'\\item\s+', r'- ', text)

    # Remove remaining LaTeX commands (like \hfill, \vspace, etc.)
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?', '', text)

    # Strip excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r"\{\d*\.\d+\}", "", text)
    text = re.sub(r'[{}]', '', text) 
    text = re.sub(r'<[^>]+>', '', text)
    cleaned = re.sub(r"^```(?:[a-zA-Z]*)?|```$", "", text.strip())

    # Step 6: Update History
    if prompt:
        chat_history.add_user_message(prompt)
    chat_history.add_ai_message(cleaned)
    session_id = session_id or "testing"
    # Saving input and response to history
    history( session_id, prompt, file_content, cleaned)
    
    # Return plain text response
    return Response(cleaned, mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=8080)
