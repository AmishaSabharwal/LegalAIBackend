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
CORS(app)  # Enable CORS to allow requests from frontend clients

# Dictionary to maintain session-specific message histories
session_histories = {}

# Retrieve or initialize a chat message history for a given session ID
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

# Save query-document-response history in a JSON file per session
def history(session_id: str, query: str, document: str, response: str):
    filename = f"memory/history_{session_id}.json"
    messages = []

    # Load existing history if available
    if os.path.exists(filename):
        try:
            with open(filename, "r") as file:
                messages = json.load(file)
        except (json.JSONDecodeError, IOError):
            messages = []

    # Add current interaction
    messages.append({"query": query, "document": document, "response": response})

    # Save updated history
    with open(filename, "w") as file:
        json.dump(messages, file, indent=4)

# Extract text from a .pdf or .docx file
def extract_text(file_path):
    file_ext = pathlib.Path(file_path).suffix
    text = ""

    if file_ext == '.pdf':
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    else:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    return text

# Basic intent detection from user query using keyword matching
def detect_intent(query: str) -> str:
    if not query:
        return 'summary'

    query_lower = query.lower()

    drafting_keywords = ["draft", "create", "write", "prepare", "compose", "generate", "agreement", "petition", "contract", "make"]
    modifier_keywords = ["modify", "edit", "revise", "update", "change", "add", "improve", "adjust"]
    findings_keywords = ["compare", "difference", "similarities", "key findings", "analyze", "insights", "recommendations", "highlight", "contrast", "strengths", "weaknesses"]
    prediction_keywords = ["success", "predict", "chance of winning", "win percentage", "case prediction", "evaluate", "probability", "score", "legal strategy"]
    summary_keywords = ["summarize", "summary", "brief", "overview", "key points", "explain", "takeaways"]

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
        return 'summary'  # Fallback default intent

# Main API route
@app.route("/", methods=['POST'])
def run():
    """
    Receives query and optional document, processes them via LegalAI Crew,
    and returns the structured response.
    """

    # Handle multipart/form-data requests (e.g., file upload via form)
    if request.content_type.startswith('multipart/form-data'):
        files = request.files.getlist('document')
        prompt = request.form.get('query')
        session_id = request.form.get('session_id')

    # Handle pure JSON requests (e.g., from frontend without file upload)
    elif request.content_type == 'application/json':  
        files = request.files.getlist('document')  # Typically None for JSON
        data = json.loads(request.data.decode('utf-8'))
        prompt = data.get('query')
        session_id = data.get('session_id')

    # Extract text from uploaded documents (PDF/DOCX)
    file_content = ""
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            file_content += extract_text(tmp_file.name) + "\n"
    
    # Retrieve session-specific chat history
    chat_history = get_session_history(session_id)
    history_text = "\n".join([
        f"{msg.type.upper()}: {msg.content}" for msg in chat_history.messages
    ])

    # Prepare input data for Legal AI processing
    input_data = {
        'query': prompt,
        'document': file_content if file_content else "",
        'history': history_text
    }

    # Detect the user's intent and select the appropriate task
    task_name = detect_intent(prompt)

    # Execute the appropriate Crew task
    answer = LegalAi().crew(task_name=task_name, session_id=session_id).kickoff(inputs=input_data)

    # Clean and sanitize raw response from AI
    raw_response = answer.raw.strip()
    text = re.sub(r'<[^>]+>', '', raw_response)
    text = re.sub(r'\\(begin|end|usepackage|documentclass|geometry|textbf|Large|vspace|hrulefill|RaggedRight|center)\{.*?\}', '', text)
    cleaned = re.sub(r"^```(?:[a-zA-Z]*)?|```$", "", text.strip())

    # Add this interaction to chat history
    chat_history.add_user_message(prompt)
    chat_history.add_ai_message(cleaned)
    session_id = session_id if session_id else "1234"

    # Save to file-based history
    history(session_id, prompt, file_content, cleaned)

    # Return plain text response
    return Response(cleaned, mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=8080)
