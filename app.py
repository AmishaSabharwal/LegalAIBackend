from flask import Flask, request, jsonify
from legal_ai.crew import LegalAi
from docx import Document
import os, json
app = Flask(__name__)

@app.route("/", methods=['POST'])
def run():
    input = request.get_json()
    answer = LegalAi().crew().kickoff(inputs=input)
    response = answer.raw
    return response

if __name__ == "__main__":
    app.run(host="192.168.2.17", port=8000, debug=True)
