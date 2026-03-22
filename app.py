# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from groq import Groq

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Try local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

try:
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("FAISS Database loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS db: {e}")
    db_retriever = None

chat_histories = {}

def get_context(query):
    if db_retriever:
        try:
            docs = db_retriever.get_relevant_documents(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "No local context available."
    return "No local context available."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pdf")
def serve_pdf():
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ipc_law.pdf')
    if os.path.exists(pdf_path):
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'ipc_law.pdf')
    return "PDF not found", 404

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    context = get_context(user_message)
    
    if session_id not in chat_histories:
        chat_histories[session_id] = []
        
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_histories[session_id][-4:]])

    prompt = f"""You are a legal chatbot specializing in the Indian Penal Code (IPC). Your role is to provide accurate, concise, and professional answers based on the user’s query.

If the user is simply greeting you (e.g., "hi", "hello", "good morning") or asking a general conversational question, respond naturally and warmly without using the formal structure below. Only use the structured format for legal inquiries.

Response Format (For legal queries ONLY):
For every relevant question, structure the answer as follows:

Summary:
A brief, clear explanation directly addressing the user’s query.

Sections Applicable:
List the relevant IPC sections by number and title.

Consequences:
Outline the possible punishments, penalties, or legal outcomes.

Response Guidelines:
- Only answer questions related to the Indian Penal Code.
- Base responses on the user’s query and relevant IPC context.
- Keep answers concise. Use plain language.
- Use the given context when applicable.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {user_message}
ANSWER:"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
              {
                "role": "user",
                "content": prompt
              }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        
        answer = completion.choices[0].message.content
        
        chat_histories[session_id].append({"role": "User", "content": user_message})
        chat_histories[session_id].append({"role": "Bot", "content": answer})
        
        full_response = answer + "\n\n*▲ Note: This AI-generated response is for reference purposes and should not replace professional legal advice.*"
        return jsonify({"response": full_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
