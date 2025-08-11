#app.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import telebot

# Setup your Telegram bot
bot = telebot.TeleBot('8335903349:AAHkJmP2CybtuuyBgnoOvTJVnR5hUfEpnT4')

# Setup your chatbot components
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = """<s>[INST]You are a legal chatbot specializing in the Indian Penal Code (IPC). Your role is to provide accurate, concise, and professional answers strictly based on the user’s query.

Response Format:
For every relevant question, structure the answer as follows:

Summary:

A brief, clear explanation directly addressing the user’s query.

Sections Applicable:

List the relevant IPC sections by number and title.

If multiple sections apply, list them in bullet points.

Consequences:

Outline the possible punishments, penalties, or legal outcomes under the relevant IPC sections.

Keep it short and easy to understand.

Response Guidelines:

Scope & Accuracy

Only answer questions related to the Indian Penal Code.

Base responses on the user’s query and relevant IPC context.

Do not create or assume additional questions or scenarios.

Brevity & Clarity

Keep answers concise, avoiding unnecessary elaboration unless needed for clarity.

Use plain language for non-legal users, but remain accurate.

Context Usage

Use the given context or knowledge base when applicable.

If the question is outside the provided context, rely solely on your own IPC knowledge — do not depend on chat history.

Interaction Rules

Do not ask the user follow-up questions.

Maintain a professional and neutral tone at all times.

Objective:
Deliver precise, legally accurate, and contextually relevant IPC information in a consistent, professional manner, always including Sections Applicable and Consequences in the answer when relevant.


CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

TOGETHER_AI_API = '488d9538dd3cfbf08816cca9ae559157f252c3daf6356eb4e10dd965ff589ddb'
llm = Together(model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.5, max_tokens=1024, together_api_key=TOGETHER_AI_API)

qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True), retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})

# Define the start command handler
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "I'm the AI lawyer. Ask me your query.")

# Define the message handler
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    input_prompt = message.text

    # Generate the response
    result = qa.invoke(input=input_prompt)
    response = "▲ Note: Information provided may be inaccurate.\n\n" + "".join(result["answer"])

    # Send the response back to the user
    bot.send_message(message.chat.id, response)

# Start the bot
bot.polling()
