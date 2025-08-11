import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import re

# ------------------ Initialize Chatbot ------------------ #
@st.cache_resource
def initialize_chatbot():
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"trust_remote_code": True,
                          "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
        )

        # Load FAISS database
        db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
        db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Prompt template
        prompt_template = """
You are a legal chatbot specializing in the Indian Penal Code (IPC).

📌 **Your Role:**
- If the user greets (e.g., "Hi", "Hello"), greet back warmly and offer assistance.
- If the query is related to IPC, answer using the structured format below.
- If unrelated to IPC, respond with: 
  "⚠️ I can only answer questions related to the Indian Penal Code (IPC)."

📌 **Response Structure (for IPC queries):**
**📄 Summary:**  
A short, clear explanation answering the question.

**📜 Sections Applicable:**  
List relevant IPC sections by number & title in bullet points.

**⚠️ Consequences:**  
List punishments, penalties, or legal outcomes.

📌 **Special Case:**
If the question is like "What is Section X?", explain the section in plain words and give a real-life example.

📌 **Rules:**
- No Q&A style.
- Keep language simple but accurate.
- Always follow the structure above.

Context: {context}  
Chat History: {chat_history}  
Question: {question}  
Answer:
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=['context', 'question', 'chat_history']
        )

        # Together AI LLM
        TOGETHER_AI_API = '488d9538dd3cfbf08816cca9ae559157f252c3daf6356eb4e10dd965ff589ddb'
        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=0.5,
            max_tokens=1024,
            together_api_key=TOGETHER_AI_API
        )

        # Conversational Retrieval Chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True),
            retriever=db_retriever,
            combine_docs_chain_kwargs={'prompt': prompt}
        )
        return qa

    except Exception as e:
        st.error(f"⚠️ Error initializing chatbot: {e}")
        return None


# ------------------ Streamlit App ------------------ #
qa = initialize_chatbot()

st.set_page_config(page_title="IPC Legal Chatbot", page_icon="⚖️")
st.title("⚖️ Indian Penal Code (IPC) Legal Assistant")
st.caption("Ask me anything about the Indian Penal Code — I'll provide relevant sections, consequences, and explanations.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("What is your legal question?"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        with st.chat_message("assistant"):
            with st.spinner("Researching IPC..."):
                result = qa.invoke(input=user_input)
                raw_answer = result.get("answer", "").strip()

                # Add disclaimer at the top
                response = (
                    "ℹ️ **Note:** This information is provided by an AI model. For serious matters, please consult a licensed lawyer.\n\n"
                    + raw_answer
                )

                # Improve heading formatting
                response = re.sub(r"(?i)\bSummary\b", "**📄 Summary**", response)
                response = re.sub(r"(?i)\bSections Applicable\b", "**📜 Sections Applicable**", response)
                response = re.sub(r"(?i)\bConsequences\b", "**⚠️ Consequences**", response)

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        error_msg = f"⚠️ Unable to process your query right now: {e}"
        st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
