import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Initialize the chatbot components
@st.cache_resource
def initialize_chatbot():
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", 
                                     model_kwargs={"trust_remote_code": True, 
                                                  "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """<s>[INST]You are a legal chatbot specializing in the Indian Penal Code (IPC). Your role is to provide accurate, concise, and professional answers strictly based on the user's query.

    Response Guidelines:
    1. For queries about specific IPC sections (e.g., "What is IPC 302?"), provide:
       **📜 Section [X] Details:**
       - [Official title/name of section]
       - [Detailed explanation]
       - **💡 Example:** [Relevant example case]

    2. For scenario-based queries (e.g., "A boy killed a lady..."), provide:
       **📝 Case Analysis:**
       - [Brief summary of the legal position]
       
       **⚖️ Relevant IPC Sections:**
       - [Section X]: [Title/Description]
       - [Section Y]: [Title/Description] (if multiple apply)
       
       **🔨 Legal Consequences:**
       - [Possible punishment/outcome 1]
       - [Possible punishment/outcome 2] (if multiple)

    3. For greetings, respond politely but briefly.
    4. For non-IPC questions, politely decline to answer.
    5. Never include Q&A format in responses.
    6. Never generate follow-up questions.
    7. Always maintain professional tone.

    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER:
    </s>[INST]
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    TOGETHER_AI_API = '488d9538dd3cfbf08816cca9ae559157f252c3daf6356eb4e10dd965ff589ddb'
    llm = Together(model="mistralai/Mistral-7B-Instruct-v0.2", 
                  temperature=0.5, 
                  max_tokens=1024, 
                  together_api_key=TOGETHER_AI_API)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        memory=ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True), 
        retriever=db_retriever, 
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa

qa = initialize_chatbot()

def format_section_response(response_text):
    """Special formatting for section queries"""
    formatted = response_text
    formatting_replacements = [
        ("**📜 Section", "\n**📜 Section"),
        ("**💡 Example:**", "\n**💡 Example:**\n"),
        ("- ", "• "),
        ("IPC Section", "**IPC Section**")
    ]
    for old, new in formatting_replacements:
        formatted = formatted.replace(old, new)
    return formatted

def format_scenario_response(response_text):
    """Special formatting for scenario-based queries"""
    formatted = response_text
    formatting_replacements = [
        ("**📝 Case Analysis:**", "\n**📝 Case Analysis**\n"),
        ("**⚖️ Relevant IPC Sections:**", "\n**⚖️ Relevant IPC Sections**\n"),
        ("**🔨 Legal Consequences:**", "\n**🔨 Legal Consequences**\n"),
        ("- ", "• "),
        ("IPC Section", "**IPC Section**")
    ]
    for old, new in formatting_replacements:
        formatted = formatted.replace(old, new)
    return formatted

def generate_response(user_query):
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    thanks_keywords = ["thank", "thanks", "appreciate"]
    section_keywords = ["section", "ipc", "what is"]
    non_ipc_response = "I specialize only in the Indian Penal Code (IPC) matters. Please ask me about IPC sections or legal scenarios."
    
    if any(greeting in user_query.lower() for greeting in greeting_keywords):
        return "Hello! I'm your IPC legal assistant. How can I help you with the Indian Penal Code today?"
    elif any(thanks in user_query.lower() for thanks in thanks_keywords):
        return "You're welcome! Let me know if you have other IPC-related questions."
    elif "ipc" not in user_query.lower() and not any(term in user_query.lower() for term in ["section", "penal code", "law", "legal"]):
        return non_ipc_response
    else:
        try:
            result = qa.invoke(input=user_query)
            raw_response = result["answer"]
            
            # Determine response type and format accordingly
            if any(keyword in user_query.lower() for keyword in ["what is section", "explain section", "meaning of section"]):
                return format_section_response(raw_response)
            elif "section" in user_query.lower() and ("what" in user_query.lower() or "explain" in user_query.lower()):
                return format_section_response(raw_response)
            else:
                return format_scenario_response(raw_response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return "I encountered an error processing your request. Please try again with a different question."

# Streamlit UI
st.set_page_config(page_title="IPC Legal Chatbot", page_icon="⚖️")
st.title("Indian Penal Code (IPC) Legal Assistant")
st.caption("Ask me about specific IPC sections or describe a legal scenario for analysis")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I'm your IPC legal assistant. You can:\n\n• Ask about specific sections (e.g., 'Explain IPC 302')\n• Describe a scenario for legal analysis (e.g., 'What if someone...')"
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("What is your legal question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = generate_response(prompt)
            
            # Add disclaimer to legal responses only
            if not any(keyword in prompt.lower() for keyword in ["hello", "hi", "hey", "thank", "thanks"]):
                disclaimer = """
                <div style='background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:15px; border-left:4px solid #dc3545;'>
                <small>▲ Legal Disclaimer: This analysis is for informational purposes only and does not constitute legal advice.
                For specific cases, please consult a qualified lawyer.</small>
                </div>
                """
                full_response = f"{disclaimer}\n\n{response}"
            else:
                full_response = response
            
            st.markdown(full_response, unsafe_allow_html=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
