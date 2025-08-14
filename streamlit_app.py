import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Streamlit UI Configuration - MUST BE FIRST COMMAND
st.set_page_config(page_title="IPC Legal Expert", page_icon="⚖️")

# Initialize the chatbot components
@st.cache_resource
def initialize_chatbot():
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", 
                                     model_kwargs={"trust_remote_code": True, 
                                                  "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt_template = """<s>[INST]You are a legal expert specializing in the Indian Penal Code (IPC). Provide responses based on the query type:

For case analysis (starts with "CASE:"):
SUMMARY:
[Brief legal position]

APPLICABLE SECTIONS:
• [Section X]: [Description]
• [Section Y]: [Description]

CONSEQUENCES:
• [Possible outcome 1]
• [Possible outcome 2]

For section learning (starts with "LEARN:"):
SECTION DEFINITION:
[Official text]

DETAILED EXPLANATION:
[Comprehensive explanation]

EXAMPLE CASE:
[Relevant example]

RELATED SECTIONS:
• [Section A]: [Brief description]
• [Section B]: [Brief description]

Rules:
1. Never use code blocks
2. Never include Q/A format
3. Use proper spacing
4. Maintain professional tone
5. For non-IPC queries, relate to nearest IPC concept

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]"""

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

# Custom CSS for better styling
st.markdown("""
<style>
    .legal-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #2c3e50;
    }
    .legal-header {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .disclaimer {
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffa000;
        margin-top: 20px;
        font-size: 0.9em;
    }
    .tab-button {
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Indian Penal Code (IPC) Legal Expert")
st.caption("Get professional legal analysis and section explanations")

qa = initialize_chatbot()

def format_response(response_text):
    """Format the response with proper styling"""
    # Remove any Q/A patterns
    response_text = response_text.split("Question:")[0].split("ANSWER:")[-1]
    
    # Enhance section headers
    headers = {
        "SUMMARY:": "<div class='legal-card'><div class='legal-header'>📝 SUMMARY</div>",
        "APPLICABLE SECTIONS:": "</div><div class='legal-card'><div class='legal-header'>⚖️ APPLICABLE SECTIONS</div>",
        "CONSEQUENCES:": "</div><div class='legal-card'><div class='legal-header'>🔨 CONSEQUENCES</div>",
        "SECTION DEFINITION:": "<div class='legal-card'><div class='legal-header'>📜 SECTION DEFINITION</div>",
        "DETAILED EXPLANATION:": "</div><div class='legal-card'><div class='legal-header'>📖 DETAILED EXPLANATION</div>",
        "EXAMPLE CASE:": "</div><div class='legal-card'><div class='legal-header'>💡 EXAMPLE CASE</div>",
        "RELATED SECTIONS:": "</div><div class='legal-card'><div class='legal-header'>🔗 RELATED SECTIONS</div>"
    }
    
    for old, new in headers.items():
        response_text = response_text.replace(old, new)
    
    # Convert dashes to bullet points
    response_text = response_text.replace("- ", "• ")
    
    # Close all card divs
    response_text += "</div>" * response_text.count("<div class='legal-card'>")
    
    return response_text.strip()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": """
        <div class='legal-card'>
            <div class='legal-header'>Welcome to IPC Legal Expert</div>
            Start your query with:<br><br>
            • <b>CASE:</b> [describe situation] for legal analysis<br>
            • <b>LEARN:</b> [section number] for section details
        </div>
        """
    })

# Display mode selector
mode = st.radio("Select input mode:", 
                ["Case Analysis", "Section Learning"], 
                horizontal=True,
                key="mode_selector",
                label_visibility="collapsed")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Enter your query..."):
    # Add prefix based on mode
    if mode == "Case Analysis":
        processed_prompt = f"CASE: {prompt}"
    else:
        processed_prompt = f"LEARN: {prompt}"
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing legally..."):
            result = qa.invoke(input=processed_prompt)
            response = format_response(result["answer"])
            disclaimer = """
            <div class='disclaimer'>
            <strong>Legal Disclaimer:</strong> This information is for general understanding only. For personal legal advice, please consult a qualified advocate.
            </div>
            """
            full_response = f"{response}{disclaimer}"
            
            st.markdown(full_response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
