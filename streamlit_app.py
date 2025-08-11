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

    prompt_template = """<s>[INST]You are a legal expert specializing in the Indian Penal Code (IPC). Provide professional legal information in this format:

    For scenario questions (e.g., "What if someone..."):
    <div style="background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px">
    <strong style="color:#2c3e50">📝 Summary</strong><br>
    [Concise 1-2 sentence summary of the legal position]
    </div>

    <div style="background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px">
    <strong style="color:#2c3e50">🔍 Case Analysis</strong><br>
    [Detailed analysis of the legal situation]
    </div>

    <div style="background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px">
    <strong style="color:#2c3e50">⚖️ Applicable IPC Sections</strong><br>
    • <strong>Section [X]:</strong> [Title/Description]<br>
    • <strong>Section [Y]:</strong> [Title/Description]
    </div>

    <div style="background:#f8f9fa;padding:10px;border-radius:5px">
    <strong style="color:#2c3e50">🔨 Legal Consequences</strong><br>
    • [Possible outcome 1]<br>
    • [Possible outcome 2]
    </div>

    For section questions (e.g., "Explain IPC 302"):
    <div style="background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px">
    <strong style="color:#2c3e50">📜 Section [X] Definition</strong><br>
    [Official definition from IPC]
    </div>

    <div style="background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px">
    <strong style="color:#2c3e50">📖 Detailed Explanation</strong><br>
    [Comprehensive explanation in simple terms]
    </div>

    <div style="background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px">
    <strong style="color:#2c3e50">💡 Practical Example</strong><br>
    [Relevant case example]
    </div>

    <div style="background:#f8f9fa;padding:10px;border-radius:5px">
    <strong style="color:#2c3e50">🔗 Related Sections</strong><br>
    • <strong>Section [A]:</strong> [Brief description]<br>
    • <strong>Section [B]:</strong> [Brief description]
    </div>

    Rules:
    1. Never include Q&A format in responses
    2. Never generate follow-up questions
    3. Always maintain professional tone
    4. For non-IPC questions, respond: "I specialize in Indian Penal Code matters"
    5. Remove any "Question:" or "Answer:" text from responses

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

def clean_response(response_text):
    """Remove any Q&A patterns and clean the response"""
    # Remove any Question/Answer patterns
    response_text = response_text.split("Question:")[0].split("QUESTION:")[0]
    response_text = response_text.split("Answer:")[-1].split("ANSWER:")[-1]
    
    # Remove redundant section headers
    response_text = response_text.replace("Summary:", "").replace("Case Analysis:", "")
    return response_text.strip()

def generate_response(user_query):
    # Skip exception handling for legal scenarios as requested
    result = qa.invoke(input=user_query)
    return clean_response(result["answer"])

# Streamlit UI
st.set_page_config(page_title="IPC Legal Expert", page_icon="⚖️")
st.title("Indian Penal Code (IPC) Legal Expert")
st.caption("Get professional legal analysis of IPC sections and scenarios")

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
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": """
        <div class='legal-card'>
            <div class='legal-header'>Welcome to IPC Legal Expert</div>
            I can provide professional legal analysis of:<br><br>
            • <strong>IPC Sections</strong> (e.g., "Explain Section 302")<br>
            • <strong>Legal Scenarios</strong> (e.g., "What if someone...")<br><br>
            Ask your legal question below.
        </div>
        """
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask your legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your legal query..."):
            if "ipc" not in prompt.lower() and not any(term in prompt.lower() for term in ["section", "penal code", "law", "legal"]):
                response = "I specialize in Indian Penal Code matters. Please ask about IPC sections or legal scenarios."
            else:
                response = generate_response(prompt)
                response += """
                <div class='disclaimer'>
                <strong>Legal Disclaimer:</strong> This information is for general understanding only. For personal legal advice, please consult a qualified advocate.
                </div>
                """
            
            st.markdown(response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
