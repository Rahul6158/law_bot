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

    prompt_template = """<s>[INST]You are a legal expert specializing in the Indian Penal Code (IPC). Provide responses in this exact format:

For scenario-based questions (e.g., "What if someone..."):
SUMMARY:
[Clear 1-2 sentence overview of the legal position]

CASE ANALYSIS:
[Detailed analysis of the legal situation in clear paragraphs]

APPLICABLE SECTIONS:
• [Section X]: [Title/Description]
• [Section Y]: [Title/Description]

CONSEQUENCES:
• [Possible outcome 1]
• [Possible outcome 2]

For section explanations (e.g., "Explain IPC 302"):
SECTION DEFINITION:
[Official text of the section]

DETAILED EXPLANATION:
[Comprehensive explanation in simple terms]

EXAMPLE CASE:
[Relevant case example with details]

RELATED SECTIONS:
• [Section A]: [Brief description]
• [Section B]: [Brief description]

Rules:
1. Never use code blocks or monospace
2. Never include "Question:" or "Answer:" 
3. Never generate follow-up questions
4. Use proper spacing between sections
5. Maintain professional legal tone
6. For non-IPC questions, respond: "I specialize in Indian Penal Code matters"

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

qa = initialize_chatbot()

def enhance_response(response_text):
    """Clean and format the response text"""
    # Remove any Q/A patterns
    response_text = response_text.split("Question:")[0].split("ANSWER:")[-1]
    
    # Enhance section headers
    headers = {
        "SUMMARY:": "📝 SUMMARY",
        "CASE ANALYSIS:": "🔍 CASE ANALYSIS",
        "APPLICABLE SECTIONS:": "⚖️ APPLICABLE SECTIONS",
        "CONSEQUENCES:": "🔨 CONSEQUENCES",
        "SECTION DEFINITION:": "📜 SECTION DEFINITION",
        "DETAILED EXPLANATION:": "📖 DETAILED EXPLANATION",
        "EXAMPLE CASE:": "💡 EXAMPLE CASE",
        "RELATED SECTIONS:": "🔗 RELATED SECTIONS"
    }
    
    for old, new in headers.items():
        response_text = response_text.replace(old, f"\n\n<strong>{new}</strong>\n")
    
    # Convert dashes to proper bullet points
    response_text = response_text.replace("- ", "• ")
    
    return response_text.strip()

# Streamlit UI Configuration
st.set_page_config(page_title="IPC Legal Expert", page_icon="⚖️")
st.title("Indian Penal Code (IPC) Legal Expert")
st.caption("Get professional legal analysis of IPC sections and scenarios")

# Custom CSS for better styling
st.markdown("""
<style>
    .legal-disclaimer {
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffa000;
        margin-top: 15px;
        font-size: 0.9em;
    }
    strong {
        color: #2c3e50;
    }
    .chat-container {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Welcome to the IPC Legal Expert. You can ask about:\n\n• Specific IPC sections (e.g., 'Explain Section 302')\n• Legal scenarios (e.g., 'What if someone...')"
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input and processing
if prompt := st.chat_input("Ask your legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            if "ipc" not in prompt.lower() and not any(term in prompt.lower() for term in ["section", "penal code", "law", "legal"]):
                response = "I specialize in Indian Penal Code matters. Please ask about IPC sections or legal scenarios."
            else:
                result = qa.invoke(input=prompt)
                response = enhance_response(result["answer"])
                disclaimer = """
                <div class='legal-disclaimer'>
                <strong>Legal Disclaimer:</strong> This analysis is for informational purposes only and does not constitute legal advice. For specific cases, please consult a qualified advocate.
                </div>
                """
                response += disclaimer
            
            st.markdown(response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
