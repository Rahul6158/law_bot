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

    prompt_template = """<s>[INST]You are a legal chatbot specializing in the Indian Penal Code (IPC). Provide professional legal information with these strict guidelines:

    For SCENARIO-BASED QUERIES (e.g., "What if someone..."):
    <legal_analysis>
    <summary>[Clear one-sentence summary]</summary>
    <case_analysis>[Detailed legal analysis of the situation]</case_analysis>
    <applicable_sections>
    - [IPC Section X]: [Title/Purpose]
    - [IPC Section Y]: [Title/Purpose]
    </applicable_sections>
    <consequences>
    - [Possible outcome 1]
    - [Possible outcome 2]
    </consequences>
    </legal_analysis>

    For SECTION EXPLANATIONS (e.g., "Explain IPC 302"):
    <section_details>
    <definition>[Official section title/definition]</definition>
    <explanation>[Detailed explanation in simple terms]</explanation>
    <example>[Practical case example]</example>
    <related_sections>
    - [Related Section A]: [Brief description]
    - [Related Section B]: [Brief description]
    </related_sections>
    </section_details>

    Strict Rules:
    1. NEVER include "Question:" or "Answer:" in responses
    2. NEVER create hypothetical Q&A
    3. For greetings: Brief polite response
    4. For non-IPC queries: Polite redirection
    5. Always maintain professional legal tone

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

def clean_response(text):
    """Remove any Q&A patterns and clean the text"""
    text = text.split("Question:")[0].split("QUESTION:")[0]
    text = text.split("Answer:")[0].split("ANSWER:")[0]
    return text.strip()

def format_scenario_response(text):
    """Format scenario-based answers"""
    text = clean_response(text)
    replacements = [
        ("<legal_analysis>", "<div style='margin-bottom:20px'>"),
        ("</legal_analysis>", "</div>"),
        ("<summary>", "<h4 style='color:#2b5876; margin-bottom:10px'>Summary</h4><p>"),
        ("</summary>", "</p>"),
        ("<case_analysis>", "<h4 style='color:#2b5876; margin-bottom:10px'>Case Analysis</h4><p>"),
        ("</case_analysis>", "</p>"),
        ("<applicable_sections>", "<h4 style='color:#2b5876; margin-bottom:10px'>Applicable IPC Sections</h4><ul>"),
        ("</applicable_sections>", "</ul>"),
        ("<consequences>", "<h4 style='color:#2b5876; margin-bottom:10px'>Legal Consequences</h4><ul>"),
        ("</consequences>", "</ul>"),
        ("- ", "<li>"),
        ("</li><li>", "</li><li>")
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def format_section_response(text):
    """Format section explanation answers"""
    text = clean_response(text)
    replacements = [
        ("<section_details>", "<div style='margin-bottom:20px'>"),
        ("</section_details>", "</div>"),
        ("<definition>", "<h4 style='color:#2b5876; margin-bottom:10px'>Section Definition</h4><p>"),
        ("</definition>", "</p>"),
        ("<explanation>", "<h4 style='color:#2b5876; margin-bottom:10px'>Detailed Explanation</h4><p>"),
        ("</explanation>", "</p>"),
        ("<example>", "<h4 style='color:#2b5876; margin-bottom:10px'>Practical Example</h4><p>"),
        ("</example>", "</p>"),
        ("<related_sections>", "<h4 style='color:#2b5876; margin-bottom:10px'>Related Sections</h4><ul>"),
        ("</related_sections>", "</ul>"),
        ("- ", "<li>"),
        ("</li><li>", "</li><li>")
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def generate_response(user_query):
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    thanks_keywords = ["thank", "thanks", "appreciate"]
    non_ipc_response = "I specialize in Indian Penal Code matters. Please ask about IPC sections or describe a legal scenario."
    
    if any(greeting in user_query.lower() for greeting in greeting_keywords):
        return "Hello! I'm your IPC legal assistant. How may I assist you today?"
    elif any(thanks in user_query.lower() for thanks in thanks_keywords):
        return "You're welcome. For personal legal matters, consulting an advocate is recommended."
    elif "ipc" not in user_query.lower() and not any(term in user_query.lower() for term in ["section", "penal code", "law", "legal"]):
        return non_ipc_response
    else:
        try:
            result = qa.invoke(input=user_query)
            raw_response = clean_response(result["answer"])
            
            # Add legal disclaimer
            disclaimer = """
            <div style='background-color:#f8f9fa; padding:12px; border-radius:6px; margin:15px 0; border-left:4px solid #dc3545;'>
            <p style='font-size:0.9em; color:#555; margin:0;'>Note: This information is for general understanding only. For specific legal advice, please consult a qualified advocate.</p>
            </div>
            """
            
            if any(keyword in user_query.lower() for keyword in ["what is section", "explain section", "meaning of section"]):
                return format_section_response(raw_response) + disclaimer
            elif "section" in user_query.lower() and ("what" in user_query.lower() or "explain" in user_query.lower()):
                return format_section_response(raw_response) + disclaimer
            else:
                return format_scenario_response(raw_response) + disclaimer
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return "I encountered difficulty processing your query. Please try rephrasing your question."

# Streamlit UI
st.set_page_config(page_title="IPC Legal Advisor", page_icon="⚖️")
st.title("Indian Penal Code Legal Advisor")
st.caption("Get professional explanations of IPC sections and scenario analysis")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": """<div style='padding:15px; background-color:#f5f9ff; border-radius:8px; border-left:4px solid #2b5876;'>
                      <h4 style='color:#2b5876; margin-top:0'>How I Can Assist You:</h4>
                      <p>• <b>Section Explanations</b>: "Explain IPC 302" or "What is Section 304?"</p>
                      <p>• <b>Scenario Analysis</b>: Describe a situation for legal assessment</p>
                      <p style='font-size:0.9em; color:#555; margin-bottom:0;'>Note: I provide general legal information, not personal advice.</p>
                      </div>"""
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Describe your legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            response = generate_response(prompt)
            st.markdown(response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
