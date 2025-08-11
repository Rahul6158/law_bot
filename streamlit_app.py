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

    Response Format:
    For every relevant question, structure the answer as follows:

    **Summary:**
    [A brief, clear explanation directly addressing the user's query]

    **Sections Applicable:**
    - [IPC Section X]: [Title/Description]
    - [IPC Section Y]: [Title/Description] (if multiple apply)

    **Consequences:**
    - [Punishment/Penalty 1]
    - [Punishment/Penalty 2] (if multiple)

    For section explanations:
    **Section [X] Explanation:**
    [Detailed explanation of the section]
    **Example:** [Relevant example]

    Response Guidelines:
    1. Only answer questions related to the Indian Penal Code.
    2. Never generate follow-up questions or hypothetical scenarios.
    3. If asked about a specific section, provide detailed explanation with example.
    4. For greetings, respond politely but briefly and ask how you can assist with IPC.
    5. For non-IPC questions, politely decline to answer.
    6. Never include Q&A format in responses - only provide direct information.

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

# Custom function to format the response with proper markdown
def format_response(response_text):
    # Remove any Q&A patterns that might have slipped through
    response_text = response_text.split("Question:")[0].split("QUESTION:")[0]
    
    # Enhance markdown formatting
    formatted_response = response_text.replace("**Summary:**", "\n**📝 Summary**\n")
    formatted_response = formatted_response.replace("**Sections Applicable:**", "\n**⚖️ Sections Applicable**\n")
    formatted_response = formatted_response.replace("**Consequences:**", "\n**🔨 Consequences**\n")
    formatted_response = formatted_response.replace("**Section", "\n**📜 Section")
    formatted_response = formatted_response.replace("**Example:**", "\n**💡 Example:**")
    
    return formatted_response

# Streamlit UI
st.set_page_config(page_title="IPC Legal Chatbot", page_icon="⚖️")
st.title("Indian Penal Code (IPC) Legal Assistant")
st.caption("Ask me anything about Indian Penal Code and I'll provide relevant sections and consequences")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("What is your legal question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Handle greetings and non-IPC questions
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(greeting in prompt.lower() for greeting in greeting_keywords):
        response = "Hello! I'm your IPC legal assistant. How can I help you with the Indian Penal Code today?"
    elif "ipc" not in prompt.lower() and not any(term in prompt.lower() for term in ["section", "penal code", "law", "legal"]):
        response = "I specialize only in the Indian Penal Code (IPC) matters. Please ask me questions related to IPC sections, crimes, or legal consequences under Indian law."
    else:
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Researching IPC..."):
                try:
                    result = qa.invoke(input=prompt)
                    raw_response = result["answer"]
                    formatted_response = format_response(raw_response)
                    
                    disclaimer = """<div style='background-color:#f8f9fa; padding:10px; border-radius:5px; margin-bottom:15px; border-left:4px solid #dc3545;'>
                                    <small>▲ Note: This information is provided by an AI model for general reference only. 
                                    For serious legal matters, please consult a qualified lawyer.</small>
                                    </div>"""
                    
                    response = f"{disclaimer}\n\n{formatted_response}"
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response = "I encountered an error while processing your request. Please try again with a different question or rephrase your query."
    
    # Display and store the response
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
