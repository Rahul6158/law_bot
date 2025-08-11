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

    Summary:

    A brief, clear explanation directly addressing the user's query.

    Sections Applicable:

    List the relevant IPC sections by number and title.

    If multiple sections apply, list them in bullet points.

    Consequences:

    Outline the possible punishments, penalties, or legal outcomes under the relevant IPC sections.

    Keep it short and easy to understand.

    Response Guidelines:

    Scope & Accuracy

    Only answer questions related to the Indian Penal Code.

    Base responses on the user's query and relevant IPC context.

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
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is your legal question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Researching IPC..."):
            result = qa.invoke(input=prompt)
            response = "▲ Note: Information provided may be inaccurate.\n\n" + "".join(result["answer"])
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
