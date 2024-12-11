import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from prompts import contextualize_q_prompt, qa_prompt
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

# Load environment
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("No GROQ_API_KEY found. Please set it in your environment.")
    st.stop()

st.title("Chat with webpages!")

# Initialize the model only once 
# Oh YEaaahhHhhhhh!!! Optimizzzeedddd CODEEEeeeEEEeeeEEe
@st.cache_resource
def initialize_llm(api_key: str):
    return ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")

llm = initialize_llm(groq_api_key)

def _get_session():
    """Gets the session ID """
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session

# History management
if "session_id" not in st.session_state:
    st.session_state.session_id = _get_session()

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Returns message history """
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

@st.cache_resource
def load_and_process_documents(url: str):
    """Load documents from URL, create embeddings and retriever."""
    st.info("Loading webpage...")
    loader = WebBaseLoader(url)
    docs = loader.load()

    st.info("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    st.info("Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(documents, embeddings)
    retriever = vectors.as_retriever()

    return retriever

def initialize_chains(llm, retriever):
    """Create the history-aware retriever and RAG chain."""
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

url = st.text_input(label="Enter the URL")
go = st.button(label="Go")

# UI

if go and url:
    # Regenerate the retriever if new URL or not cached
    try:
        st.session_state.retriever = load_and_process_documents(url)
        st.session_state.initialized = True
        st.success("Vector Store created and ready!")
    except Exception as e:
        st.error(f"Failed to load or process the URL. Error: {e}")
        st.session_state.initialized = False

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# A fail-safe
if st.session_state.initialized and "retriever" not in st.session_state:
    st.session_state.retriever = load_and_process_documents(url)

if st.session_state.initialized and "conversational_rag_chain" not in st.session_state:
    st.session_state.conversational_rag_chain = initialize_chains(llm, st.session_state.retriever)


history = get_session_history(st.session_state.session_id)
for msg in history.messages:
    if isinstance(msg, HumanMessage):
        role = "user"
    else:
        role = "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if st.session_state.initialized:
    user_input = st.chat_input("Enter your question")
    if user_input and "conversational_rag_chain" in st.session_state:
        start_time = time.process_time()
        response = st.session_state.conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        end_time = time.process_time()

        st.write("Response Time:", end_time - start_time)
        st.write(response["answer"])
