import streamlit as st
from langchain.chains import RetrievalQAChain
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit app title and description
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# Input for OpenAI API Key
api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Check if OpenAI API key is provided
if api_key:
    # Initialize OpenAI LLM
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4")

    # Session ID input
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize HuggingFace Embeddings
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Statefully manage chat history using Streamlit's session state
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # File uploader to upload PDF files
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    # Process the uploaded PDFs
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # System prompt for contextualizing the question based on chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Standard question-answer chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create the RetrievalQA chain
        question_answer_chain = RetrievalQAChain(llm, retriever, qa_prompt)

        # Function to retrieve or create a chat history based on session ID
        def get_session_history(session: str):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Set up conversational chain
        def conversational_chain(input_text, session_id):
            session_history = get_session_history(session_id)
            response = question_answer_chain.invoke(
                {"input": input_text, "chat_history": session_history.messages}
            )
            session_history.add_user_message(input_text)
            session_history.add_ai_message(response["answer"])
            return response['answer']

        # Input box for the user to ask a question
        user_input = st.text_input("Your question:")
        if user_input:
            response = conversational_chain(user_input, session_id)
            st.write("Assistant:", response)
            st.write("Chat History:", st.session_state.store[session_id].messages)

else:
    st.warning("Please enter the OpenAI API Key.")
