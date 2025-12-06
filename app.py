import streamlit as st
from streamlit_server_state import server_state, server_state_lock
import json
from streamlit.web import cli as stcli
import sys
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Customer Support AI", page_icon="🤖")

# 2. SIDEBAR - FILE UPLOAD
with st.sidebar:
    st.title("📂 Knowledge Base")
    st.markdown("Upload your **User Manual** or **Solution Guide** here.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    
    api_key = st.text_input("Enter OpenAI API Key", type="password")

# 3. MAIN APPLICATION LOGIC
st.header("🤖 AI Support Portal")

if pdf_file is not None and api_key:
    
    # A. READ THE PDF
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # B. SPLIT TEXT INTO CHUNKS
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # C. CREATE EMBEDDINGS
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # D. USER INTERFACE FOR QUESTIONS
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        docs = vector_store.similarity_search(user_question)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
        
        context_text = "\n".join([doc.page_content for doc in docs])
        prompt = f"Based on this context: {context_text}\n\nAnswer this question: {user_question}"
        
        with st.spinner("Analyzing document..."):
            response = llm.invoke(prompt)
            
        st.write("### Answer:")
        st.write(response.content)
        
        with st.expander("See Source Context"):
            for doc in docs:
                st.write(doc.page_content)

elif not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")
else:
    st.info("Please upload a PDF document to start.")
    
# Create API endpoint
if st.query_params.get("api") == "true":
    # API mode - handle POST requests
    import os
    from pathlib import Path
    
    def get_response(prompt):
        # Your AI logic here
        llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Process prompt and return response
        response = llm.invoke(prompt)
        return response.content
    
    # Read POST data
    data = st.query_params
    if "prompt" in data:
        response = get_response(data["prompt"])
        st.json({"response": response})
else:
    # Normal Streamlit UI
    st.title("AI Assistant")
    # Your existing UI code
