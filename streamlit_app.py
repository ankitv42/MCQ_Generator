import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up OpenAI API Key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # Store in Streamlit secrets for security
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI Model
openai_model = ChatOpenAI(model="gpt-4")

# Streamlit UI
st.title("📘 AI-Powered MCQ Generator")
st.write("Upload a **PDF book or document**, and the AI will generate 12 Multiple Choice Questions.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and split PDF
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    text_data = "\n\n".join([doc.page_content for doc in splits[:5]])  # First 5 chunks

    # Define prompt
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        You are a highly experienced exam question setter. Please generate **exactly 12 multiple-choice questions** from the given text.
        - Each question should have 4 options: A, B, C, D.
        - Clearly mention the correct answer at the end in this format: **Answer: <Correct Option>**.

        TEXT:
        {text}
        """
    )

    # Define chain
    parser = StrOutputParser()
    chain = prompt | openai_model | parser

    # Generate MCQs
    with st.spinner("Generating MCQs..."):
        mcqs = chain.invoke({"text": text_data})

    # Display Output
    st.subheader("Generated MCQs:")
    st.write(mcqs)
