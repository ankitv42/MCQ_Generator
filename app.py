import os
import langchain
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from google.colab import files
import tempfile
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["GROQ_API_KEY"] = "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"

uploaded = files.upload()

filename = list(uploaded.keys())[0]
file_content = uploaded[filename]

with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as temp_file:
    temp_file.write(file_content)
    temp_file_path = temp_file.name
	
loader = PyPDFLoader(temp_file_path)
docs = loader.load()

model = ChatGroq(model="llama-3.1-8b-instant")

model_name="BAAI/bge-small-en"

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf_embeddings=HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore=FAISS.from_documents(documents=splits, embedding=hf_embeddings)
retriever=vectorstore.as_retriever()

prompt = PromptTemplate(
    input_variables=["text"],
    template="""You are a highly experienced high school teacher who has been preparing Multiple Choice Questions (MCQs) for annual student exams for many years. I need your help in preparing **exactly 12 high-quality MCQs** (each with 4 options) for the given {text}.

**Formatting Guidelines:**
- Do not include any explanations.
- The question should be followed by four answer options (A, B, C, D).
- The correct answer should be mentioned at the end in this format: **Answer: <Correct Option>**.
"""
)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"text": docs}))
