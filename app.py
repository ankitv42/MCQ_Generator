import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.colab import files
import tempfile

# Load API key securely (Do this manually or via Google Colab secrets)
os.environ["OPENAI_API_KEY"] = "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"

# Initialize OpenAI Model
openai_model = ChatOpenAI(model="gpt-4")

# Upload file
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Save file temporarily
with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as temp_file:
    temp_file.write(uploaded[filename])
    temp_file_path = temp_file.name

# Load and process PDF
loader = PyPDFLoader(temp_file_path)
docs = loader.load()

# Convert documents into a single text block
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
text_data = "\n\n".join([doc.page_content for doc in splits[:5]])  # Limiting to first few chunks

# Define Prompt
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
mcqs = chain.invoke({"text": text_data})

# Print output
print(mcqs)
