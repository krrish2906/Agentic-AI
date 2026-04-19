# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"
PDF_PATH = os.path.join(os.path.dirname(__file__), "knowledge_source.pdf")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + Index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(splits, embedding)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

chain = parallel | prompt | llm | parser

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")

query = input("\nQuery: ")
response = chain.invoke(query.strip())

print("\nResponse:", response)
