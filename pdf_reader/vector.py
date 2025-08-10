from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
import PyPDF2

docs_folder = "./documents"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_loc = "./chroma_db"

pdf_files = []
for filename in os.listdir(docs_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(docs_folder, filename)
        text = ""
        with open(pdf_path,"rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
            pdf_files.append((filename,text))
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
documents =[]
for file_name, pdf_text in pdf_files:
    chunks = splitter.split_text(pdf_text)
    for i, chunk in enumerate(chunks):
        documents.append(Document(page_content = chunk, metadata = {"source":file_name}, id=f"{file_name}_{i}"))

add_documents = not os.path.exists(db_loc)
vector_store = Chroma(
    collection_name = "pdf_docs",
    persist_directory = db_loc,
    embedding_function = embeddings
)

if add_documents:
    ids = [doc.id for doc in documents]
    vector_store.add_documents(documents=documents,ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs = {"k":3}
)
