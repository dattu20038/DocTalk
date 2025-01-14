from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm  # Progress bar library

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    print("Loading PDF files...")
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} PDF pages.")
    return documents

documents = load_pdf_files(data=DATA_PATH)

# Step 2: Create Chunks with Progress Tracking
def create_chunks(extracted_data):
    print("\nCreating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = list(tqdm(text_splitter.split_documents(extracted_data), 
                            desc="Splitting documents", 
                            total=len(extracted_data)))
    print(f"Created {len(text_chunks)} text chunks.")
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Create Vector Embeddings
def get_embedding_model():
    print("\nLoading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded.")
    return embedding_model

embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
print("\nStoring embeddings in FAISS...")
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"Embeddings saved locally at '{DB_FAISS_PATH}'.")
