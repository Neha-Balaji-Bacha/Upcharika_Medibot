# Builds FAISS index from PDFs.
import os
import warnings
import time  # ⬅️ Execution time measure karne ke liye
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Suppress warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(DB_FAISS_PATH, exist_ok=True)

def load_pdf_files(data_path):
    print("📂 Loading PDFs from:", data_path)
    pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError("⚠ No PDFs found in data directory!")

    documents = []
    start_time = time.time()  # 🕒 Start Time

    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as doc:
                print(f"🔍 Processing: {os.path.basename(pdf_path)}")
                for page_num, page in enumerate(doc):
                    text = page.get_text("text").strip()
                    if text:
                        metadata = {
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1
                        }
                        documents.append({
                            "text": text,
                            "metadata": metadata
                        })
                        print(f"✅ Page {page_num+1}: {len(text)} chars")
        except Exception as e:
            print(f"❌ Error in {pdf_path}: {str(e)}")
    
    end_time = time.time()  # 🕒 End Time
    print(f"⏳ PDF Loading Time: {end_time - start_time:.2f} seconds")  # ⬅️ Print time taken

    if not documents:
        raise ValueError("⚠ No text extracted from PDFs!")
    return documents

def create_chunks(extracted_data):
    print("⏳ Creating text chunks...")
    start_time = time.time()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Chunk size ko increase kiya
        chunk_overlap=100
    )

    texts = [doc["text"] for doc in extracted_data]
    metadatas = [doc["metadata"] for doc in extracted_data]
    
    chunks = text_splitter.create_documents(texts, metadatas=metadatas)

    end_time = time.time()
    print(f"✅ Chunking completed in {end_time - start_time:.2f} seconds")  # ⬅️ Print time taken
    return chunks

def store_in_faiss(text_chunks):
    print("⏳ Storing in FAISS...")
    start_time = time.time()

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(
        documents=text_chunks,
        embedding=embedding_model
    )
    db.save_local(DB_FAISS_PATH)

    end_time = time.time()
    print(f"✅ FAISS Vector Store created in {end_time - start_time:.2f} seconds")  # ⬅️ Print time taken

if __name__ == "__main__":
    print("🚀 Starting document processing...")
    
    extracted_data = load_pdf_files(DATA_PATH)  # PDF Load Time Check
    text_chunks = create_chunks(extracted_data)  # Chunking Time Check
    store_in_faiss(text_chunks)  # FAISS Store Time Check
    
    print("🎉 Vector store creation complete!")