from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

pdf_folder = "pdfs"

documents = []

# Read PDFs and store metadata
for filename in os.listdir(pdf_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, filename)

        print("Reading PDF:", filename)

        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages):

            text = page.extract_text()

            if text:

                documents.append(
                    {
                        "text": text,
                        "source": filename,
                        "page": page_num + 1
                    }
                )

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

texts = []
metadatas = []

for doc in documents:

    chunks = splitter.split_text(doc["text"])

    for chunk in chunks:

        texts.append(chunk)

        metadatas.append(
            {
                "source": doc["source"],
                "page": doc["page"]
            }
        )

print("Chunks:", len(texts))

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings loaded!")

# Create FAISS index with metadata
db = FAISS.from_texts(
    texts,
    embeddings,
    metadatas=metadatas
)

# Save index
db.save_local("faiss_index")

print("PDF data stored successfully!")