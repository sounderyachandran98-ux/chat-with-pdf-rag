import chainlit as cl
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

db = None


@cl.on_chat_start
async def start():

    global db
    db = None

    files = None

    while files is None:

        files = await cl.AskFileMessage(
            content="Please upload a PDF file.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180
        ).send()

    file = files[0]

    await cl.Message(
        content=f"Processing file: {file.name}"
    ).send()

    reader = PdfReader(file.path)

    documents = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            documents.append(
                {
                    "text": text,
                    "source": file.name,
                    "page": page_num + 1
                }
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
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

    await cl.Message(
        content=f"Created {len(texts)} chunks"
    ).send()

    await cl.Message(
        content=f"Building FAISS for {file.name}"
    ).send()

    try:

        db = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )

        await cl.Message(
            content=f"FAISS built for {file.name}"
        ).send()

        await cl.Message(
            content="PDF loaded successfully. Ask your questions."
        ).send()

    except Exception as e:

        await cl.Message(
            content=f"Embedding Error: {str(e)}"
        ).send()

        return


@cl.on_message
async def main(message: cl.Message):

    global db

    if db is None:

        await cl.Message(
            content="Please upload a PDF first."
        ).send()

        return

    question = message.content

    docs = db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not available in the context,
say "I don't have enough information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        answer = str(response.content)
    else:
        answer = str(response)

    citations = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown PDF"
        )

        page = doc.metadata.get(
            "page",
            "Unknown Page"
        )

        citations.append(
            f"{source} | Page {page}"
        )

    citation_text = "\n".join(
        dict.fromkeys(citations)
    )

    final_response = f"""
## Answer

{answer}

---

## Sources

{citation_text}
"""

    await cl.Message(
        content=final_response
    ).send()
