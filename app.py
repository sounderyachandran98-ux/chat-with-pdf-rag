import chainlit as cl
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="""
# 📄 Google Annual Report Assistant

Ask questions about Google's 2025 Annual Report.

Examples:
- What are Google's main sources of revenue?
- What risks does Google mention?
- What AI initiatives are discussed?
- Summarize Google's business model.
"""
    ).send()


@cl.on_message
async def main(message: cl.Message):

    question = message.content

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not available in the context,
say "I don't have enough information in the provided documents."

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

        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "Unknown Page")

        citations.append(
            f"📄 {source} | 📍 Page {page}"
        )

    citation_text = "\n".join(dict.fromkeys(citations))

    final_response = f"""
## 🤖 Answer

{answer}

---

## 📚 Sources

{citation_text}
"""

    await cl.Message(content=final_response).send()