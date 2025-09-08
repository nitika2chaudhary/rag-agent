import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from transformers import pipeline
from typing import List

salary_info = """
Salary is structured into monthly and annual components.
Annual salary is monthly salary times 12
Deductions may include income tax, provident fund, and professional tax.
Net salary is credited monthly after deductions.
"""

insurance_info = """
Insurance benefits include health coverage, accident coverage, and hospitalization.
Premiums are paid monthly or annually.
Claims are submitted with bills and reports via the company/insurer portal.
Coverage typically lists inclusions, exclusions, and claim limits.
"""

documents = [
    Document(page_content=salary_info.strip(), metadata={"topic": "salary"}),
    Document(page_content=insurance_info.strip(), metadata={"topic": "insurance"}),
]

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
split_documents: List[Document] = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = FAISS.from_documents(split_documents, embeddings)

model = pipeline("text2text-generation", model="google/flan-t5-small")

def generate_response(prompt: str, max_tokens: int = 180) -> str:
    output = model(prompt, max_new_tokens=max_tokens, temperature=0.1, do_sample=False)
    return output[0]["generated_text"].strip()

def get_context(topic: str, query: str, top_k: int = 4) -> str:
    results = store.similarity_search(query, k=top_k)
    results = [r for r in results if r.metadata.get("topic") == topic]
    return " ".join([r.page_content for r in results])

def salary_handler(query: str) -> str:
    context = get_context("salary", query)
    if not context:
        return "I can only answer salary-related questions."
    prompt = f"You are an HR assistant. Use ONLY the following context to answer.\nContext:\n{context}\nQuestion: {query}\nAnswer briefly:"
    return generate_response(prompt)

def insurance_handler(query: str) -> str:
    context = get_context("insurance", query)
    if not context:
        return "I can only answer insurance-related questions."
    prompt = f"You are an HR benefits assistant. Use ONLY the following context to answer.\nContext:\n{context}\nQuestion: {query}\nAnswer briefly:"
    return generate_response(prompt)

def route_query(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["salary", "pay", "deduction", "ctc", "net"]):
        return salary_handler(query)
    if any(w in q for w in ["insurance", "policy", "coverage", "premium", "claim"]):
        return insurance_handler(query)
    ctx_salary = get_context("salary", query)
    ctx_ins = get_context("insurance", query)
    if len(ctx_ins) > len(ctx_salary):
        return insurance_handler(query) if ctx_ins else "Please ask about salary or insurance."
    else:
        return salary_handler(query) if ctx_salary else "Please ask about salary or insurance."

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Multi-Agent HR Chatbot")
st.markdown("Ask questions about salary or insurance.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Your Question:", placeholder="e.g., What is included in my insurance?")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Ask") and query:
        answer = route_query(query)
        st.session_state.history.append(("You", query))
        st.session_state.history.append(("Assistant", answer))
with col2:
    if st.button("Run Sample Queries"):
        q1, q2 = "How do I calculate annual salary?", "What is included in my insurance policy?"
        a1, a2 = route_query(q1), route_query(q2)
        sample_text = f"Sample 1:\nQ: {q1}\nA: {a1}\n\nSample 2:\nQ: {q2}\nA: {a2}"
        st.session_state.history.append(("Assistant", sample_text))

if st.button("Clear Chat"):
    st.session_state.history = []

for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")
