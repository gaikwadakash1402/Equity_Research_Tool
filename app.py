import os
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

st.title("Equity Research Tool")
st.sidebar.title("Article URLs")

# ========== Configuration ==========
CHUNK_SIZE = 1500  # Increased from 1000
CHUNK_OVERLAP = 200
MODEL_NAME = "TinyLlama:latest"  # Consider "llama3:8b" for faster responses


# ========== Cached Resources ==========
@st.cache_resource
def init_llm():
    try:
        return Ollama(model=MODEL_NAME, temperature=0.3)  # Lower temperature for faster responses
    except Exception as e:
        st.error(f"❌ LLM Error: {e}\nRun: `ollama pull {MODEL_NAME}`")
        st.stop()


@st.cache_resource
def init_embeddings():
    try:
        return OllamaEmbeddings(model="nomic-embed-text", num_gpu=1)  # Enable GPU if available
    except Exception as e:
        st.error(f"❌ Embeddings Error: {e}\nRun: `ollama pull nomic-embed-text`")
        st.stop()


llm = init_llm()
embeddings = init_embeddings()


# ========== Vector Store Management ==========
def get_vector_store(urls):
    with st.status("⚡ Processing Documents...", expanded=True) as status:
        # 1. Load Documents
        st.write("📥 Downloading articles...")
        loader = UnstructuredURLLoader(urls)
        documents = loader.load()

        # 2. Optimized Text Splitting
        st.write("✂️ Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=['\n\n', '\n', '(?<=\. )', ' ']
        )
        chunks = text_splitter.split_documents(documents)

        # 3. Batch Embedding Creation
        st.write("🔨 Creating embeddings (this may take a while)...")
        vector_db = FAISS.from_documents(chunks, embeddings)

        # 4. Save optimized index
        vector_db.save_local("faiss_index")
        status.update(label="✅ Processing Complete!", state="complete")
    return vector_db


# ========== UI Components ==========
urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
process_clicked = st.sidebar.button("🚀 Process URLs")
query = st.text_input("🔍 Ask a question:")

# ========== Core Logic ==========
if process_clicked:
    if any(urls):
        vector_db = get_vector_store(urls)
    else:
        st.error("Please enter at least one URL")

if query:
    if os.path.exists("faiss_index"):
        with st.spinner("💡 Analyzing documents..."):
            # Load pre-processed index
            vector_db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )

            # Optimized QA chain
            qa_chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vector_db.as_retriever(search_kwargs={"k": 3})  # Limit sources
            )

            # Stream response
            response_container = st.empty()
            result = qa_chain({"question": query}, return_only_outputs=True)

            # Display partial results immediately
            response_container.markdown(f"**Answer:**\n{result['answer'][:500]}...")
            st.markdown(f"**Full Answer:**\n{result['answer']}")

            if result.get("sources"):
                st.subheader("🔗 Sources:")
                st.write(result["sources"])
    else:
        st.error("Process URLs first!")